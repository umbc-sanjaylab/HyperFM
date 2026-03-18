# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
#import wandb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.3.2" 
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets_finetune import build_dataset,build_val_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import (train_one_epoch, evaluate, evaluate_mode)
from util.spectral_grouping import make_band_groups, make_linear_groups

def get_args_parser():
    parser = argparse.ArgumentParser('HyperFm Finetuning', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='hyperFM_conv', choices=['hyperFM_conv'],
                        help='Use channel model')
    parser.add_argument('--model', default='hyperFM_enc4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tt_rank', default=3, type=int, help='Rank of hypoformer decomposition matrix')
    parser.add_argument('--input_size', default=96, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')


    # * Finetuning params
    parser.add_argument('--finetune_dec_only', action='store_true', help='Finetune decoder only')
    parser.add_argument('--finetune', default='./output_dir/checkpoint-50.pth', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_root_dir', default='preprocessed_data', type=str, help='dataset folder path') 
    parser.add_argument('--train_path', default='train.csv', type=str,
                        help='Train .csv path')
    parser.add_argument('--val_path', default='val.csv', type=str,
                        help='Test .csv path')
    parser.add_argument('--test_path', default='test.csv', type=str,
                        help='Test .csv path')
    parser.add_argument('--data_mode', default='cot', choices=['cot', 'cer', 'cth', 'cwp', 'cmask','all'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--dataset_type', default='pace', choices=['pace'],
                        help='Whether to use pace, or other dataset.')
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC mae")

    parser.add_argument('--nb_classes', default=4, type=int, help='number of the regression tasks') # 4 clases for COT, CER, CWP, CTH
    parser.add_argument('--output_dir', default='./finetune_logs', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./finetune_logs', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=1, help='How frequently (in epochs) to save ckpt')
    parser.add_argument('--wandb', type=str, default=None, help="Wandb project name, eg: sentinel_finetune")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_checkpoint', default='./fine_dir/checkpoint-50.pth', help='eval checkpoint')

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (in epochs)')
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_val_dataset(is_train=False, args=args)
    dataset_test = build_dataset(is_train=False, args=args)

    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    # Define the model
    if args.model_type == 'hyperFM_conv':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = make_band_groups()

        print(f"Grouping bands {args.grouped_bands}")

        import model_arch.models_hyperFM_conv as HyperFMConv
        model = HyperFMConv.__dict__[args.model](
        patch_size=args.patch_size, img_size=args.input_size, in_chans=291,
        channel_groups=args.grouped_bands, decoder_ch=args.nb_classes
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")


    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu',weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
    elif args.eval:
        checkpoint = torch.load(args.eval_checkpoint, map_location='cpu',weights_only=False)
        print("Load eval checkpoint from: %s" % args.eval_checkpoint)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict() 
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    if args.finetune_dec_only:
        # freeze everything except decoder
        model.freeze_encoder_update_decoder()
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    if args.finetune_dec_only:   
        # finetune decoder only (ignore LRD here) 
        decoder_params = model_without_ddp.decoder_parameters()
        optimizer = torch.optim.AdamW(decoder_params, lr=args.lr, weight_decay=args.weight_decay)

    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    '''
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)
    '''

    if args.eval:
        test_stats = evaluate_mode(data_loader_test, model, device,args.data_mode)
        if args.data_mode == 'all':
            print(f"Evaluation on {len(dataset_test)} test images - MSE: {test_stats['loss_cot']:.4f}")
            print(f"Evaluation on {len(dataset_test)} test images - MSE: {test_stats['loss_cer']:.4f}")
            print(f"Evaluation on {len(dataset_test)} test images - MSE: {test_stats['loss_cwp']:.4f}")
            print(f"Evaluation on {len(dataset_test)} test images - MSE: {test_stats['loss_cth']:.4f}")
        else:
            print(f"Evaluation on {len(dataset_test)} test images - MSE: {test_stats['loss']:.4f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, 
            log_writer=log_writer,
            args=args
        )

        # --- Validation and early stopping ---
        val_stats = evaluate(data_loader_val, model, device)
        val_loss = val_stats.get('loss', None)
        if val_loss is not None:
            print(f"Validation loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print("Validation loss improved — saving model.")
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    break

        # Turn it ON when finetuing on ALL DATSET
        # if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate_mode(data_loader_test, model, device,args.data_mode)

        if log_writer is not None:
            if args.data_mode == 'all':
                log_writer.add_scalar('perf/test_cot_loss', test_stats['loss_cot'], epoch)
                log_writer.add_scalar('perf/test_cer_loss', test_stats['loss_cer'], epoch)
                log_writer.add_scalar('perf/test_cwp_loss', test_stats['loss_cwp'], epoch)
                log_writer.add_scalar('perf/test_cth_loss', test_stats['loss_cth'], epoch)
            else:
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
        # --- Logging ---
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }


        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            '''
            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print(f"Invalid stats?")
            '''

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    torch.distributed.destroy_process_group()

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        # Update the output_dir
        # Get current timestamp as folder-safe string
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Use it to create a new folder path
        folder_name = f"./outputs/{timestamp}"
        args.output_dir = args.output_dir + "/"+folder_name
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)