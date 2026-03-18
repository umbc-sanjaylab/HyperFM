# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# SatMAEpp: https://github.com/techmn/satmae_pp
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
#import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory
from timm.optim import create_optimizer_v2

import util.misc as misc
from util.datasets import build_dataset, build_val_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch, evaluate
from util.spectral_grouping import make_band_groups, make_linear_groups

def get_args_parser():
    parser = argparse.ArgumentParser('HyperFM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='hyperFM', choices=['hyperFM'], help='Define model')
    parser.add_argument('--model', default='hyperFM_enc4', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--tt_rank', default=3, type=int, help='Rank of hypoformer decomposition matrix')
    parser.add_argument('--input_size', default=96, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--spatial_mask', action='store_true', help='Use spatial masking only')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use normalized pixels as targets for loss calculation')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.0001, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    # Dataset parameters
    parser.add_argument('--csv_path', default=None, type=str, help='Train list')
    parser.add_argument('--data_root_dir', default=None, type=str, help='Root directory of the dataset')
    parser.add_argument('--train_path', default='dataset/aux/train.csv', type=str, help='Train .csv path')
    parser.add_argument('--val_path', default='dataset/aux/val.csv', type=str, help='val.csv path')
    parser.add_argument('--test_path', default='dataset/aux/test.csv', type=str, help='test.csv path')
    parser.add_argument('--dataset_type', default='pace', choices=['pace'],
                        help='Whether to use pace, or other dataset.')
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC mae")

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--wandb', type=str, default=None,help="Wandb project name, eg: sentinel_pretrain")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    # parser.add_argument('--local-rank', default=os.getenv('LOCAL_RANK', 0), type=int)  # prev default was -1
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
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


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
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
    # define the model
    if args.model_type == 'hyperFM':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = make_band_groups()

        import model_arch.models_mae_hyperFM as models_mae_hyperFM
        model = models_mae_hyperFM.__dict__[args.model](img_size=args.input_size,
                                                               patch_size=args.patch_size,
                                                               in_chans=291,
                                                               channel_groups=args.grouped_bands,
                                                               tt_rank = args.tt_rank,
                                                               spatial_mask=args.spatial_mask,
                                                               norm_pix_loss=args.norm_pix_loss)
    else::
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=False)
        model_without_ddp = model.module

   
    optimizer = create_optimizer_v2(model_or_params=model_without_ddp, opt='adamw',
        lr=args.lr, weight_decay=args.weight_decay,betas=(0.9, 0.95)
    )

    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    '''
    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)
    '''

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
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


        test_stats = evaluate(data_loader_test, model, device)

        if log_writer is not None:
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
