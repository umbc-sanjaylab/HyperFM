# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
#import wandb
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, targets = samples['img'], samples['label']
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            # print("Output shape: ", outputs.shape)
            # print("Target shape: ", targets.shape)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            '''
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass
            '''

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch['img'], batch['label']
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            output = model(images)
            loss = criterion(output, target)

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print('* loss {loss:.3f}'.format(loss=metric_logger.meters['loss'].global_avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_mode(data_loader, model, device,mode):

    factor=1
    if mode=='cer':
        factor=30.0
    elif mode=='cth':
        factor=10.0
    elif mode=='all':
        factor={'cer':30.0,'cth':10.0}
    else:
        factor=1.0


    criterion = torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch['img'], batch['label']
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            output = model(images)
            if mode=='all':
                loss_cot = criterion(output[:,0,:,:], target[:,0,:,:])
                loss_cer = criterion(output[:,1,:,:]*factor['cer'], target[:,1,:,:]*factor['cer'])
                loss_cwp = criterion(output[:,2,:,:], target[:,2,:,:])
                loss_cth = criterion(output[:,3,:,:]*factor['cth'], target[:,3,:,:]*factor['cth'])
                loss = loss_cot + loss_cer + loss_cwp + loss_cth

                metric_logger.meters['loss_cot'].update(loss_cot.item())
                metric_logger.meters['loss_cer'].update(loss_cer.item())
                metric_logger.meters['loss_cwp'].update(loss_cwp.item())
                metric_logger.meters['loss_cth'].update(loss_cth.item())
            else:
                loss = criterion(output*factor, target*factor)

                metric_logger.update(loss=loss.item())


    metric_logger.synchronize_between_processes()
    if mode=='all':
        print('* loss_cot {loss_cot:.3f} loss_cer {loss_cer:.3f} loss_cwp {loss_cwp:.3f} loss_cth {loss_cth:.3f}'.format(
            loss_cot=metric_logger.meters['loss_cot'].global_avg,
            loss_cer=metric_logger.meters['loss_cer'].global_avg,
            loss_cwp=metric_logger.meters['loss_cwp'].global_avg,
            loss_cth=metric_logger.meters['loss_cth'].global_avg))
    else:
        print('* loss {loss:.3f}'.format(loss=metric_logger.meters['loss'].global_avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
