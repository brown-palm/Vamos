import torch
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
import torch.distributed as dist
import contextlib


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, wandb_run, use_vis, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "fp32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
            vqa_loss = model(data, use_vis=use_vis)

        loss = vqa_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        dist.all_reduce(vqa_loss, op=dist.ReduceOp.SUM)
        mean_vqa_loss = vqa_loss / dist.get_world_size()
        if (data_iter_step + 1) % accum_iter == 0 and dist.get_rank() == 0:
            wandb_run.log({"train/vqa_loss": mean_vqa_loss.item(), "train/epoch": epoch, "train/lr": optimizer.param_groups[0]["lr"]}, step=data_iter_step + len(data_loader) * epoch)
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_mixed_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, wandb_run, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / (2*len(data_loader)) + epoch, args)

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "fp32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
            vqa_loss = model(data, use_vis=False)

        loss = vqa_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        dist.all_reduce(vqa_loss, op=dist.ReduceOp.SUM)
        mean_vqa_loss = vqa_loss / dist.get_world_size()
        if (data_iter_step + 1) % accum_iter == 0 and dist.get_rank() == 0:
            wandb_run.log({"train/vqa_loss": mean_vqa_loss.item(), "train/epoch": epoch, "train/lr": optimizer.param_groups[0]["lr"]}, step=data_iter_step + 2 * len(data_loader) * epoch)
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, (data_iter_step+len(data_loader)) / (2*len(data_loader)) + epoch, args)

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "fp32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
            vqa_loss = model(data, use_vis=True)

        loss = vqa_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        dist.all_reduce(vqa_loss, op=dist.ReduceOp.SUM)
        mean_vqa_loss = vqa_loss / dist.get_world_size()
        if (data_iter_step + 1) % accum_iter == 0 and dist.get_rank() == 0:
            wandb_run.log({"train/vqa_loss": mean_vqa_loss.item(), "train/epoch": epoch, "train/lr": optimizer.param_groups[0]["lr"]}, step=data_iter_step + len(data_loader) + 2*len(data_loader) * epoch)
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, use_vis, logits_dict=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        answer = data['answer'].cuda()
        qids = data['qid']
        bsz = answer.shape[0]

        with torch.no_grad():
            logits = model(data, use_vis=use_vis, inference=True)
        
        count = (logits != 0).sum(-1)
        save_logits = logits.sum(-1) / count
        if type(logits_dict) is dict:
            for i in range(bsz):
                logits_dict[qids[i]] = save_logits[i].cpu()
        prediction = save_logits.argmin(-1)

        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
        
        misc.log_qtype(data, eval, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
