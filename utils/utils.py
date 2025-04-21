import os
import time
import math
import torch
import random
import numpy as np
import shutil
from torch import Tensor
from typing import Optional


def time_since(since):
    # ????
    s = time.time() - since
    h = math.floor(s / 3600)
    m = math.floor((s - h * 3600) / 60)
    s = s - (m * 60 + h * 3600)
    return '%dh %dm %ds' % (h, m, s)


def log_config(log_path, model_name, model_pt, n_class, BATCH_SIZE, LR, EPOCHS, data_path):
    # ---------------------- ?????? ----------------------

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_path = os.path.join(log_path, 'training_log_{}.txt'.format(model_name))
    log_file = open(log_file_path, 'a')
    # ????????
    log_file.write('-' * 100 + '\n')
    log_file.write('Time: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    log_file.write('Model: {} (pretrained={}, fc replaced for {} classes)\n'.format(model_name, model_pt, n_class))
    log_file.write('BATCH_SIZE: {} | LR: {} | EPOCHS: {}\n'.format(BATCH_SIZE, LR, EPOCHS))
    log_file.write('Dataset Path: {}\n'.format(data_path))
    log_file.write(f'CUDA Device Name: {torch.cuda.get_device_name(0)} \n')
    log_file.write('-' * 100 + '\n')
    # -----------------------------------------------------------
    return log_file


def save_checkpoint(model, is_best, checkpoint_path, model_name, lr, batch_size):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # ????
    # script_model = torch.jit.script(model)
    example_input = torch.randn(1, 3, 224, 224).cuda()
    traced_model = torch.jit.trace(model, example_input)
    filename = os.path.join(checkpoint_path, model_name + '_lr_{}_bs_{}_last.pt'.format(lr, batch_size))
    traced_model.save(filename)
    # torch.jit.save(script_model,filename)
    # torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_path, model_name + '_lr_{}_bs_{}_best.pt'.format(lr, batch_size)))


def save_checkpoint_wiht_pth(model, is_best, checkpoint_path, model_name, lr, batch_size):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    filename = os.path.join(checkpoint_path, model_name + '_lr_{}_bs_{}_last.pth'.format(lr, batch_size))
    torch.save(model, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_path, model_name + '_lr_{}_bs_{}_best.pth'.format(lr, batch_size)))


def setup_seed(seed):
    # ????????????????
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def save_checkpoint_with_edge(
        model: torch.nn.Module,
        is_best: bool,
        checkpoint_path: str,
        model_name: str,
        lr: float,
        batch_size: int,
        example_input: Optional[Tensor] = None,  # ???Trace?????????
        input_channels: int = 3,  # ???????
        input_size: int = 224  # ??????
):
    """?????????? TorchScript ??

    Args:
        model: ??????
        is_best: ?????????
        checkpoint_path: ???????
        model_name: ??????
        lr: ??? (?????)
        batch_size: ???? (?????)
        example_input: ?????????????
        input_channels: ????? (???????????)
        input_size: ???? (???????????)
    """
    # ??????
    os.makedirs(checkpoint_path, exist_ok=True)

    # ?????????
    filename_template = f"{model_name}_lr_{lr}_bs_{batch_size}"
    last_model_path = os.path.join(checkpoint_path, f"{filename_template}_last.pt")
    best_model_path = os.path.join(checkpoint_path, f"{filename_template}_best.pt")

    try:
        # ?????? (????/DDP??)
        device = next(model.parameters()).device

        # ???????????????
        if example_input is None:
            example_input = torch.randn(
                1, input_channels, input_size, input_size
            ).to(device)
            print(f"Generated example input: {example_input.shape}")

        # ????????????
        model.eval()

        # ??1: ???? Tracing ?? (???????)
        traced_model = torch.jit.trace(model, example_input)
        torch.jit.save(traced_model, last_model_path)
        print(f"Traced model saved to {last_model_path}")

        # ??2: ?? Scripting ?? (????????)
        script_model = torch.jit.script(model)
        script_path = os.path.join(checkpoint_path, f"{filename_template}_script.pt")
        torch.jit.save(script_model, script_path)
        print(f"Scripted model saved to {script_path}")

        # ???????? (??????)
        torch.save({
            'model_state_dict': model.state_dict(),
            'example_input': example_input  # ????????????
        }, os.path.join(checkpoint_path, f"{filename_template}_full.pth"))

        # ??????
        if is_best:
            shutil.copyfile(last_model_path, best_model_path)
            print(f"New best model copied to {best_model_path}")

    except RuntimeError as e:
        # ?? TorchScript ????
        if "Expected a value of type 'Tensor' for argument 'input'" in str(e):
            print("\n[Critical] TorchScript ???????:")
            print("?????")
            print("1. ??????? None ??? -> ???????? Tensor")
            print("2. ???????? -> ?? tracing ???? scripting")
            print("3. ?????? -> ???????? Tensor ????")
            print("???????")
            print(f"?????? traced ?? ({last_model_path}) ????")
        raise e

    except Exception as e:
        print(f"??????: {str(e)}")
        raise


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def run_eval(model, data_loader, device, chrono, logger, step):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1, all_top5 = [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits = model(x)
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1, top5 = topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
                f"top1 {np.mean(all_top1):.2%}, "
                f"top5 {np.mean(all_top5):.2%}")
    logger.flush()
    return all_c, all_top1, all_top5