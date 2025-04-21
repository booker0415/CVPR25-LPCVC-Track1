import os
import torch
import yaml  # 新增配置加载
import time
from torch import nn, optim


from utils.ema import ExponentialMovingAverage
from utils import utils
import data_loader
from utils.bypass_bn import disable_running_stats,enable_running_stats
from utils.smooth_crossentropy import smooth_crossentropy
from utils.sam import SAM

# 加载配置文件
from utils.train import evaluate

config_path = 'config_sam.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ??????
utils.setup_seed(config['seed'])

# ????
BATCH_SIZE = config['batch_size']
LR = config['lr']
EPOCHS = config['epochs']
NUM_WORKERS = config['num_workers']
MODEL_NAMES = config['model']
RHO = config['rho']
MOMENTUM = config['momentum']
WEIGHT_DECAY = config['weight_decay']
Date = config['date']
data_path = config['data_path']
USE_EMA = config['ema']

checkpoint_path = f'./checkpoint/{Date}/{MODEL_NAMES}'
log_path = f'./log/{Date}/'


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the data
    data_manager = data_loader.DataManager(dataset_path=data_path, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS)
    train_loader, val_loader, n_class, n_trainset, n_valset = data_manager.create_dataloaders()
    model = timm.create_model('fastvit_mci1', pretrained=True, num_classes=64)
    # model = torch.load('./checkpoint/pretrained_weights/fastvit_mci1.pth',map_location='cpu')
    # new_dropout_value = 0.3  # ????????
    # model.head.drop.p = new_dropout_value

    model.to(device)
    model_name = model.__class__.__name__
    model_name = model_name.lower()
    model_pt = any(param.requires_grad for _, param in model.named_parameters())


    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    base_optimizer = optim.AdamW
    optimizer = SAM([

        {'params': backbone_params, 'lr': LR*0.1},  # ?????????
        {'params':  head_params, 'lr': LR},  # ???????
    ], base_optimizer, rho=RHO, adaptive=True, weight_decay=WEIGHT_DECAY)
    warmup_epochs = 5  # ??epoch????????????
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0001,  # ?0??????
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - warmup_epochs,  # ?????
        eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # EMA module
    adjust = 1 * BATCH_SIZE * 32 / EPOCHS
    alpha = 1.0 - 0.99998
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
    # Log the results
    log_file = utils.log_config(log_path, MODEL_NAMES, model_pt, n_class, BATCH_SIZE, LR, EPOCHS, data_path)
    # EMA module
    best_val_acc = 0.0
    best_val_loss = 0.0


    best_acc = 0.0
    best_ema = 0.0
    start = time.time()
    for epoch in range(EPOCHS):
        epoch_start = time.time()  # ????epoch????
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for idx, (images, labels) in enumerate(train_loader, start=0):
            images = images.to(device)
            labels = labels.to(device)
            enable_running_stats(model)
            # optimizer.zero_grad()
            outputs = model(images)
            loss = smooth_crossentropy(outputs, labels, smoothing=0.1)
            loss = loss.mean()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            smooth_crossentropy(model(images), labels, smoothing=0.1).mean().backward()
            optimizer.second_step(zero_grad=True)
            #
            # loss.backward()
            if USE_EMA:
                if idx % 32 == 0:
                    model_ema.update_parameters(model)
                    # if epoch < args.lr_warmup_epochs:
                    # # Reset ema buffer to keep copying weights during warmup period
                    #     model_ema.n_averaged.fill_(0)
                    # ??EMA????????
                    # ema.update()
            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            # train_loss += loss.item()
            train_loss += loss.item()
            # scheduler(epoch)

            rate = (idx + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        train_acc = train_acc / n_trainset
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, n_valset)
        if USE_EMA:
            ema_loss, ema_acc = evaluate(model_ema, val_loader, criterion, device, n_valset)
        epoch_time = f'{utils.time_since(epoch_start)}'
        # Log the results
        epoch_log = '[{:03d}/{:03d}]  Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f}  Loss: {:3.6f} | Epoch Time: {}'.format(
            epoch + 1, EPOCHS, train_acc, train_loss / len(train_loader), val_acc, val_loss / len(val_loader),
            epoch_time)
        print(epoch_log)
        if USE_EMA:
            ema_epoch_log = '[{:03d}/{:03d}]  EMA Acc: {:3.6f} Loss: {:3.6f} '.format(
                epoch + 1, EPOCHS, ema_acc, ema_loss / len(val_loader))
        log_file.write(epoch_log + "\n")
        if USE_EMA:
            log_file.write(ema_epoch_log + "\n")
        log_file.flush()
        is_best = val_acc > best_acc
        # Save weights
        utils.save_checkpoint_wiht_pth(model=model,
                                       is_best=is_best,
                                       checkpoint_path=checkpoint_path,
                                       model_name=MODEL_NAMES,
                                       lr=LR,
                                       batch_size=BATCH_SIZE)

        if USE_EMA:
            if ema_acc > best_ema:
                best_ema = ema_acc
                filename = os.path.join(checkpoint_path,
                                        MODEL_NAMES + '_lr_{}_bs_{}_best_ema_acc.pth'.format(LR, BATCH_SIZE,
                                                                                             epoch))
                torch.save(model_ema.state_dict(), filename)

        # If best_eval, best_save_path
        if is_best:
            best_acc = val_acc
            no_improve = 0
            print('best_acc:', best_acc)

        if epoch % 1 == 0:
            filename = os.path.join(checkpoint_path,
                                    MODEL_NAMES + '_lr_{}_bs_{}_@epoch{}.pth'.format(LR, BATCH_SIZE,
                                                                                                   epoch))
            torch.save(model, filename)
    # 训练结束，记录训练时间
    total_time = utils.time_since(start)
    log_file.write("checkpint saved at best val_acc: {}\n".format(best_acc))
    if USE_EMA:
        log_file.write("checkpint saved at best ema_acc: {}\n".format(best_ema))
    final_log = "Finished Training. Total time: {}\n".format(total_time)
    log_file.write(final_log)
    log_file.write('\n')
    log_file.close()
    print(f'Run_time: {total_time}')
    print("Finished Training")


