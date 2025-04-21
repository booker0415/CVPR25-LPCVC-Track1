import torch
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights


from torchvision import models
from data_loader import DataManager
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")



EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 0.0001
checkpoint_path = 'E:/torch test/LPCV/tuili/checkpoint/mbv3'
data_path = 'E:/torch test/LPCV/tuili/cocoset_split'
log_path = 'E:/torch test/LPCV/tuili/log'

def train(model, train_loader, optimizer, criterion, device, n_trainset):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for idx, (images, labels) in enumerate(train_loader, start=0):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        train_acc += (train_pred == labels).sum().item()
        train_loss += loss.item()

        rate = (idx + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    return train_loss, train_acc / n_trainset


def evaluate(model, val_loader, criterion, device, n_valset):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            outputs = model(val_images)
            preds = torch.max(outputs, 1)[1]
            loss1 = criterion(outputs, val_labels)

            val_acc += (preds == val_labels).sum().item()
            val_loss += loss1.item()

    val_acc = val_acc / n_valset
    return val_loss, val_acc

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    # Load the model
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    model_name = model.__class__.__name__
    model_name = model_name.lower()

    finetune_params = [name for name, param in model.named_parameters() if param.requires_grad]
    if finetune_params:
        model_pt = True
    else:
        model_pt = False
    # Load the data
    data_manager = DataManager(dataset_path=data_path, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS)
    train_loader, val_loader, n_class, n_trainset, n_valset = data_manager.create_dataloaders()
    # mobilenetv3
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_class)
    optimizer = optim.Adam(model.classifier[3].parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    log_file = log_config(log_path, model_name, model_pt, n_class, BATCH_SIZE, LR, EPOCHS, data_path)
    # # ---------------------- 日志记录设置 ----------------------
    #
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # log_file_path = os.path.join(log_path, 'training_log_{}.txt'.format(model_name))
    # log_file = open(log_file_path, 'a')
    # # 写入初始配置信息
    # log_file.write('-' * 50 + '\n')
    # log_file.write('Time: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    # log_file.write('Model: {} (pretrained={}, fc replaced for {} classes)\n'.format(model_name, model_pt, n_class))
    # log_file.write('BATCH_SIZE: {} | LR: {} | EPOCHS: {}\n'.format(BATCH_SIZE, LR, EPOCHS))
    # log_file.write('Dataset Path: {}\n'.format(data_path))
    # log_file.write('--------------------------------------------------\n')
    # # -----------------------------------------------------------
    best_acc = 0.0
    start = time.time()
    for epoch in range(EPOCHS):
        # Train the model
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, n_trainset)

        # Evaluate the model
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, n_valset)

        # Log the results
        epoch_log = '[{:03d}/{:03d}]  Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f}  Loss: {:3.6f}'.format(
            epoch + 1, EPOCHS, train_acc, train_loss / len(train_loader), val_acc,val_loss / len(val_loader))
        print(epoch_log)
        log_file.write(epoch_log + "\n")
        log_file.flush()

        is_best = val_acc > best_acc

        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint_path=checkpoint_path,
                              model_name=model_name,
                              lr=LR,
                              batch_size=BATCH_SIZE)
        # If best_eval, best_save_path
        if is_best:
            best_acc = val_acc
            print('best_acc:', best_acc)
        # # Optional: Save the model with the best validation accuracy
        # if val_acc > best_acc:
        #     is_best = True
        #     best_acc = val_acc
        #     # torch.save(model.state_dict(), save_path)  # Save model weights
        #     script_model = torch.jit.script(model)
        #     if not os.path.exists(checkpoint_path):
        #         os.makedirs(checkpoint_path)
        #
        #     script_model.save(os.path.join(checkpoint_path, 'model_{}_best_with_acc{:.3f}.pt'.format(model_name, best_acc)))
        #     # torch.save(model.state_dict(), save_path)
        #     print('saving model with acc {:.3f}'.format(best_acc))
        # else:
        #     script_model = torch.jit.script(model)
        #     script_model.save(os.path.join(checkpoint_path, 'model_{}_last.pt'.format(model_name)))
    total_time = time_since(start)
    final_log = "Finished Training. Total time: {}\n".format(total_time)
    log_file.write(final_log)
    log_file.write('\n')
    log_file.close()
    print(f'Run_time: {total_time}')
    print("Finished Training")