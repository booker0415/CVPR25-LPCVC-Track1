import os


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


train_transform = transforms.Compose([

    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandAugment(
        num_ops=2,  # ????2???
        magnitude=9,  # ?????0-10?
        num_magnitude_bins=11,  # ?????
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),  # 转换为Tensor
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),  # ?? Cutout
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正则化
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    # transforms.CenterCrop(224),  # 中心裁剪为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正则化
])


class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # 自定义类别顺序
        classes = [
            "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat",
            "Traffic Light", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog",
            "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Backpack", "Umbrella",
            "Handbag", "Tie", "Skis", "Sports Ball", "Kite", "Tennis Racket", "Bottle",
            "Wine Glass", "Cup", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Orange",
            "Broccoli", "Hot Dog", "Pizza", "Donut", "Chair", "Couch", "Potted Plant",
            "Bed", "Dining Table", "Toilet", "TV", "Laptop", "Mouse", "Remote",
            "Keyboard", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink",
            "Refrigerator", "Book", "Clock", "Vase", "Teddy Bear", "Hair Dryer"]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class DataManager:
    """数据管理类，封装数据集创建和数据加载逻辑"""

    def __init__(self, dataset_path='data', BATCH_SIZE=32, NUM_WORKERS=4):
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS
        self.dataset_path = dataset_path

    def _create_datasets(self):
        """创建训练集和验证集"""

        # 数据集路径
        # 训练集和验证集
        train_path = os.path.join(self.dataset_path, 'trainset')
        val_path = os.path.join(self.dataset_path, 'valset')
        train_dataset = CustomImageFolder(train_path, transform=train_transform)
        val_dataset = CustomImageFolder(val_path, transform=eval_transform)

        return train_dataset, val_dataset

    def create_dataloaders(self):
        """创建数据加载器"""
        train_dataset, val_dataset = self._create_datasets()
        n_trainset = len(train_dataset)
        n_valset = len(val_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            # num_workers=0,
            # pin_memory=False,
            # persistent_workers=False
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            # num_workers=0,
            # pin_memory=False,
            # persistent_workers=False
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

        return train_loader, val_loader, len(train_dataset.classes), n_trainset, n_valset


