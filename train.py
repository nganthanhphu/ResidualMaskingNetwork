import datetime
import os
import random

import cv2
import imgaug
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from models import resmasking_dropout1
from utils.augmenters.augment import seg
from utils.metrics.metrics import accuracy
from utils.radam import RAdam


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FER2013Dataset(Dataset):
    def __init__(
        self,
        stage,
        data_path,
        image_size,
        use_tta=False,
        tta_size=48,
    ):
        self._stage = stage
        self._data_path = data_path
        self._image_size = (image_size, image_size)
        self._use_tta = use_tta
        self._tta_size = tta_size

        csv_path = os.path.join(self._data_path, f"{self._stage}.csv")
        self._data = pd.read_csv(csv_path)

        self._pixels = self._data["pixels"].tolist()
        self._targets = self._data["emotion"].tolist()

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def is_tta(self):
        return self._use_tta

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):
        pixels = self._pixels[idx]
        pixels = list(map(int, pixels.split(" ")))
        image = np.asarray(pixels).reshape(48, 48).astype(np.uint8)

        image = cv2.resize(image, self._image_size)
        image = np.dstack([image] * 3)

        if self._stage == "train":
            image = seg(image=image)

        target = int(self._targets[idx])

        if self._stage == "test" and self._use_tta:
            images = [seg(image=image) for _ in range(self._tta_size)]
            images = [self._transform(tta_image) for tta_image in images]
            return images, target

        image = self._transform(image)
        return image, target


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def build_datasets(data_path, image_size, use_tta, tta_size):
    train_set = FER2013Dataset(
        stage="train",
        data_path=data_path,
        image_size=image_size,
    )
    val_set = FER2013Dataset(
        stage="val",
        data_path=data_path,
        image_size=image_size,
    )
    test_set = FER2013Dataset(
        stage="test",
        data_path=data_path,
        image_size=image_size,
        use_tta=use_tta,
        tta_size=tta_size,
    )
    return train_set, val_set, test_set


def build_dataloaders(train_set, val_set, test_set, batch_size, num_workers, use_tta):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_loader = None
    if not use_tta:
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

    return train_loader, val_loader, test_loader


def visualize_dataset(data_path):
    stages = ("train", "val", "test")
    class_ids = list(FER_2013_EMO_DICT.keys())
    class_labels = [FER_2013_EMO_DICT[class_id] for class_id in class_ids]
    counts = pd.Series(0, index=class_ids, dtype=np.int64)

    for stage in stages:
        csv_path = os.path.join(data_path, f"{stage}.csv")
        df = pd.read_csv(csv_path)
        stage_counts = df["emotion"].value_counts()
        counts = counts.add(stage_counts, fill_value=0).astype(int)

    plt.figure(figsize=(8, 4))
    plt.bar(class_labels, counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Samples")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for images, targets in tqdm(loader, total=len(loader), leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)[0]

        total_loss += loss.item()
        total_acc += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    num_batches = len(loader)
    return total_loss / num_batches, total_acc / num_batches


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, total=len(loader), leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]

            total_loss += loss.item()
            total_acc += acc.item()

    num_batches = len(loader)
    return total_loss / num_batches, total_acc / num_batches


def eval_test_without_tta(model, loader, device):
    model.eval()
    total_acc = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, total=len(loader), leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            acc = accuracy(outputs, targets)[0]
            total_acc += acc.item()

    return total_acc / len(loader)


def eval_test_with_tta(model, dataset, device):
    model.eval()
    total_acc = 0.0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), total=len(dataset), leave=False):
            images, target = dataset[idx]

            images = torch.stack(images, dim=0).to(device, non_blocking=True)
            target = torch.LongTensor([target]).to(device, non_blocking=True)

            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            outputs = torch.sum(outputs, dim=0, keepdim=True)

            acc = accuracy(outputs, target)[0]
            total_acc += acc.item()

    return total_acc / len(dataset)


def save_checkpoint(path, model, train_params, metrics):
    state = {
        "net": model.state_dict(),
        "config": train_params,
        **metrics,
    }
    torch.save(state, path)


def train(
    data_path="data",
    image_size=224,
    lr=1e-4,
    weight_decay=1e-3,
    batch_size=48,
    num_workers=2,
    device_name="cuda:0",
    max_epoch_num=50,
    max_plateau_count=8,
    plateau_patience=2,
    log_dir="log",
    checkpoint_dir="checkpoint",
    seed=1234,
    use_tta=True,
    tta_size=10,
):
    train_params = {
        "data_path": data_path,
        "image_size": image_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device_name": device_name,
        "max_epoch_num": max_epoch_num,
        "max_plateau_count": max_plateau_count,
        "plateau_patience": plateau_patience,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "seed": seed,
        "use_tta": use_tta,
        "tta_size": tta_size,
    }

    set_seed(seed)
    device = resolve_device(device_name)

    cwd = os.getcwd()
    log_root = os.path.join(cwd, log_dir)
    ckpt_root = os.path.join(cwd, checkpoint_dir)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    start_time = datetime.datetime.now().replace(microsecond=0)
    run_name = f"resmasking_dropout1_train_{start_time.strftime('%d%m%Y_%H%M%S')}"

    writer = SummaryWriter(os.path.join(log_root, run_name))
    checkpoint_path = os.path.join(ckpt_root, f"{run_name}.pt")

    print("Start training")
    print(train_params)
    print(f"Device: {device}")

    model = resmasking_dropout1(
        in_channels=3,
        num_classes=7,
    ).to(device)

    train_set, val_set, test_set = build_datasets(
        data_path=data_path,
        image_size=image_size,
        use_tta=use_tta,
        tta_size=tta_size,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tta=use_tta,
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = RAdam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=plateau_patience,
        min_lr=1e-6,
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = 1e9
    best_val_acc = 0.0
    best_train_loss = 1e9
    best_train_acc = 0.0

    plateau_count = 0
    current_epoch = 0

    while current_epoch < max_epoch_num and plateau_count <= max_plateau_count:
        current_epoch += 1

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if current_epoch == 1 or val_acc > best_val_acc:
            plateau_count = 0
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_train_loss = train_loss

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                train_params=train_params,
                metrics={
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "best_train_loss": best_train_loss,
                    "best_train_acc": best_train_acc,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accs": train_accs,
                    "val_accs": val_accs,
                    "current_epoch": current_epoch,
                },
            )
        else:
            plateau_count += 1

        scheduler.step(100.0 - val_acc)

        writer.add_scalar("Accuracy/Train", train_acc, current_epoch)
        writer.add_scalar("Accuracy/Val", val_acc, current_epoch)
        writer.add_scalar("Loss/Train", train_loss, current_epoch)
        writer.add_scalar("Loss/Val", val_loss, current_epoch)

        consume_time = str(datetime.datetime.now() - start_time)
        message = (
            f"E{current_epoch:03d}  "
            f"{train_loss:.3f}/{val_loss:.3f}/{best_val_loss:.3f} "
            f"{train_acc:.3f}/{val_acc:.3f}/{best_val_acc:.3f} "
            f"| p{plateau_count:02d}  Time {consume_time[:-7]}"
        )
        print(message)

    best_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_state["net"])

    if use_tta:
        test_acc = eval_test_with_tta(model, test_set, device)
    else:
        test_acc = eval_test_without_tta(model, test_loader, device)

    best_state["test_acc"] = test_acc
    torch.save(best_state, checkpoint_path)

    consume_time = str(datetime.datetime.now() - start_time)
    writer.add_text(
        "Summary",
        f"Converged after {current_epoch} epochs, consume {consume_time[:-7]}",
    )
    writer.add_text("Results", f"Best validation accuracy: {best_val_acc:.3f}")
    writer.add_text("Results", f"Best training accuracy: {best_train_acc:.3f}")
    writer.add_text("Results", f"Test accuracy: {test_acc:.3f}")
    writer.close()

    print(f"Best checkpoint saved at: {checkpoint_path}")
    print(f"Test accuracy: {test_acc:.3f}")


def eval_test_with_tta_conf_matrix(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    config = checkpoint.get("config")

    data_path = config.get("data_path")
    image_size = config.get("image_size")
    device_name = config.get("device_name")
    use_tta = config.get("use_tta")
    tta_size = config.get("tta_size")

    if not use_tta:
        raise Exception

    device = resolve_device(device_name)

    model = resmasking_dropout1(
        in_channels=3,
        num_classes=7,
    ).to(device)
    model.load_state_dict(checkpoint["net"])
    model.eval()

    _, _, test_set = build_datasets(
        data_path=data_path,
        image_size=image_size,
        use_tta=True,
        tta_size=tta_size,
    )

    y_true = []
    y_pred = []

    total_acc = 0.0
    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, target = test_set[idx]
            images = torch.stack(images, dim=0).to(device, non_blocking=True)
            target = torch.LongTensor([target]).to(device, non_blocking=True)

            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            outputs = torch.sum(outputs, dim=0, keepdim=True)
            pred = torch.argmax(outputs, dim=1).item()

            acc = accuracy(outputs, target)[0]
            total_acc += acc.item()

            y_true.append(int(target))
            y_pred.append(int(pred))

    class_ids = list(FER_2013_EMO_DICT.keys())
    class_labels = [FER_2013_EMO_DICT[class_id] for class_id in class_ids]
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    acc = total_acc / len(test_set) if len(test_set) > 0 else 0.0
    print(f"Test accuracy: {acc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # visualize_dataset(data_path="data")
    # eval_test_with_tta_conf_matrix("checkpoint/branch1.pt")
    train()
