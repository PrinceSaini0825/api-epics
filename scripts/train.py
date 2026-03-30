"""
train.py – Train ModelA, ModelB, and the Hybrid Fusion model.

Usage:
    python scripts/train.py --data_dir /path/to/fire_dataset --output_dir ./weights

Dataset layout expected by torchvision.datasets.ImageFolder:
    fire_dataset/
        fire_images/      <- class 0 or 1 depending on sort order
        non_fire_images/
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import ModelA, ModelB, HybridFireDetector

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_A   = 10
EPOCHS_B   = 15
EPOCHS_H   = 20
LR         = 1e-4
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += ((out > 0.5).float() == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        out    = model(imgs)
        loss   = criterion(out, labels)
        total_loss += loss.item()
        correct    += ((out > 0.5).float() == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def train_model(model, train_loader, val_loader, epochs, lr, save_path, tag):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val  = float("inf")
    best_wts  = None

    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"[{tag}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tl:.4f} Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f} Acc: {va:.4f}")
        if vl < best_val:
            best_val = vl
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, save_path)
            print(f"  ✅ Best model saved → {save_path}")

    model.load_state_dict(best_wts)
    return history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Fire Detection Models")
    parser.add_argument("--data_dir",   required=True, help="Path to fire_dataset folder")
    parser.add_argument("--output_dir", default="./weights", help="Where to save .pth files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transform)
    class_names  = full_dataset.classes
    n            = len(full_dataset)
    train_size   = int(0.70 * n)
    val_size     = int(0.15 * n)
    test_size    = n - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply clean transform to val/test (no augmentation leakage)
    val_ds_clean = copy.deepcopy(full_dataset); val_ds_clean.transform = test_transform
    test_ds_clean = copy.deepcopy(full_dataset); test_ds_clean.transform = test_transform
    val_ds.dataset  = val_ds_clean
    test_ds.dataset = test_ds_clean

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes : {class_names}")
    print(f"Train   : {train_size} | Val: {val_size} | Test: {test_size}")

    # ── Train Model A ─────────────────────────────────────────────────────────
    model_a = ModelA().to(DEVICE)
    train_model(model_a, train_loader, val_loader, EPOCHS_A, LR,
                os.path.join(args.output_dir, "best_ModelA.pth"), "Model A")

    # ── Train Model B ─────────────────────────────────────────────────────────
    model_b = ModelB().to(DEVICE)
    train_model(model_b, train_loader, val_loader, EPOCHS_B, LR,
                os.path.join(args.output_dir, "best_ModelB.pth"), "Model B")

    # ── Train Hybrid ──────────────────────────────────────────────────────────
    hybrid = HybridFireDetector().to(DEVICE)
    hybrid.load_backbones(
        os.path.join(args.output_dir, "best_ModelA.pth"),
        os.path.join(args.output_dir, "best_ModelB.pth"),
        DEVICE,
    )
    train_model(hybrid, train_loader, val_loader, EPOCHS_H, LR * 0.5,  # lower LR for fine-tuning
                os.path.join(args.output_dir, "best_Hybrid.pth"), "Hybrid")

    print("\nAll models trained and saved!")


if __name__ == "__main__":
    main()
