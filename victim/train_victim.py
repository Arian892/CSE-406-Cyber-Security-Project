

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import VictimCNN

SEED = 1337


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mnist(data_dir: str, train: bool):
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    return datasets.MNIST(data_dir, train=train, download=True, transform=tf)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train(args) -> float:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = load_mnist(args.data_dir, train=True)
    test_ds = load_mnist(args.data_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = VictimCNN(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        train_loss = running / len(train_ds)
        test_acc = evaluate(model, test_loader, device)
        print(f"[victim-train] epoch {epoch:02d}/{args.epochs}  "
              f"loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    final_acc = evaluate(model, test_loader, device)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "VictimCNN",
        "num_classes": 10,
        "seed": args.seed,
        "test_accuracy": final_acc,
    }, args.out)
    print(f"[victim-train] saved {args.out}  (test accuracy = {final_acc:.4f})")
    return final_acc


def main():
    ap = argparse.ArgumentParser(description="Train the secret victim classifier on MNIST.")
    ap.add_argument("--data-dir", default="../data",
                     help="torchvision cache dir (shared with the attacker's pool)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="victim.pt")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
