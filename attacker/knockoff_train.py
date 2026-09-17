"""
knockoff_train.py  --  ARIAN.  The OFFLINE extraction step.

No network here. It loads the transfer set written by attack_client.py and
distills a substitute ("knockoff") model f_hat from the victim's soft labels,
using a temperature-scaled distillation loss. The knockoff architecture may
differ from the victim's -- functionality, not weights, is what we copy.

It reports:
  * val fidelity  -- how often f_hat's argmax agrees with the victim's argmax
                     on a held-out slice of the transfer set (offline proxy for
                     true fidelity; a networked re-query gives the exact number).
  * val accuracy  -- against ground-truth labels, only if a labelled test set
                     is available (MNIST test via torchvision).

Run:
    python knockoff_train.py --transfer transfer_set_10k.npz \
        --epochs 20 --T 2.0 --out knockoff_10k.pt
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Knockoff architecture  --  deliberately may differ from the victim's
# --------------------------------------------------------------------------- #

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)                            # logits


# --------------------------------------------------------------------------- #
# Distillation loss (temperature T)
# --------------------------------------------------------------------------- #

def distillation_loss(student_logits, teacher_probs, T: float):
    """
    Soft-label distillation. The victim gives us probabilities, not logits, so
    we recover pseudo-logits (log p) and re-soften them at temperature T -- this
    makes T act on the teacher side too, as in Hinton et al. At T=1 this reduces
    to plain soft-label cross-entropy. The T*T factor keeps gradient magnitudes
    comparable across temperatures.
    """
    eps = 1e-8
    teacher_logits = torch.log(teacher_probs.clamp_min(eps))
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    student_logsoft = F.log_softmax(student_logits / T, dim=1)
    return -(teacher_soft * student_logsoft).sum(dim=1).mean() * (T * T)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_transfer(path):
    z = np.load(path, allow_pickle=True)
    imgs = z["images"].astype(np.float32)          # (N, H, W, 1)
    probs = z["probs"].astype(np.float32)          # (N, K)
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # (N, 1, H, W)
    p = torch.from_numpy(probs)
    meta = {k: z[k].item() if z[k].ndim == 0 else z[k]
            for k in z.files if k not in ("images", "probs", "victim_labels")}
    print(f"[train] loaded {len(x)} pairs from {path}  meta={meta}")
    return x, p


def load_mnist_test(data_dir):
    """Return (x, y) test tensors for accuracy, or None if unavailable."""
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(data_dir, train=False, download=True,
                            transform=transforms.ToTensor())
    except Exception as e:
        print(f"[train] no labelled test set for accuracy ({e})")
        return None
    x = torch.stack([ds[i][0] for i in range(len(ds))])         # (N,1,28,28)
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return x, y


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #

def train(model, x, p, epochs, batch_size, lr, T, val_frac, seed, device):
    torch.manual_seed(seed)
    n = len(x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr = DataLoader(TensorDataset(x[tr_idx], p[tr_idx]),
                    batch_size=batch_size, shuffle=True)
    xv, pv = x[val_idx].to(device), p[val_idx].to(device)
    victim_val_labels = pv.argmax(1)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, pb in tr:
            xb, pb = xb.to(device), pb.to(device)
            opt.zero_grad()
            loss = distillation_loss(model(xb), pb, T)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        # validation fidelity: agreement with the victim on held-out transfer pairs
        model.eval()
        with torch.no_grad():
            fid = (model(xv).argmax(1) == victim_val_labels).float().mean().item()
        print(f"[train] epoch {ep:02d}/{epochs}  "
              f"loss={running / len(tr_idx):.4f}  val_fidelity={fid:.4f}")
    return model


def evaluate_accuracy(model, data_dir, device):
    test = load_mnist_test(data_dir)
    if test is None:
        return None
    x, y = test
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(x), 1000):
            preds.append(model(x[i:i + 1000].to(device)).argmax(1).cpu())
        acc = (torch.cat(preds) == y).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser(description="Distill a knockoff from the transfer set.")
    ap.add_argument("--transfer", default="transfer_set.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="knockoff.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, p = load_transfer(args.transfer)
    model = SmallCNN(num_classes=p.shape[1])

    model = train(model, x, p, args.epochs, args.batch_size, args.lr,
                  args.T, args.val_frac, args.seed, device)

    acc = evaluate_accuracy(model, args.data_dir, device)
    torch.save({"state_dict": model.state_dict(),
                "arch": "SmallCNN", "num_classes": p.shape[1]}, args.out)
    print(f"[train] saved knockoff -> {args.out}")
    if acc is not None:
        print(f"[train] knockoff test accuracy vs ground truth = {acc:.4f}")
    print("[train] note: report accuracy, fidelity (label agreement with the "
          "victim), and adversarial-example transfer in the final report.")


if __name__ == "__main__":
    main()
