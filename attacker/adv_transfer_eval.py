

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "victim"))

from model import load_victim_model          # victim/model.py
from knockoff_train import SmallCNN          # attacker/knockoff_train.py


def load_holdout(data_dir: str, split: str, n: int, seed: int):
    ds = datasets.MNIST(data_dir, train=(split == "train"), download=True,
                         transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    n = min(n, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    x = torch.stack([ds[int(i)][0] for i in idx])
    y = torch.tensor([ds[int(i)][1] for i in idx])
    return x, y


def load_knockoff(path: str) -> SmallCNN:
    ck = torch.load(path, map_location="cpu")
    m = SmallCNN(num_classes=ck.get("num_classes", 10))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def fgsm(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """One-step sign-of-gradient adversarial perturbation (Goodfellow et al. 2015)."""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


@torch.no_grad()
def batched_argmax(model: torch.nn.Module, x: torch.Tensor, batch: int = 1000) -> torch.Tensor:
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(out)


def evaluate(knockoff: SmallCNN, victim: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             eps_list, batch: int = 500) -> dict:
    v_pred = batched_argmax(victim, x)
    k_pred = batched_argmax(knockoff, x)

    result = {
        "n_samples": len(x),
        "victim_accuracy": (v_pred == y).float().mean().item(),
        "knockoff_accuracy": (k_pred == y).float().mean().item(),
        "knockoff_fidelity_vs_victim": (v_pred == k_pred).float().mean().item(),
        "adversarial_transfer": {},
    }
    for eps in eps_list:
        fooled_knock = transferred = total = 0
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            x_adv = fgsm(knockoff, xb, yb, eps)
            with torch.no_grad():
                k_adv = knockoff(x_adv).argmax(1)
                v_adv = victim(x_adv).argmax(1)
                v_clean = victim(xb).argmax(1)
            fooled_knock += (k_adv != yb).sum().item()
            transferred += (v_adv != v_clean).sum().item()
            total += len(xb)
        result["adversarial_transfer"][eps] = {
            "white_box_fool_rate_on_knockoff": fooled_knock / total,
            "transfer_fool_rate_on_victim": transferred / total,
        }
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Offline: fidelity, accuracy, and FGSM adversarial transfer for a knockoff.")
    ap.add_argument("--knockoff", required=True, help="checkpoint from knockoff_train.py")
    ap.add_argument("--victim", default="../victim/victim.pt", help="victim checkpoint")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                     help="holdout split; 'train' is disjoint from the --pool mnist query pool")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None, help="optional .json path to save the result")
    args = ap.parse_args()

    victim = load_victim_model(args.victim, num_classes=10, seed=args.seed)
    victim.eval()
    knockoff = load_knockoff(args.knockoff)

    x, y = load_holdout(args.data_dir, args.split, args.n_samples, args.seed)
    result = evaluate(knockoff, victim, x, y, args.eps)

    print(f"[adv-transfer] {args.knockoff}  ({args.split} split, n={result['n_samples']})")
    print(f"[adv-transfer] victim accuracy    = {result['victim_accuracy']:.4f}")
    print(f"[adv-transfer] knockoff accuracy  = {result['knockoff_accuracy']:.4f}")
    print(f"[adv-transfer] fidelity vs victim = {result['knockoff_fidelity_vs_victim']:.4f}")
    for eps, v in result["adversarial_transfer"].items():
        print(f"[adv-transfer] eps={eps}: white-box fool rate on knockoff = "
              f"{v['white_box_fool_rate_on_knockoff']:.4f}, transfer fool rate on victim = "
              f"{v['transfer_fool_rate_on_victim']:.4f}")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[adv-transfer] wrote {args.out}")


if __name__ == "__main__":
    main()
