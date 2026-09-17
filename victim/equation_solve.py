"""
equation_solve.py  --  PRITU.  Attack B: functionally-equivalent extraction
of a LINEAR/softmax victim via least squares.

Design report, Section 4.3 (component E) and Table (Attack B, row):
    "For a linear/softmax victim, logit_i(x) = w_i . x + b_i; querying
     d+1 independent inputs and reading back the confidences gives a
     solvable linear system per class, recovering w_i, b_i by least
     squares (Attack B, functionally equivalent)."
This mirrors the equation-solving attack of Tramer et al., "Stealing
Machine Learning Models via Prediction APIs" (Sec. 4.1): for a genuine
softmax model, log p_i(x) - log p_0(x) = (w_i - w_0).x + (b_i - b_0) is
EXACTLY linear in x, so no gradient descent or training loop is needed --
one batch of non-adaptive queries plus one ordinary-least-squares solve
recovers the model, up to the softmax's inherent per-reference-class
gauge freedom (fixing w_0 = 0, b_0 = 0 without loss of generality).

This file deliberately demonstrates BOTH sides of the report's honest
evaluation:
  * --target linear  -- a genuine softmax victim -> near-exact
                         functional equivalence after only (K-1)*(d+1)
                         queries, no training.
  * --target cnn      -- the real trained VictimCNN (victim.pt) -- the
                         attack's linearity assumption breaks and
                         fidelity collapses. This is exactly why the
                         report calls Attack B a *special case* and
                         leaves the general-purpose CNN extraction to
                         the learning-based attack in attacker/.

Two oracle backends, selected by --mode:
  * offline  -- in-process, no network (fast, deterministic)
  * network  -- real REQUEST/RESPONSE round trips via protocol.py
                against victim_server.py (or mock_victim.py) -- genuine
                black-box access: inputs in, probabilities out, no
                weights or gradients used.

Run:
    # Sanity check against a synthetic linear/softmax victim (no network):
    python equation_solve.py --mode offline --target linear

    # Diagnostic against the real trained CNN (shows the linearity
    # assumption breaking down -- expected, and reported as such):
    python equation_solve.py --mode offline --target cnn --checkpoint victim.pt

    # Online: extract victim_server.py over the real wire protocol.
    python equation_solve.py --mode network --host 127.0.0.1 --port 9009 \
        --dim 784 --num-classes 10
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# protocol.py is the SHARED file, one directory up (repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol as P

from model import image_to_tensor, load_victim_model


# --------------------------------------------------------------------------- #
# Victims to query (offline mode only -- network mode queries the real
# server instead)
# --------------------------------------------------------------------------- #

def make_linear_victim(dim: int, num_classes: int, seed: int) -> nn.Module:
    """A genuine softmax model: logits = Wx + b. Attack B's ideal target."""
    torch.manual_seed(seed)
    layer = nn.Linear(dim, num_classes)
    layer.eval()
    return layer


def query_local_linear(layer: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(x).float().unsqueeze(0)
        return F.softmax(layer(t), dim=1).squeeze(0).numpy()


def query_local_cnn(model: nn.Module, x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = x.reshape(hw[0], hw[1], 1).astype(np.float32)
    with torch.no_grad():
        return F.softmax(model(image_to_tensor(img)), dim=1).squeeze(0).numpy()


class NetworkOracle:
    """Queries a real protocol.py-speaking server (victim_server.py / mock_victim.py)."""

    def __init__(self, host: str, port: int, hw: tuple[int, int]):
        self.hw = hw
        self.next_id = 1
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        img = x.reshape(self.hw[0], self.hw[1], 1).astype(np.float32)
        P.send_frame(self.sock, P.encode_request(img, self.next_id))
        self.next_id += 1
        msg_type, rid, payload = P.recv_frame(self.sock)
        if msg_type == P.MSG_ERROR:
            raise RuntimeError(f"victim returned ERROR code {P.decode_error(payload)}")
        probs, _flags = P.decode_response(payload)
        return probs

    def close(self):
        self.sock.close()


# --------------------------------------------------------------------------- #
# Probes -- the non-adaptive query set (design report: fixed in advance)
# --------------------------------------------------------------------------- #

def make_probes(dim: int, n: int, seed: int, kind: str) -> np.ndarray:
    """
    n probe vectors in R^dim. Almost surely affinely independent (continuous
    distribution), which is all the least-squares solve below needs.
    """
    rng = np.random.default_rng(seed)
    if kind == "pixels":       # image-shaped victims: valid normalised pixels
        return rng.random((n, dim), dtype=np.float32)
    return (rng.standard_normal((n, dim)).astype(np.float32)) * 0.5   # "gaussian"


# --------------------------------------------------------------------------- #
# The equation-solving core
# --------------------------------------------------------------------------- #

def solve_equations(oracle, dim: int, num_classes: int, n_queries: int,
                     seed: int, probe_kind: str, eps: float = 1e-9):
    """
    Query `oracle` n_queries times and solve, by ordinary least squares, for
    the softmax weights relative to class 0:
        log p_i(x) - log p_0(x) = (w_i - w_0).x + (b_i - b_0),  i = 1..K-1
    Returns (dW, db, X, elapsed_seconds) where dW is (K-1, dim), db is (K-1,).
    """
    X = make_probes(dim, n_queries, seed, probe_kind)
    Y = np.empty((n_queries, num_classes - 1), dtype=np.float64)

    t0 = time.time()
    for row in range(n_queries):
        probs = oracle(X[row])
        logp = np.log(np.clip(probs, eps, None))
        Y[row] = logp[1:] - logp[0]
    elapsed = time.time() - t0

    A = np.hstack([X.astype(np.float64), np.ones((n_queries, 1))])   # (n, dim+1)
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)                      # (dim+1, K-1)
    dW = sol[:dim].T          # (K-1, dim)
    db = sol[dim]             # (K-1,)
    return dW, db, X, elapsed


def recovered_logits(dW: np.ndarray, db: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Recovered model's logits, with logit_0 fixed at 0 (the gauge choice)."""
    rest = dW @ x + db
    return np.concatenate([[0.0], rest])


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def load_mnist_test_flat(data_dir: str, n: int, dim: int) -> np.ndarray | None:
    """Real MNIST test images, flattened to (n, dim). None if unavailable/wrong dim."""
    if dim != 28 * 28:
        return None
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(data_dir, train=False, download=True,
                             transform=transforms.ToTensor())
    except Exception:
        return None
    n = min(n, len(ds))
    imgs = np.stack([ds[i][0].numpy().reshape(-1) for i in range(n)]).astype(np.float32)
    return imgs


def evaluate(oracle, dW: np.ndarray, db: np.ndarray, dim: int, num_classes: int,
             n_test: int, seed: int, probe_kind: str, X_test: np.ndarray | None = None):
    """
    Fidelity (label agreement) and mean total-variation distance on a test
    set. If X_test is not given, fresh probes from the *same* distribution
    used for training are drawn -- fine for the linear target, but for a
    non-linear (CNN) target this only tests the single ReLU-activation
    region the probes happened to land in, not the true data manifold, so
    callers should pass real data as X_test there (see main()).
    """
    if X_test is None:
        X_test = make_probes(dim, n_test, seed + 1, probe_kind)
    agree = 0
    tv_sum = 0.0
    for x in X_test:
        probs = oracle(x)
        true_label = int(np.argmax(probs))
        logits_hat = recovered_logits(dW, db, x)
        pred_label = int(np.argmax(logits_hat))
        agree += int(pred_label == true_label)
        probs_hat = softmax_np(logits_hat)
        tv_sum += 0.5 * np.abs(probs_hat - probs).sum()
    n = len(X_test)
    return agree / n, tv_sum / n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Attack B: functionally-equivalent equation-solving extraction.")
    ap.add_argument("--mode", choices=["offline", "network"], default="offline")
    ap.add_argument("--target", choices=["linear", "cnn"], default="linear",
                     help="offline mode only: synthetic softmax victim, or the real CNN")
    ap.add_argument("--checkpoint", default="victim.pt", help="--target cnn checkpoint")
    ap.add_argument("--data-dir", default="../data",
                     help="--target cnn: torchvision cache dir for the real evaluation set")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--dim", type=int, default=784, help="input dimensionality d")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--height", type=int, default=28)
    ap.add_argument("--width", type=int, default=28)
    ap.add_argument("--queries-scale", type=float, default=1.0,
                     help="alpha: query budget = alpha * (K-1) * (d+1)")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None, help="optional .npz to save recovered weights")
    args = ap.parse_args()

    dim, K = args.dim, args.num_classes
    n_train = max(dim + 1, int(round(args.queries_scale * (dim + 1))))
    net = None

    if args.mode == "offline":
        if args.target == "linear":
            layer = make_linear_victim(dim, K, args.seed)
            oracle = lambda x: query_local_linear(layer, x)
            probe_kind = "gaussian"
            label = "offline:linear-softmax"
        else:
            model = load_victim_model(args.checkpoint, num_classes=K, seed=args.seed)
            hw = (args.height, args.width)
            oracle = lambda x: query_local_cnn(model, x, hw)
            probe_kind = "pixels"
            label = f"offline:cnn({args.checkpoint})"
    else:
        hw = (args.height, args.width)
        net = NetworkOracle(args.host, args.port, hw)
        oracle = net
        probe_kind = "pixels"
        label = f"network:{args.host}:{args.port}"

    print(f"[equation-solve] target={label}  dim={dim}  classes={K}  "
          f"queries={n_train} (train) + {args.n_test} (test)")

    dW, db, _X, elapsed = solve_equations(oracle, dim, K, n_train, args.seed, probe_kind)

    # Evaluating on more probes drawn from the *same* distribution as the
    # training queries would only test the single ReLU-activation region
    # those probes landed in -- against a non-linear victim that can look
    # like a perfect fit while telling us nothing about the real data
    # manifold. Whenever the input is image-shaped (dim == 28*28), use the
    # real MNIST test set instead, for every mode/target: it is the honest
    # test of whether the recovered *global* linear model generalises to
    # the data the victim actually cares about.
    eval_X = None
    eval_desc = f"{args.n_test} fresh probes"
    real_imgs = load_mnist_test_flat(args.data_dir, args.n_test, dim)
    if real_imgs is not None:
        eval_X = real_imgs
        eval_desc = f"{len(real_imgs)} real MNIST test images"

    fidelity, tv = evaluate(oracle, dW, db, dim, K, args.n_test, args.seed, probe_kind,
                             X_test=eval_X)

    print(f"[equation-solve] solved {(K - 1) * (dim + 1)} unknowns from "
          f"{n_train} queries in {elapsed:.2f}s")
    print(f"[equation-solve] label agreement (fidelity) on {eval_desc} = {fidelity:.4f}")
    print(f"[equation-solve] mean total-variation distance = {tv:.6f}")

    if args.mode == "offline" and args.target == "linear":
        assert fidelity > 0.99, (
            f"expected near-perfect fidelity for a genuine linear/softmax victim, "
            f"got {fidelity:.4f} -- equation-solving core may be broken")
        print("[equation-solve] self-check passed: linear/softmax victim recovered "
              "near-exactly, matching the report's Attack-B claim.")
    elif args.mode == "offline" and args.target == "cnn":
        print("[equation-solve] NOTE: fidelity is expected to be well below the "
              "linear case here -- the real victim is a CNN, not a softmax model, "
              "so Attack B's linearity assumption does not hold. This is the "
              "documented limitation motivating the learning-based attack "
              "(attacker/attack_client.py + knockoff_train.py).")

    if net is not None:
        net.close()

    if args.out:
        np.savez(args.out, dW=dW, db=db, fidelity=fidelity, tv=tv,
                 dim=dim, num_classes=K, n_train_queries=n_train)
        print(f"[equation-solve] saved recovered weights -> {args.out}")


if __name__ == "__main__":
    main()
