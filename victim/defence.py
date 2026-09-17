"""
defence.py  --  PRITU.  The victim's defence layer (design report, Section 6).

Sits between the secret classifier's raw softmax output and the RESPONSE
frame the victim sends back. Implements the layered defences the report
evaluates on-vs-off:

  6.1 Output minimisation (the primary control)
      - round probabilities to r decimals
      - expose only the top-k classes (rest zeroed out)
      - label-only / argmax mode (the k=1 special case)
  6.2 Prediction perturbation
      - small calibrated noise added to the soft-label vector, argmax
        preserved for honest users, but the distillation signal is
        corrupted for an attacker training a knockoff on it
  6.3 Query-pattern monitoring and rate limiting
      - a per-client sliding-window quota; over quota -> ERROR code 4
        (protocol.ERR_RATE_LIMIT), enforced by victim_server.py

Every probability transform sets the RESPONSE FLAGS bits (protocol.py)
so the attacker's client can see, from the wire, which defences were
applied (attacker/attack_client.py already decodes and reports these).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

import os
import sys

# protocol.py is the SHARED file, one directory up (repo root). Reach it
# without touching any attacker-owned file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol as P


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class DefenceConfig:
    """One knob per defence; None / 0 / False disables that defence."""

    round_decimals: int | None = None      # 6.1: round(p, r)
    top_k: int | None = None                # 6.1: keep only top-k classes
    label_only: bool = False                # 6.1: extreme case, k=1, one-hot
    noise_std: float = 0.0                  # 6.2: stddev of additive noise
    rate_limit_queries: int | None = None   # 6.3: max queries per window
    rate_limit_window_s: float = 60.0       # 6.3: sliding window length

    def any_output_minimisation(self) -> bool:
        return self.round_decimals is not None or self.top_k is not None or self.label_only

    def any_defence(self) -> bool:
        return self.any_output_minimisation() or self.noise_std > 0 or self.rate_limit_queries is not None


# --------------------------------------------------------------------------- #
# 6.1 Output minimisation
# --------------------------------------------------------------------------- #

def _apply_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest entries, zero out the rest (Knockoff Nets, Sec 6.1)."""
    k = max(1, min(k, len(probs)))
    if k >= len(probs):
        return probs.copy()
    idx = np.argpartition(probs, -k)[-k:]
    out = np.zeros_like(probs)
    out[idx] = probs[idx]
    return out


def _apply_label_only(probs: np.ndarray) -> np.ndarray:
    """Return a one-hot vector at the argmax -- the extreme top-1 case."""
    out = np.zeros_like(probs)
    out[int(np.argmax(probs))] = 1.0
    return out


def apply_output_minimisation(probs: np.ndarray, cfg: DefenceConfig):
    """Round / truncate the soft label. Returns (probs_out, flag_bits)."""
    out = np.asarray(probs, dtype=np.float32).copy()
    flags = 0

    if cfg.label_only:
        out = _apply_label_only(out)
        flags |= P.FLAG_TOPK          # top-1 is a special case of top-k truncation
    elif cfg.top_k is not None:
        out = _apply_top_k(out, cfg.top_k)
        flags |= P.FLAG_TOPK

    if cfg.round_decimals is not None:
        out = np.round(out, cfg.round_decimals).astype(np.float32)

    return out, flags


# --------------------------------------------------------------------------- #
# 6.2 Prediction perturbation
# --------------------------------------------------------------------------- #

def apply_perturbation(probs: np.ndarray, cfg: DefenceConfig, rng: np.random.Generator):
    """
    Add small calibrated noise, re-normalise, and guarantee the argmax label
    an honest user sees is unchanged. Returns (probs_out, flag_bits).
    """
    if cfg.noise_std <= 0:
        return np.asarray(probs, dtype=np.float32), 0

    true_top = int(np.argmax(probs))
    scale = cfg.noise_std
    for _ in range(3):  # damp the noise down if it flips the decision
        noisy = probs + rng.normal(0.0, scale, size=probs.shape)
        noisy = np.clip(noisy, 1e-6, None)
        noisy = (noisy / noisy.sum()).astype(np.float32)
        if int(np.argmax(noisy)) == true_top:
            return noisy, P.FLAG_NOISED
        scale *= 0.1
    # Last resort: noise kept flipping the label even damped -- serve the
    # true label with clean probabilities rather than mislead an honest user.
    return np.asarray(probs, dtype=np.float32), 0


# --------------------------------------------------------------------------- #
# Combined defence layer
# --------------------------------------------------------------------------- #

def apply_defence(probs: np.ndarray, cfg: DefenceConfig, rng: np.random.Generator):
    """
    Run the full defence pipeline: perturbation first (operates on the true
    distribution), then output minimisation (what actually leaves the wire).
    Returns (probs_out, flags) ready for protocol.encode_response.
    """
    out, flags = apply_perturbation(probs, cfg, rng)
    out, min_flags = apply_output_minimisation(out, cfg)
    flags |= min_flags | P.FLAG_PROBS
    return out.astype(np.float32), flags


# --------------------------------------------------------------------------- #
# 6.3 Query-pattern monitoring / rate limiting
# --------------------------------------------------------------------------- #

class RateLimiter:
    """
    Per-client sliding-window query quota. Each client_id (e.g. peer IP, or
    IP:port for stricter per-connection accounting) gets its own deque of
    query timestamps; queries older than the window are dropped lazily.
    """

    def __init__(self, max_queries: int | None, window_s: float = 60.0):
        self.max_queries = max_queries
        self.window_s = window_s
        self._history: dict[str, deque] = defaultdict(deque)

    def allow(self, client_id: str, now: float | None = None) -> bool:
        """Record one query attempt; return False if the client is over quota."""
        if self.max_queries is None:
            return True
        now = time.time() if now is None else now
        hist = self._history[client_id]
        cutoff = now - self.window_s
        while hist and hist[0] < cutoff:
            hist.popleft()
        if len(hist) >= self.max_queries:
            return False
        hist.append(now)
        return True

    def usage(self, client_id: str) -> int:
        return len(self._history.get(client_id, ()))


# --------------------------------------------------------------------------- #
# Self-test: exercise every defence in isolation and combined.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(1337)
    probs = np.array([0.01, 0.02, 0.05, 0.72, 0.10, 0.03, 0.02, 0.02, 0.02, 0.01],
                      dtype=np.float32)
    assert abs(probs.sum() - 1.0) < 1e-6

    # -- rounding --------------------------------------------------------- #
    cfg = DefenceConfig(round_decimals=2)
    out, flags = apply_output_minimisation(probs, cfg)
    assert np.allclose(out, np.round(probs, 2))
    assert flags == 0

    # -- top-k -------------------------------------------------------------#
    cfg = DefenceConfig(top_k=3)
    out, flags = apply_output_minimisation(probs, cfg)
    assert np.count_nonzero(out) == 3
    assert int(np.argmax(out)) == int(np.argmax(probs))
    assert flags & P.FLAG_TOPK

    # -- label-only ----------------------------------------------------- #
    cfg = DefenceConfig(label_only=True)
    out, flags = apply_output_minimisation(probs, cfg)
    assert out.sum() == 1.0 and out[int(np.argmax(probs))] == 1.0
    assert flags & P.FLAG_TOPK

    # -- perturbation preserves argmax and stays a valid distribution --- #
    cfg = DefenceConfig(noise_std=0.02)
    for _ in range(200):
        out, flags = apply_perturbation(probs, cfg, rng)
        assert int(np.argmax(out)) == int(np.argmax(probs)), "perturbation flipped the label"
        assert abs(out.sum() - 1.0) < 1e-4
        assert flags == P.FLAG_NOISED

    # -- combined pipeline -------------------------------------------------#
    cfg = DefenceConfig(round_decimals=3, top_k=5, noise_std=0.01)
    out, flags = apply_defence(probs, cfg, rng)
    assert flags & P.FLAG_PROBS and flags & P.FLAG_TOPK and flags & P.FLAG_NOISED
    assert np.count_nonzero(out) <= 5

    # -- no defence configured -> passthrough ------------------------------#
    cfg = DefenceConfig()
    out, flags = apply_defence(probs, cfg, rng)
    assert np.allclose(out, probs)
    assert flags == P.FLAG_PROBS

    # -- rate limiter: sliding window quota --------------------------------#
    rl = RateLimiter(max_queries=3, window_s=10.0)
    t0 = 1000.0
    assert rl.allow("1.2.3.4", now=t0)
    assert rl.allow("1.2.3.4", now=t0 + 1)
    assert rl.allow("1.2.3.4", now=t0 + 2)
    assert not rl.allow("1.2.3.4", now=t0 + 3), "4th query within window should be blocked"
    assert rl.allow("5.6.7.8", now=t0 + 3), "a different client has its own quota"
    assert rl.allow("1.2.3.4", now=t0 + 11), "window has slid past the first 3 queries"

    print("defence.py self-test OK  (rounding, top-k, label-only, noise, "
          "rate-limit all pass)")
