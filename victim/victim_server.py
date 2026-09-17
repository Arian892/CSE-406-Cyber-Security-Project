"""
victim_server.py  --  PRITU.  The real victim: secret classifier over TCP.

Loads the secret CNN (victim.pt, trained by train_victim.py), listens on a
TCP socket, decodes REQUEST frames (protocol.py), runs inference, routes
the softmax output through the defence layer (defence.py), and returns
RESPONSE frames -- exactly the contract mock_victim.py stood in for, so
the attacker's client (attacker/attack_client.py) is unmodified when
pointed at this server instead.

Component A of the design report:
    "Loads the secret CNN, listens on TCP, decodes REQUEST frames, runs
     inference, applies the defence, returns RESPONSE frames."

The victim answers every well-formed query (the report's threat model: a
single bad request cannot get the attacker blocked) -- the only lever held
here is how much information each answer leaks, via --round-decimals,
--top-k, --label-only, --noise-std, and the per-client quota via
--rate-limit / --rate-window (Section 6 of the report).

Run:
    python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt
    # with defences on:
    python victim_server.py --port 9009 --model victim.pt \
        --top-k 3 --noise-std 0.02 --rate-limit 2000 --rate-window 60
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F

# protocol.py is the SHARED file, one directory up (repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol as P

from model import image_to_tensor, load_victim_model
from defence import DefenceConfig, RateLimiter, apply_defence

# The server is threaded (one thread per connection, Section 2.2: the
# victim answers every well-formed query regardless of who else is
# connected), but the model, RNG, and rate-limiter state are shared across
# those threads. torch inference-only forward passes are safe to interleave,
# but numpy's Generator and the limiter's per-client deques are not
# guaranteed thread-safe, so every request is serialised through this lock.
_INFERENCE_LOCK = threading.Lock()


def handle_client(conn: socket.socket, addr, model, cfg: DefenceConfig,
                   limiter: RateLimiter, rng: np.random.Generator, verbose: bool):
    client_id = addr[0]  # rate-limit per peer IP, not per connection
    n_served = 0
    with conn, torch.no_grad():
        while True:
            try:
                msg_type, rid, payload = P.recv_frame(conn)
            except (ConnectionError, P.ProtocolError):
                break

            if msg_type != P.MSG_REQUEST:
                P.send_frame(conn, P.encode_error(rid, P.ERR_BAD_TYPE))
                continue

            with _INFERENCE_LOCK:
                if not limiter.allow(client_id):
                    P.send_frame(conn, P.encode_error(rid, P.ERR_RATE_LIMIT))
                    continue

                try:
                    img = P.decode_request(payload)             # (H, W, C) float32
                    logits = model(image_to_tensor(img))
                    probs = F.softmax(logits, dim=1).squeeze(0).numpy()
                except Exception:
                    P.send_frame(conn, P.encode_error(rid, P.ERR_MALFORMED))
                    continue

                out_probs, flags = apply_defence(probs, cfg, rng)

            P.send_frame(conn, P.encode_response(out_probs, rid, flags=flags))
            n_served += 1

    if verbose:
        print(f"[victim] {client_id} disconnected after {n_served} queries "
              f"(quota used: {limiter.usage(client_id)})")


def build_defence_config(args) -> DefenceConfig:
    return DefenceConfig(
        round_decimals=args.round_decimals,
        top_k=args.top_k,
        label_only=args.label_only,
        noise_std=args.noise_std,
        rate_limit_queries=args.rate_limit,
        rate_limit_window_s=args.rate_window,
    )


def main():
    ap = argparse.ArgumentParser(description="Victim server: secret classifier over TCP.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--model", default="victim.pt", help="checkpoint from train_victim.py")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)

    # Defences (Section 6) -- all off by default, matching mock_victim's
    # undefended passthrough so a bare-server smoke test is still possible.
    ap.add_argument("--round-decimals", type=int, default=None,
                     help="6.1: round returned probabilities to r decimals")
    ap.add_argument("--top-k", type=int, default=None,
                     help="6.1: return only the top-k class probabilities")
    ap.add_argument("--label-only", action="store_true",
                     help="6.1: extreme case, return a one-hot label only")
    ap.add_argument("--noise-std", type=float, default=0.0,
                     help="6.2: stddev of calibrated noise added to probabilities")
    ap.add_argument("--rate-limit", type=int, default=None,
                     help="6.3: max queries per client within --rate-window")
    ap.add_argument("--rate-window", type=float, default=60.0,
                     help="6.3: sliding window length in seconds")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model_path = args.model if os.path.exists(args.model) else None
    if model_path is None:
        print(f"[victim] WARNING: checkpoint '{args.model}' not found -- "
              f"serving an UNTRAINED model (random weights). Run "
              f"train_victim.py first for meaningful predictions.")
    model = load_victim_model(model_path, num_classes=args.num_classes, seed=args.seed)

    cfg = build_defence_config(args)
    limiter = RateLimiter(cfg.rate_limit_queries, cfg.rate_limit_window_s)
    rng = np.random.default_rng(args.seed)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(16)
    print(f"[victim] listening on {args.host}:{args.port}  "
          f"(model={'trained' if model_path else 'RANDOM'}, "
          f"defence={cfg.__dict__})")
    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, model, cfg, limiter, rng, not args.quiet),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\n[victim] shutting down")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
