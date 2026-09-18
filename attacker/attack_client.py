

import argparse
import socket
import time

import numpy as np

import protocol as P


# --------------------------------------------------------------------------- #
# Query pools  --  the fixed set of inputs the attacker will send
# --------------------------------------------------------------------------- #

def _try_torchvision(pool: str, data_dir: str, n: int):
    """Load MNIST / Fashion-MNIST test images via torchvision if available."""
    try:
        from torchvision import datasets, transforms
    except Exception:
        return None
    tf = transforms.ToTensor()   # -> (1, 28, 28) float in [0,1]
    try:
        if pool == "mnist":
            ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        elif pool == "fashion":
            ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        else:
            return None
    except Exception as e:
        print(f"[client] torchvision load failed ({e}); falling back to noise")
        return None
    n = min(n, len(ds))
    imgs = np.stack([ds[i][0].squeeze(0).numpy() for i in range(n)]).astype(np.float32)
    return imgs[:, :, :, None]   # (N, 28, 28, 1)


def build_query_pool(pool: str, n: int, data_dir: str, seed: int, hw=(28, 28)):
    """
    Return an (N, H, W, 1) float32 array of query inputs in [0,1].

    'uniform' is a synthetic OOD pool (also used as a fallback when the
    datasets cannot be downloaded in a sandboxed network).
    """
    rng = np.random.default_rng(seed)
    if pool in ("mnist", "fashion"):
        imgs = _try_torchvision(pool, data_dir, n)
        if imgs is not None:
            if len(imgs) < n:   # top up with noise if the test set is smaller
                extra = rng.random((n - len(imgs), *hw, 1), dtype=np.float32)
                imgs = np.concatenate([imgs, extra], axis=0)
            print(f"[client] query pool: {pool}  ({len(imgs)} images)")
            return imgs
        pool = "uniform"        # fall through
    h, w = hw
    imgs = rng.random((n, h, w, 1), dtype=np.float32)
    print(f"[client] query pool: uniform noise  ({n} images)")
    return imgs


# --------------------------------------------------------------------------- #
# Query engine  --  one protocol round trip per input
# --------------------------------------------------------------------------- #

class RateLimitedError(RuntimeError):
    """Raised when the victim's per-client quota is exceeded (ERROR code 4).

    Kept distinct from other protocol errors so the campaign loop can treat
    it as an expected stopping condition -- a rate-limited attacker should
    keep what it already collected, not lose the whole run.
    """

    def __init__(self, code: int):
        super().__init__(f"victim returned ERROR code {code}")
        self.code = code


def send_query(sock: socket.socket, image: np.ndarray, request_id: int):
    """One black-box round trip: send REQUEST, read RESPONSE, return (probs, flags)."""
    P.send_frame(sock, P.encode_request(image, request_id))
    msg_type, rid, payload = P.recv_frame(sock)
    if msg_type == P.MSG_ERROR:
        code = P.decode_error(payload)
        if code == P.ERR_RATE_LIMIT:
            raise RateLimitedError(code)
        raise RuntimeError(f"victim returned ERROR code {code}")
    if msg_type != P.MSG_RESPONSE:
        raise RuntimeError(f"unexpected message type {msg_type}")
    if rid != request_id:
        raise RuntimeError(f"response id {rid} != request id {request_id}")
    probs, flags = P.decode_response(payload)
    return probs, flags


def run_campaign(host, port, images, log_every=1000):
    """
    Stream every image to the victim and collect the prediction vectors.
    If the victim's rate limiter kicks in, stop and keep whatever was
    already collected instead of losing the whole campaign.
    Returns (kept_images (M,H,W,1), probs (M,K), flags_seen set).
    """
    kept_imgs, probs_list, flags_seen = [], [], set()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    t0 = time.time()
    stopped_early = False
    try:
        for i, img in enumerate(images):
            try:
                probs, flags = send_query(sock, img, request_id=i + 1)
            except RateLimitedError as e:
                print(f"[client] quota exceeded after {len(kept_imgs)} queries "
                      f"({e}) -- stopping, keeping what was already collected")
                stopped_early = True
                break
            kept_imgs.append(img)
            probs_list.append(probs)
            flags_seen.add(flags)
            if (i + 1) % log_every == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[client] {i + 1}/{len(images)} queries  "
                      f"({rate:.0f} q/s)  last argmax={int(np.argmax(probs))}")
    finally:
        sock.close()
    imgs = np.stack(kept_imgs).astype(np.float32)
    probs_arr = np.stack(probs_list).astype(np.float32)
    dt = time.time() - t0
    print(f"[client] campaign done: {len(imgs)} pairs in {dt:.1f}s, "
          f"flags observed = {sorted(flags_seen)}"
          + ("  (stopped early: rate limited)" if stopped_early else ""))
    return imgs, probs_arr, flags_seen


def decode_flags(flags_seen) -> dict:
    """Human-readable summary of which defences the responses advertised."""
    any_flags = 0
    for f in flags_seen:
        any_flags |= f
    return {
        "probs_present": bool(any_flags & P.FLAG_PROBS),
        "topk_truncated": bool(any_flags & P.FLAG_TOPK),
        "noised": bool(any_flags & P.FLAG_NOISED),
    }


def main():
    ap = argparse.ArgumentParser(description="Black-box query engine (attacker).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--pool", choices=["mnist", "fashion", "uniform"], default="mnist")
    ap.add_argument("--queries", type=int, default=10000, help="query budget q")
    ap.add_argument("--data-dir", default="./data", help="torchvision cache dir")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="transfer_set.npz")
    args = ap.parse_args()

    images = build_query_pool(args.pool, args.queries, args.data_dir, args.seed)
    imgs, probs, flags_seen = run_campaign(args.host, args.port, images)

    defences = decode_flags(flags_seen)
    np.savez_compressed(
        args.out,
        images=imgs,               # (N, H, W, 1) float32 in [0,1]
        probs=probs,               # (N, K) float32 -- the leaked soft labels
        victim_labels=np.argmax(probs, axis=1).astype(np.int64),
        pool=args.pool,
        queries=args.queries,
        seed=args.seed,
        defences_observed=str(defences),
    )
    print(f"[client] wrote transfer set -> {args.out}")
    print(f"[client] defences advertised by responses: {defences}")
    print(f"[client] tip: sweep budgets with q in 1000/5000/10000/30000, "
          f"one --out file each.")


if __name__ == "__main__":
    main()
