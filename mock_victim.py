"""
mock_victim.py  --  A DEVELOPMENT STAND-IN for the real victim server.

This is NOT a graded file and NOT the real victim. Pritu owns the real
victim_server.py (loads the secret victim.pt, applies defence.py, etc.).
This mock exists purely so Arian can build and test the attack client and
the knockoff trainer against something that speaks the agreed protocol,
without waiting for the real server to be finished.

Contract it honours (identical to the real server):
  * accepts REQUEST frames, replies with RESPONSE frames using protocol.py
  * returns a K-class probability vector with FLAG_PROBS set
  * unknown magic/version/type -> ERROR frame

The "secret model" here is a small, randomly-initialised CNN by default, or
a checkpoint you point it at with --model (so you can smoke-test fidelity).
Swap this out for the real server at integration time -- the client does not
change, because both speak protocol.py.

Run:
    python mock_victim.py --host 127.0.0.1 --port 9009
"""

import argparse
import socket

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import protocol as P


class MockCNN(nn.Module):
    """A throwaway CNN standing in for the secret classifier (28x28x1 -> 10)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                            # logits


def load_model(model_path: str | None, seed: int = 1337) -> nn.Module:
    torch.manual_seed(seed)
    model = MockCNN()
    if model_path:
        state = torch.load(model_path, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        print(f"[mock] loaded weights from {model_path}")
    else:
        print("[mock] using RANDOM weights (fine for protocol/pipeline testing)")
    model.eval()
    return model


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, C) float32 -> (1, C, H, W) tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    return t.float()


def handle_client(conn: socket.socket, model: nn.Module) -> None:
    with conn, torch.no_grad():
        while True:
            try:
                msg_type, rid, payload = P.recv_frame(conn)
            except (ConnectionError, P.ProtocolError):
                return
            if msg_type != P.MSG_REQUEST:
                P.send_frame(conn, P.encode_error(rid, P.ERR_BAD_TYPE))
                continue
            try:
                img = P.decode_request(payload)              # (H, W, C)
                logits = model(image_to_tensor(img))
                probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            except Exception:
                P.send_frame(conn, P.encode_error(rid, P.ERR_MALFORMED))
                continue
            # Raw probs, no defence -- the real server routes this through defence.py.
            P.send_frame(conn, P.encode_response(probs, rid, flags=P.FLAG_PROBS))


def main():
    ap = argparse.ArgumentParser(description="Mock victim server (dev stand-in).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--model", default=None, help="optional checkpoint to load")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    model = load_model(args.model, args.seed)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    print(f"[mock] victim listening on {args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        while True:
            conn, addr = srv.accept()
            handle_client(conn, model)   # iterative server: one client at a time
    except KeyboardInterrupt:
        print("\n[mock] shutting down")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
