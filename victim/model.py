

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VictimCNN(nn.Module):
    """The secret classifier f: 3 conv blocks (2 pooled) + 1 hidden FC layer."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # 14 -> 7
        x = F.relu(self.bn3(self.conv3(x)))                    # 7 -> 7 (no pool)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)                                     # logits


def load_victim_model(model_path: str | None, num_classes: int = 10,
                       seed: int = 1337, device: str = "cpu") -> nn.Module:
    """Instantiate VictimCNN and optionally load a trained checkpoint."""
    torch.manual_seed(seed)
    model = VictimCNN(num_classes=num_classes)
    if model_path:
        state = torch.load(model_path, map_location=device, weights_only=True)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, C) float32 numpy array, as decoded off the wire -> (1, C, H, W) tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    return t.float()


if __name__ == "__main__":
    # Smoke test: forward pass shape/param sanity check, no training or I/O.
    m = VictimCNN(num_classes=10)
    m.eval()
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (4, 10), f"unexpected output shape {tuple(out.shape)}"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"model.py self-test OK  (output shape {tuple(out.shape)}, {n_params} params)")
