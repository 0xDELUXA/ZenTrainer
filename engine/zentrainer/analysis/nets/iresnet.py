"""ArcFace's IResNet (InsightFace's w600k_r50 and glintr100), with the batch norms that the exported
files fold into the convolutions folded the same way."""

from __future__ import annotations

from torch import nn


class _Block(nn.Module):
    """BN, 3x3 conv, PReLU, 3x3 conv (which carries the stride), plus the shortcut."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(cin, eps=1e-5)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.prelu = nn.PReLU(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride, 1)
        self.down = nn.Conv2d(cin, cout, 1, stride) if stride != 1 else None

    def forward(self, x):
        out = self.conv2(self.prelu(self.conv1(self.bn(x))))
        return out + (self.down(x) if self.down is not None else x)


class IResNet(nn.Module):
    """(N, 3, 112, 112) aligned faces, RGB as (x - 127.5) / 127.5 -> (N, 512) identity embeddings
    (not unit length)."""

    LAYERS = {"r50": (3, 4, 14, 3), "r100": (3, 13, 30, 3)}  # blocks per stage

    def __init__(self, depth: str):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.prelu = nn.PReLU(64)
        blocks, cin = [], 64
        for cout, n in zip((64, 128, 256, 512), self.LAYERS[depth]):
            blocks += [_Block(cin, cout, 2)] + [_Block(cout, cout, 1) for _ in range(n - 1)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(512, eps=1e-5)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-5)

    def forward(self, x):
        x = self.bn(self.blocks(self.prelu(self.conv(x))))
        return self.features(self.fc(x.flatten(1)))
