"""SCRFD-10G, InsightFace's face detector (buffalo_l's det_10g, the same file as antelopev2's
scrfd_10g_bnkps), with its batch norms folded into the convolutions as the exported file has them."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _Block(nn.Module):
    """ResNet-D basic block: two 3x3 convs and a shortcut (an average pool, then a 1x1 conv, when it
    downsamples)."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.down = nn.Conv2d(cin, cout, 1) if stride != 1 or cin != cout else None
        self.stride = stride

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        if self.down is not None:
            if self.stride != 1:
                x = F.avg_pool2d(x, 2, 2, ceil_mode=True, count_include_pad=False)
            x = self.down(x)
        return F.relu(out + x)


class _Head(nn.Module):
    """One stride's head: three 3x3 conv + relu, then scores (2 anchors a cell), box and 5-keypoint
    distances."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(56, 80, 3, 1, 1), nn.Conv2d(80, 80, 3, 1, 1), nn.Conv2d(80, 80, 3, 1, 1)])
        self.cls = nn.Conv2d(80, 2, 3, 1, 1)
        self.reg = nn.Conv2d(80, 8, 3, 1, 1)
        self.kps = nn.Conv2d(80, 20, 3, 1, 1)
        self.register_buffer("scale", torch.ones(()))

    def forward(self, x):
        for c in self.convs:
            x = F.relu(c(x))
        # (1, C, H, W) -> (H * W * anchors, k): the anchor varies fastest, as the anchor centres do
        flat = lambda t, k: t.permute(2, 3, 0, 1).reshape(-1, k)
        return flat(self.cls(x), 1).sigmoid(), flat(self.reg(x) * self.scale, 4), flat(self.kps(x), 10)


class SCRFD(nn.Module):
    """(1, 3, H, W) RGB as (x - 127.5) / 128, H and W multiples of 32 -> for strides 8, 16 and 32:
    (scores (N, 1), box distances (N, 4), keypoint offsets (N, 10)), in units of the stride."""

    def __init__(self):
        super().__init__()
        self.stem = nn.ModuleList([nn.Conv2d(3, 28, 3, 2, 1), nn.Conv2d(28, 28, 3, 1, 1), nn.Conv2d(28, 56, 3, 1, 1)])
        # (in, out, blocks, stride): the stages end at strides 4, 8, 16 and 32
        for i, (cin, cout, n, stride) in enumerate([(56, 56, 3, 1), (56, 88, 4, 2), (88, 88, 2, 2), (88, 224, 3, 2)]):
            setattr(self, f"layer{i + 1}", nn.Sequential(_Block(cin, cout, stride), *[_Block(cout, cout, 1) for _ in range(n - 1)]))
        self.lateral = nn.ModuleList([nn.Conv2d(88, 56, 1), nn.Conv2d(88, 56, 1), nn.Conv2d(224, 56, 1)])
        self.fpn = nn.ModuleList([nn.Conv2d(56, 56, 3, 1, 1) for _ in range(3)])
        self.down = nn.ModuleList([nn.Conv2d(56, 56, 3, 2, 1) for _ in range(2)])
        self.pafpn = nn.ModuleList([nn.Conv2d(56, 56, 3, 1, 1) for _ in range(2)])
        self.heads = nn.ModuleList([_Head() for _ in range(3)])

    def forward(self, x):
        for c in self.stem:
            x = F.relu(c(x))
        x = F.max_pool2d(x, 2, 2)
        c2 = self.layer2(self.layer1(x))
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        # top-down, then bottom-up (PAFPN)
        p4 = self.lateral[2](c4)
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        o2, o3, o4 = self.fpn[0](p2), self.fpn[1](p3), self.fpn[2](p4)
        n3 = o3 + self.down[0](o2)
        n4 = o4 + self.down[1](n3)
        return [h(f) for h, f in zip(self.heads, (o2, self.pafpn[0](n3), self.pafpn[1](n4)))]
