"""Weight averaging (EMA) of the trained adapter (optim.ema): checkpoints, samples and the saved
LoRA come from a running average of the recent steps rather than whichever step the save landed
on, so each reflects the whole dataset instead of the tail of the shuffle. Fizgig ships 0.98 on
by default and measured it higher and steadier on likeness than off.

The average lives beside the live weights in fp32 (an adapter is small), with a warmup so early
checkpoints don't lag behind the start. The resume state keeps both: the live weights to train
on, the average to keep averaging.
"""

from __future__ import annotations

import contextlib

import torch


class EMA:
    def __init__(self, params: dict[str, torch.nn.Parameter], decay: float):
        self.params = params
        self.decay = decay
        self.shadow = {k: p.detach().float().clone() for k, p in params.items()}

    def rate(self, step: int) -> float:
        return min(self.decay, (1.0 + step) / (10.0 + step))  # short warmup: early saves track the weights

    @torch.no_grad()
    def update(self, step: int) -> None:
        d = self.rate(step)
        for k, p in self.params.items():
            self.shadow[k].lerp_(p.detach().float(), 1.0 - d)

    @contextlib.contextmanager
    def applied(self):
        """The averaged weights in place for a moment (a save, a sample); yields the live ones."""
        live = {k: p.detach().clone() for k, p in self.params.items()}
        try:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(self.shadow[k].to(p.dtype))
            yield live
        finally:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(live[k])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.get("shadow", {}).items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                self.shadow[k].copy_(v.to(self.shadow[k]))
