"""Websocket events the engine sends. The web client subscribes to these on
ComfyUI's /ws alongside the stock progress/executing messages.

Every event carries `run_id`, so a tab that connects mid-run can filter and rebuild
state (the run store keeps the full history in metrics.jsonl).
"""

from __future__ import annotations

import logging
from typing import Any

METRICS = "zentrainer.metrics"  # {run_id, step, total, loss, lr, grad_norm, epoch, elapsed, it_s}
CHECKPOINT = "zentrainer.checkpoint"  # {run_id, step, path, final, lora_name}
STATUS = "zentrainer.status"  # {run_id, status, message?, step}
SAMPLE = "zentrainer.sample"  # {run_id, step, index, prompt, filename, subfolder, type}
# live sampling: {run_id, step, phase: requested|sampling|done, index, count, prompt,
#                 s_step, s_total, preview (JPEG data URL)}
SAMPLE_PROGRESS = "zentrainer.sample_progress"
VALIDATION = "zentrainer.validation"  # {run_id, step, val_loss, by_noise: {noise level: loss}}
ZENKIT_JOB = "zenkit.job"  # mirrored so ZenKit's taskbar shows the run too


def emit(event: str, data: dict[str, Any]) -> None:
    """Best effort: telemetry must never be what fails a training run."""
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(event, data)
    except Exception:  # outside ComfyUI, server not started, socket gone
        logging.debug("[ZenTrainer] dropped %s event", event, exc_info=True)
