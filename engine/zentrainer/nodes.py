"""ComfyUI nodes. Imported only inside a running ComfyUI."""

from __future__ import annotations

import time

from comfy_api.latest import io

from . import state
from .harness import control
from .harness.trainer import Trainer
from .spec import TrainSpec


class ZenTrainSession(io.ComfyNode):
    """Hosts one training run. The run is defined by its spec in the run store, not
    by graph inputs: this node only claims ComfyUI's execution lane (and with it the
    GPU) for as long as the harness trains, and makes ComfyUI's cancel button work."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZenTrainSession",
            display_name="Training Session (ZenTrainer)",
            category="zentrainer",
            is_output_node=True,
            inputs=[io.String.Input("run_id", tooltip="A run created through the ZenTrainer app or API.")],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, run_id):
        return time.time()  # never serve a session from cache

    @classmethod
    def execute(cls, run_id) -> io.NodeOutput:
        store = state.store()
        spec = TrainSpec.from_dict(store.spec(run_id))
        meta = store.meta(run_id)
        ctl = control.register(run_id)
        try:
            Trainer(spec, run_id, store=store, control=ctl,
                    resume_from=meta.get("resume_from"), resume_state=meta.get("resume_state")).run()
        finally:
            control.unregister(run_id)
        return io.NodeOutput()


class ZenCaptionSession(io.ComfyNode):
    """Hosts one auto-captioning job (settings held by the engine, keyed by job id)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZenCaptionSession",
            display_name="Caption Dataset (ZenTrainer)",
            category="zentrainer",
            is_output_node=True,
            inputs=[io.String.Input("job_id")],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, job_id):
        return time.time()

    @classmethod
    def execute(cls, job_id) -> io.NodeOutput:
        from . import captioner

        job = captioner.get(job_id)
        if job is None:
            raise RuntimeError(f"caption job {job_id} is gone (ComfyUI restarted?); start it again")
        captioner.run(job)
        return io.NodeOutput()


class ZenSegmentSession(io.ComfyNode):
    """Hosts one SAM3 mask request from the dataset mask editor (engine: segment.py)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZenSegmentSession",
            display_name="Segment for Mask (ZenTrainer)",
            category="zentrainer",
            is_output_node=True,
            inputs=[io.String.Input("job_id")],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, job_id):
        return time.time()

    @classmethod
    def execute(cls, job_id) -> io.NodeOutput:
        from . import segment

        job = segment.get(job_id)
        if job is None:
            raise RuntimeError(f"segment job {job_id} is gone (ComfyUI restarted?); try again")
        segment.run(job)
        return io.NodeOutput()


NODES: list[type[io.ComfyNode]] = [ZenTrainSession, ZenCaptionSession, ZenSegmentSession]
