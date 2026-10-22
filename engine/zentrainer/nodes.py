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
        from .datasets import segment

        segment.release()  # the mask editor's SAM3, if it's still held: the run wants the memory
        ctl = control.register(run_id)
        # sampling a saved checkpoint trains nothing: that checkpoint's LoRA, no training state. A run
        # that was once resumed still names what it resumed from, and that state may be cleaned up
        # since (or it would load that LoRA instead of the one asked for)
        job = meta.get("sample_job")
        try:
            Trainer(spec, run_id, store=store, control=ctl,
                    resume_from=None if job else meta.get("resume_from"),
                    resume_state=None if job else meta.get("resume_state"),
                    sample_job=job).run()
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
        from .datasets import captioner

        job = captioner.get(job_id)
        if job is None:
            raise RuntimeError(f"caption job {job_id} is gone (ComfyUI restarted?); start it again")
        captioner.run(job)
        return io.NodeOutput()


class ZenSegmentSession(io.ComfyNode):
    """Hosts one SAM3 mask request from the dataset mask editor (engine: datasets/segment.py)."""

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
        from .datasets import segment

        job = segment.get(job_id)
        if job is None:
            raise RuntimeError(f"segment job {job_id} is gone (ComfyUI restarted?); try again")
        segment.run(job)
        return io.NodeOutput()


class ZenMOSSEditModel(io.ComfyNode):
    """MOSS Image with references (harness/moss_refs.py): what an edit LoRA trained with
    references needs to sample as it trained."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZenMOSSEditModel",
            display_name="MOSS Edit Model (references)",
            category="zentrainer/moss",
            description="Lets MOSS Image take reference pictures, as ZenTrainer trains its edit LoRAs: put it after the LoRA loader, and give it conditioning from MOSS Edit Encode (or ReferenceLatent).",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        from .harness import moss_refs

        model = model.clone()
        moss_refs.apply(model)
        return io.NodeOutput(model)


class ZenMOSSEditEncode(io.ComfyNode):
    """An edit's conditioning as ZenTrainer trains it: the input pictures as references at the
    output's size, and (Encoder sees the input) the text encoder looking at them whole."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZenMOSSEditEncode",
            display_name="MOSS Edit Encode",
            category="zentrainer/moss",
            description="The instruction and the picture(s) to edit, for MOSS Edit Model. The latent is the empty output: the first picture's shape at this many megapixels.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative", multiline=True, default="", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Boolean.Input("encoder_sees_input", default=True, tooltip="The text encoder looks at the pictures with the instruction. Match the LoRA's 'Encoder sees the input'"),
                io.Combo.Input("reference_method", options=["index_timestep_zero", "index"], default="index_timestep_zero", tooltip="How the pictures join the sequence, as the LoRA trained: index_timestep_zero (the default: clean, attending to each other, ai-toolkit's layout) or index (at the step's noise level, attending to everything)"),
                io.Float.Input("megapixels", default=1.0, min=0.25, max=4.0, step=0.05),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Conditioning.Output(display_name="negative"),
                     io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, clip, vae, image, prompt, negative="", image2=None, image3=None, encoder_sees_input=True,
                reference_method="index_timestep_zero", megapixels=1.0) -> io.NodeOutput:
        import comfy.utils
        import torch

        from .harness.data.edit import encode_grounded, moss_grounding, with_refs

        pictures = [p[:1, :, :, :3] for p in (image, image2, image3) if p is not None]
        h0, w0 = pictures[0].shape[1:3]
        step = 64  # as training's buckets
        area = megapixels * 1024 * 1024
        w = max(step, round((area * w0 / h0) ** 0.5 / step) * step)
        h = max(step, round((area * h0 / w0) ** 0.5 / step) * step)

        def fit(px, fw, fh, crop):
            return comfy.utils.common_upscale(px.movedim(-1, 1), fw, fh, "lanczos", crop).movedim(1, -1).clamp(0, 1)

        refs = [vae.encode(fit(p, w, h, "center")) for p in pictures]  # cropped to the output, as training's are
        if encoder_sees_input:  # whole, as training's encoder saw them
            seen = [moss_grounding(p) for p in pictures]
            pos, neg = encode_grounded(clip, prompt, seen), encode_grounded(clip, negative, seen)
        else:
            pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
        return io.NodeOutput(with_refs(pos, refs, reference_method), with_refs(neg, refs, reference_method),
                             {"samples": torch.zeros_like(refs[0])})


NODES: list[type[io.ComfyNode]] = [ZenTrainSession, ZenCaptionSession, ZenSegmentSession, ZenMOSSEditModel,
                                   ZenMOSSEditEncode]
