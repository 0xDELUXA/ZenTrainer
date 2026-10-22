"""ZenTrainer engine — a ComfyUI node pack.

Install by linking this folder into ComfyUI:
    ln -s /path/to/ZenTrainer/engine /path/to/ComfyUI/custom_nodes/zentrainer
"""

import logging
import os

from comfy_api.latest import ComfyExtension, io


def _setup() -> None:
    import folder_paths
    from server import PromptServer

    from .zentrainer import paths
    from .zentrainer.server import routes

    # Trained LoRAs (final and mid-run checkpoints) land under output/zentrainer/loras, or the
    # ZenTrainer folder's output/loras when one is set (read here, once: changing it needs a restart).
    # Registering it as a loras folder makes them loadable by LoraLoader at once,
    # which is what sample generation and the inference apps rely on.
    paths.ensure()
    lora_dir = str(paths.lora_dir())
    os.makedirs(lora_dir, exist_ok=True)
    folder_paths.add_model_folder_path("loras", lora_dir)
    legacy = str(paths.legacy_output_root() / "loras")
    if paths.folder() and os.path.isdir(legacy):  # LoRAs trained before the folder was set still load
        folder_paths.add_model_folder_path("loras", legacy)

    routes.register(PromptServer.instance)
    too_old = routes.comfy_too_old()
    if too_old:
        logging.warning("[ZenTrainer] %s", too_old)


class ZenTrainerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .zentrainer.nodes import NODES

        return NODES


async def comfy_entrypoint() -> ComfyExtension:
    try:
        _setup()
    except Exception:
        logging.exception("[ZenTrainer] setup failed; nodes will load but the app API is unavailable")
    return ZenTrainerExtension()


WEB_DIRECTORY = "./js"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
