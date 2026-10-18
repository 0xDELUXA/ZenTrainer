"""ZenTrainer as a ComfyUI node pack: clone this repository into ComfyUI/custom_nodes (or install
it from the Registry / Manager), restart ComfyUI, and open /zentrainer/ on its port.

The engine itself lives in engine/ (which `scripts/zt` links on its own for a pinned, separate
install); this file only hands ComfyUI the engine's entry point, so both ways in run the same code.
"""

try:
    from .engine import comfy_entrypoint
except ModuleNotFoundError as e:  # imported outside ComfyUI (tests, tools): nothing to register
    if e.name not in ("comfy_api", "comfy", "folder_paths", "server"):
        raise
else:
    WEB_DIRECTORY = "./engine/js"

    __all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
