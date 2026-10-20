"""The LoRA library behind zenkit-ui's LoRA picker (LoraPicker / LoraBrowser): listing with
previews, details from Lora Manager's sidecar metadata and Civitai info, and bookmarks.

meta.py, lora_manager.py and favorites.py are ComfyUI-NynxzNodes' nodes/lora modules (same
code, so both packs read a LoRA the same way); only where they keep their files differs:
ZenTrainer's under <user>/zentrainer/. The routes are in server/routes/loras.py (/zentrainer/api/loras...).
"""
