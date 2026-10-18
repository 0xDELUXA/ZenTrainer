Theme packs bundled with ZenTrainer so the app is themed without ComfyUI-ZenKit.

Synced from ZenKit (`plugins/ComfyUI-ZenKit/themes`); `comfyui-brand` (shown as Comfy Classic, ComfyUI's previous palette) is from lora.moe; `maroon-volt` is ComfyUI's current one.
At runtime the engine also reads a linked ZenKit's themes and the shared user drop-in
folder `<ComfyUI user dir>/zenkit/themes`, so a theme added there shows up in both apps.
Later sources win on an id clash: bundled < ZenKit < user.
