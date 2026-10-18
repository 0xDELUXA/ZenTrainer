# Apps — design

An **app** is a ComfyUI workflow with a purpose-built interface: real controls, a layout,
outputs presented the way they're meant to be seen. ComfyUI's app mode mounts a
workflow's widgets in a sidebar; that is a form, not an interface. Apps here are built
from a small manifest, run through the same queue as training, and leave their results
in one Assets library.

Training is the first app with a hand-written view. Captioning and "Test LoRA" are the
next two, and they validate the format before anyone else writes one.

## The app format (current)

An app is a folder: `app.json`, its workflows, optionally a cover image. Built-ins live in
`engine/apps/<id>/`, the user's in `<user>/zentrainer/apps/<id>/`. Three levels, each
building on the last:

1. **Declarative** (most apps): `app.json` + workflows, drawn by a shared layout
   (`"layout": "generate"`). No code.
2. **A special component in a shared layout**: an input uses a richer component from the
   shared library (planned: `bbox-editor`, `mask-painter`, `compare`).
3. **A custom view** (rare): `"view": "ui/View.vue"`, compiled into the app for built-ins,
   written against the same app API the shared layouts use (planned).

`app.json`:

- `workflows`: files beside it, API or editor format (a normal ComfyUI save is converted on
  load). `when: {"family": "<model family id>"}` gives one workflow per model, picked by a
  `family` input; `when: {"layout": "checkpoint" | "split"}` is the older file-picker form.
- `inputs`: `component` (`family`, `prompt`, `text`, `size`, `loras`, `seed`, `number`,
  `slider`, `toggle`, `choice`, ...), `bind` to `"Node title.input"` (alternatives:
  `"Sampler.seed | Noise.noise_seed"`, the first the workflow has), `optional` (bound only
  where the node exists; the UI hides it otherwise), `default` (none = the workflow's own
  value), `tip`, `section` (`main` | `advanced`), `sweep`.
- `outputs`: `{from: "Save", display: "gallery"}`. `cover`: an image beside `app.json`.

Every binding is checked against every workflow when the app loads, so a broken app fails on
the Apps page, not mid-run. `GET /zentrainer/api/apps/<id>/defaults?family=` returns the
selected workflow's own values and which inputs it takes; the form starts there.

**Text to Image** is the first app in this format: 16 per-model workflows in
`engine/apps/text-to-image/workflows/`, first made from ComfyUI's text-to-image templates
(converted, nodes titled Prompt / Negative / Size / Sampler / Save, fixed switches folded,
nodes that don't feed the image pruned, a LoRA placeholder added) and ours to tune from
there. Each starts from its template's settings (8 steps for a turbo build, 25 for a base
one); the LLM prompt enhancers some templates carry are left out for now.

ComfyUI's App-mode templates and the user's App-mode workflows (`apps/catalog.py`,
converted by `apps/workflow.py`, checked against all 601 image templates) are what the
importer will turn into app folders. The manifest format below still describes Test LoRA.

## Goals

- A useful app from a workflow in minutes, without writing code.
- Interfaces that fit the task: a mask painter for inpainting, a before/after slider for
  upscales, a grid for LoRA sweeps, a video player for video. Not just widgets.
- One app, many models: pick the matching workflow for the chosen model family.
- Honest about requirements: missing models or nodes are shown with a fix before Run.
- Every output remembers how it was made; "remix" reopens the app with those values.
- The same app runs locally or on Comfy Cloud (`@comfyorg/sdk`).

## An app is a folder

```
apps/inpaint/
  app.json           manifest (below)
  workflows/
    sdxl.json        API-format workflows; titles mark the nodes the manifest binds to
    flux.json
  icon.svg
  view.vue           optional: a custom region, for what the built-in controls can't do
```

Built-ins ship with the engine; the user's live in `<user>/zentrainer/apps/`.

## Manifest

```jsonc
{
  "id": "inpaint",
  "name": "Inpaint",
  "description": "Repaint part of an image.",
  "category": "edit",                       // generate | edit | upscale | video | caption | train
  "workflows": [
    // the first one whose `when` matches is used; `model` names the input that decides
    { "file": "workflows/flux.json", "when": { "family": ["flux", "flux2"] } },
    { "file": "workflows/sdxl.json", "when": { "family": ["sdxl"] } }
  ],
  "inputs": [
    { "id": "model",  "type": "model",  "folder": "checkpoints", "bind": "Load Model.ckpt_name" },
    { "id": "image",  "type": "image",  "bind": "Source.image", "required": true },
    { "id": "mask",   "type": "mask",   "of": "image", "bind": "Mask.image" },
    { "id": "prompt", "type": "prompt", "bind": "Positive.text", "placeholder": "what should be there" },
    { "id": "loras",  "type": "loras",  "bind": "LoRA Stack" },
    { "id": "denoise","type": "slider", "bind": "Sampler.denoise", "min": 0, "max": 1, "step": 0.05, "default": 0.85,
      "group": "advanced" },
    { "id": "seed",   "type": "seed",   "bind": "Sampler.seed" }
  ],
  "outputs": [
    { "id": "result", "from": "Save", "view": "compare", "against": "image" }
  ],
  "layout": {
    "stage": ["image", "mask"],              // the big canvas: image + mask painter
    "inspector": ["model", "prompt", "loras", "denoise", "seed"],
    "result": "result"
  },
  "requires": {
    "nodes": ["InpaintModelConditioning"],
    "models": [{ "folder": "checkpoints", "match": "inpaint", "hint": "huggingface:…" }]
  }
}
```

- **Binding** is `"<node title>.<input>"`. Titles are stable across workflow edits;
  node ids aren't. A binding to a node that doesn't exist is a load-time error, not a
  runtime surprise.
- **Input types** are a fixed, growing library: `prompt`, `text`, `number`, `slider`,
  `seed`, `toggle`, `choice`, `model` (searchable, filtered by folder or family),
  `loras` (picker with strengths; expands to a LoRA chain), `image` (drop, paste,
  pick from Assets), `mask` (painter over another image input), `images`, `video`,
  `audio`, `resolution` (presets plus aspect lock), `dataset`.
- **Output views:** `image`, `gallery`, `grid` (from a sweep), `compare` (slider
  against an input), `video`, `audio`, `text`.
- **Layout regions:** `stage` (main canvas), `inspector` (controls), `result`, and
  `history` (this app's recent outputs). Unlisted inputs fall into the inspector;
  `group: "advanced"` collapses them.
- **Sweeps:** any `number`, `seed`, `choice` or `loras` input can take a list, which
  expands to a batch and lands in a `grid` view. That's "Test LoRA" for free.

## Runtime

1. Pick the workflow (`when` against the chosen model's family, found from the
   model's metadata or the loader's detected type).
2. Validate inputs and requirements (object_info + folder_paths), then show what's
   missing with a fix: a model download, a node pack.
3. Compile: copy the workflow, set every bound input, expand sweeps and LoRA chains.
4. Submit as a **job** to the same queue as training, with priorities. A generation can
   run between training runs, or wait for one.
5. Stream progress and latent previews (the same events training samples use).
6. Write outputs to Assets with provenance: app, workflow, inputs, seed, models, LoRAs.

## Assets

One library for everything ZenTrainer makes: generations, training samples, captions,
trained LoRAs. Filter by app, model or LoRA; open in the lightbox; "remix" reopens the
source app with the same inputs; "use as input" sends an image to another app.

## Creating apps

- **From a workflow:** load a workflow and pick the widgets to expose (they become
  inputs with guessed types). Arrange regions, preview the live UI, save. This writes
  the same manifest a hand-made app uses.
- **By hand:** write `app.json`. Validation errors name the binding and the reason.
- **Custom regions:** `view.vue` mounts in a region and gets the app's inputs and
  outputs as props. This is the escape hatch, not the default.

## Onboarding

First run asks "What brings you here?": **generate**, **train**, or **both**. The answer
sets the sidebar order and the home page:

- generate: Apps and Assets first, with training under "More";
- train: Runs, Datasets and Queue first, with apps under "Create";
- both: the current layout.

It can be changed later in Settings. The same screen checks the setup: the GPU, ComfyUI
version against the pin, and models found per family, then offers presets and apps that
match what's installed ("Krea 2 found: Krea 2 likeness preset, Krea 2 generate app").

## Order of work

1. Manifest types + loader + validation (engine), with tests.
2. Runtime: compile, submit to the job queue, outputs to Assets.
3. Generic app view: regions, the input library, output views.
4. Built-ins: Generate, Test LoRA (sweep grid), Caption, Inpaint, Upscale.
5. Assets library with provenance and remix.
6. "Create app from workflow".
7. Onboarding + sidebar profiles.
8. Comfy Cloud execution through `@comfyorg/sdk`.
