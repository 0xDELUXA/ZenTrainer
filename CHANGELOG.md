# Changelog

What changed in each ZenTrainer release, newest first. Versions are the ones the ComfyUI Registry
and Manager show.

## Unreleased

### Training
- **Fixed: "illegal memory access" with weights offloaded:** a streamed layer's weights could be
  freed while its backward was still queued on the GPU, and the run crashed a step or two later
  (often right after a resume or once the weights moved between CPU and GPU). Most likely with
  Automagic and EMA, whose long optimizer step lets the CPU run far ahead of the GPU. The backward
  now keeps its own copy of each streamed weight (a copy on the GPU, not over PCIe).
- **Fixed: out of VRAM after samples with weights offloaded:** coming back from a sample pass, the
  model went back onto the card between the sampler's leftover cache fragments, and the next step
  ran out of VRAM; each time used one of the run's three retries, so a long run with samples
  stopped (Krea 2 int8 on 16 GB: at step 500). The cache is cleared first now, and the retries
  count only in a row.
- **Pausing lets ComfyUI work:** a paused run that sees a workflow queued in ComfyUI saves its
  state and lets go of ComfyUI (its queue and the GPU), and stays paused; Resume continues from
  there (a model reload). Nothing else in ZenTrainer's queue starts meanwhile. Needs resume states
  (Output → Resume states, on by default).
- **Faster by default on int8 models:** the int8 backward (W8A8) is now on, so the frozen layers'
  gradients run on int8 tensor cores too: about 16% faster on Krea 2 int8 at 1024. It rounds the
  gradient to int8 (stochastically, so unbiased); turn it off in Memory to train with a bf16
  backward.
- **Compile blocks is Auto by default:** on for runs of 500 steps or more when Triton is installed
  (Krea 2 at 1024: about 14% faster once compiled). Never with weights offloaded, even when turned
  on (or set by a recipe): weights moving between CPU and GPU crashed compiled runs with an illegal
  memory access on a 16 GB card.
- **Experimental: fused q/k norm + rope for Krea 2** (memory.fused_qk_rope, off by default): the
  attention's norms and rope run as one op with its own backward instead of six separate ones:
  6–7% a step on Krea 2 and MOSS at 1024. Same maths, slightly less rounding. Skipped when the
  blocks compile, which fuse them already (and faster with it off).
- **Steps wait for the GPU once instead of six times:** the loss, gradient norm and noise chart are
  read together at the step's end, and the run page's phase timings come from every 10th step.
  The GPU no longer idles between phases.
- **"int8 forward" off now does what it says:** int8 models dequantize to bf16 for every matmul.
  ComfyUI had started multiplying in int8 whenever a layer trains, so turning it off changed
  nothing.

### Datasets and captions
- **A datasets folder of their own** (Settings → Storage): input/datasets/, so datasets stop
  sharing ComfyUI's input folder with everything else uploaded there, or any folder. Nothing is
  moved: Settings lists the datasets still in the old place, and runs still find them there.
- **Caption length, cut-offs and re-rolls:** captions are no longer cut at 300 tokens: most styles
  have no limit now (up to 4096; the model ends a caption itself), a short caption keeps 300, and
  a Length choice when captioning sets one (600, 1200, or no limit); captions that ran out of room are flagged, listed on the job ("N cut off",
  to review them or re-caption them with no limit), and marked in the try-on; and Wording: Vary
  samples each run with a new seed, so captioning an image again words it differently. Advanced
  adds a fixed seed (repeatable re-rolls), temperature, top-p, top-k, a repetition penalty (for a
  model that loops until it's cut off) and a custom length; a job's log says what it used.
- **Dataset analysis** (Health → Full analysis): the pictures worth a look, grouped by issue; how
  training buckets and crops them at a resolution you pick, with every crop shown; which phrases
  most captions repeat (things the LoRA will lean towards); the spread of brightness, contrast,
  sharpness, size and shape; and near-duplicate pairs side by side.
- **Health finds copies properly:** near-duplicates are re-saved, resized, lightly edited or
  mirrored copies, and different pictures that only share a composition no longer count. Bars
  are real letterboxing, not a plain sky. New checks for transparency and heavy JPEG compression,
  and sizes follow a photo's rotation as training does. Health scores move once as a result.
- **A choice of face model** (Settings → Faces): InsightFace's standard one, or the larger
  antelopev2 (ArcFace R100), which tells similar faces apart better and is about 1.5 times slower.
  A run keeps the model it started with. antelopev2's download, which unpacks a folder too deep,
  now loads as it is.

### Install and run
- **`zt` no longer says a port is in use just after a server stops.** Its check counted the
  stopped server's lingering connections (about a minute) as the port being taken, so
  `zt train --port` right after a restart refused a free port.

## 0.2.0 (2026-10-24)

### Install and run
- **Start scripts.** From a clone, `start-linux.sh` or `start-windows.bat` is all you run: they
  install uv if it's missing (asking first), which brings the right Python, then open `zt`.
  Only git is needed beforehand.
- **`zt`, the command line, rebuilt.** On its own it shows a menu: a first-run setup that asks
  where your models are and which PyTorch build to use, an update check, start, and more.
  Arrow keys throughout.
- **`zt monitor`:** a live dashboard in the terminal with the queue, runs, progress, loss and
  memory, and keys to pause, stop or save a checkpoint. It only needs ZenTrainer's API, so it
  can watch a machine you're connected to remotely.
- **Safe updates.** `zt` checks for a new version when it starts and updates only a clean
  checkout. Your own changes or commits are never touched; it says why it skipped instead.
- **Ports.** If ZenTrainer is already running, `zt start` says where. If another program has
  the port, it uses the next free one.
- **No Node.js needed** to run from a clone: the web app comes prebuilt.
- **A smaller node pack:** the Registry download is the engine and the built app only (3 MB).

### Training
- **Resolution stages:** train at a lower resolution first and blend into the full one.
- **A Focus slider** for where training spends its steps, plus timestep types, bias and
  ComfyUI's schedulers, with charts of the noise levels they pick.
- **Base LoRA:** train a new LoRA on top of an existing one. It stays on in samples and is
  recorded in the file.
- **Skip blocks:** leave chosen blocks as the base model's.
- **Automagic v3** batches its step across tensors (faster) and has an optional
  learning-rate floor.
- **Offloading** measures how much VRAM the first steps need and fits as many weights on the
  card as are left. Each move is a trial, undone if it makes steps slower.
- **Train a finished run further,** and download its LoRAs from the run page.
- LoRA files are saved in bf16 by default: half the size of fp32.

### Models
- **MOSS Image edits:** train edits from reference pictures, with nodes to sample them.
  Edit pairs can also be channel inputs.
- **Fibo 2.0:** a profile with Bria's own LoRA recipe, and captions in its structured JSON.
- **The model picker:** makers' logos, a catalogue of corrections, and Hugging Face re-listed
  for new models.

### Datasets and captions
- **Storage:** a ZenTrainer folder of its own, so datasets and results don't have to fill
  ComfyUI's input and output folders.
- **Big datasets:** a cache on disk for datasets bigger than RAM, and large folders open fast.
  Caption encodings are cached once and shared between runs.
- **The mask editor:** an image list, a box tool, and one mask applied to many images (with
  undo).
- **Control sets:** named layers of input pictures beside a dataset, for edit training.
- Tidier dataset cards, list and side panel; captions and health in one badge.

### Runs and samples
- **Compare runs** side by side, from the Jobs page.
- **The loss chart** shows speed and time, and its lines can be hidden. Likeness shows its
  change since the first scored samples.
- **Edit runs** show their samples beside the input pictures.
- **The sidebar** shows the queue on the running run's card, and what's loaded in VRAM.
- **Swap** to a queued run: the current one saves and goes back in the queue.

### Simple mode
- **Recipe pages** that guide you through a run, check your photos, and show the results.

### Tools (experimental)
- **Compact:** shrink a LoRA or LoKr to a size you pick. Rank goes to the layers that change
  most, optionally weighed against the base model.
- **Inspect a model file:** every layer, shape, precision and metadata entry in a .safetensors
  file. Save a copy with other metadata, another precision, or layers left out.

### Fixes
- Opening the app over plain http on a LAN address works again.
- Skip-layer guidance buttons are clickable again.
- Batches whose captions can't stack run as separate calls instead of failing.
- SAM3 lets go of its VRAM when training or captioning starts.

### Upgrading
- **Tools, Apps and Gallery are experimental pages, hidden by default.** Turn them on under
  Settings → Appearance → Experimental pages.
- **From a clone:** `scripts/zt` still works, and is now `zt`. Run `./start-linux.sh` or
  `start-windows.bat` once to set up its environment.
- **The pinned PyTorch is 2.13.0** (in `comfy.lock`). `zt setup` and `zt update` install that
  version; `zt doctor` says when yours differs.

## 0.1.0 (2026-10-18)

The first release on the ComfyUI Registry.
