# ZenTrainer in a container

Train on a rented GPU (RunPod, Vast.ai, a cloud VM) or any Linux machine with Docker and an
NVIDIA GPU. The image is the same pinned ComfyUI and engine a clone sets up with `zt setup`,
with the web app built in and a password in front of it.

## What's in the container

| Port | What | Who can reach it |
|---|---|---|
| 8188 | the app (`/` opens `/zentrainer/`) | anyone with the user name and password |
| 22 | SSH, when `PUBLIC_KEY` is set | your key only, as user `zen`: no passwords, no root |
| 8189 | ComfyUI itself | only from inside the container |

ComfyUI has no login of its own, so it listens on localhost only and every request from outside goes
through Caddy, which asks for the password (WebSockets included). ComfyUI and Caddy run as the
unprivileged user `zen`. Each container makes its own SSH host keys; none ship in the image.

Port 8188 is plain HTTP: put HTTPS in front of it. RunPod's HTTP proxy does that for you. On a
machine of your own, reach it through an SSH tunnel (`ssh -L 8188:localhost:8188 …`) or a
reverse proxy with TLS, and don't leave 8188 open to the internet.

## Your data

Everything you make lives in `/workspace/zentrainer`, so mount a volume at `/workspace` (it says so
in the log if you didn't):

```
/workspace/zentrainer/
  input/      datasets (one folder each)
  output/     trained LoRAs (output/zentrainer/loras), samples
  user/       run history, settings, the queue
  models/     checkpoints/, diffusion_models/, vae/, text_encoders/, loras/, …
  insightface/  face models, downloaded when you first turn on faces for a dataset
  .password   the generated password, when you didn't set one
```

Models you download in the app land in `models/`, and you can put your own there too (scp, or
your provider's file browser). Updating the image or restarting the pod keeps all of it. Runs that
were training when the container stopped show as interrupted; resume them from their last
checkpoint.

## Settings (environment variables)

| Variable | |
|---|---|
| `ZT_PASSWORD` | the app's password, 12 characters or more. Unset: one is made, saved in `/workspace/zentrainer/.password` and printed in the log at every start |
| `ZT_USER` | the app's user name (default `zentrainer`) |
| `PUBLIC_KEY` | SSH public keys, one per line, for logging in as `zen` (RunPod sets it from your account) |
| `HF_TOKEN` | a Hugging Face token, for gated model repositories |
| `ZT_COMFY_ARGS` | more ComfyUI flags, e.g. `--reserve-vram 2` |
| `ZT_DATA` | where your data lives (default `/workspace/zentrainer`) |

## The image

Each release is published to GHCR by CI (`.github/workflows/docker-image.yml`):

| Tag | |
|---|---|
| `ghcr.io/nynxz/zentrainer:latest` | the newest release |
| `ghcr.io/nynxz/zentrainer:0.3.1`, … | one release, for a template that shouldn't move |

It runs PyTorch's CUDA 13.0 build, so it needs an NVIDIA driver for CUDA 13.0 or newer, and covers
current GPUs including Blackwell. The container checks the driver at start and says so if it's too
old.

To try a branch before a release, run the workflow by hand (Actions → docker-image → Run
workflow): it publishes `:<branch>`.

To build it yourself, from the repository root:

```sh
docker build -f docker/Dockerfile -t zentrainer .
```

`--build-arg TORCH=<build>` picks another of comfy.lock's torch builds, for older drivers.

## Run it locally

```sh
docker run --gpus all -p 127.0.0.1:8188:8188 -v zentrainer:/workspace \
  -e ZT_PASSWORD='something long' ghcr.io/nynxz/zentrainer:latest
```

Then open http://127.0.0.1:8188/.

## RunPod

The quickest way: [deploy the ZenTrainer template](https://console.runpod.io/hub/template/2amo4895gj?ref=oe5l65t0)
(a referral link: it supports ZenTrainer at no cost to you).
Pick a GPU with **Allowed CUDA versions** 13.0 and up, give it a volume, and deploy. The rest of
this section is what the template holds, for making your own.

Make a template (Templates → New Template) with:

- **Container image**: `ghcr.io/nynxz/zentrainer:latest` (or a pinned tag, see above)
- **Container disk**: 30 GB (the image, plus ComfyUI's temporary files)
- **Volume disk**: as much as your models and datasets need, mounted at `/workspace` (or attach
  a network volume when deploying)
- **Expose HTTP ports**: `8188`; **Expose TCP ports**: `22`
- **Environment variables**: `ZT_PASSWORD` = `{{ RUNPOD_SECRET_zentrainer_password }}` (make that
  secret under Settings → Secrets first), and `HF_TOKEN` the same way if you need gated models

Deploy it on a GPU with **Allowed CUDA versions** set to 13.0 and up. Once it's running, Connect → HTTP Service 8188 opens the app. Log in with
`zentrainer` and your password. Without `ZT_PASSWORD`, the generated one is in the pod's logs.

For a terminal, `ssh zen@<pod ip> -p <mapped port>` (Connect → SSH over exposed TCP) works with
the keys in your RunPod account. `zt monitor`, `zt runs`, `zt train spec.json` and the rest work
there as they do on your own machine.

Stop the pod when you're done: a stopped pod keeps the volume and stops billing for the GPU.
