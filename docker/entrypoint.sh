#!/usr/bin/env bash
# ZenTrainer's container entrypoint (docker/Dockerfile). Runs as root only to set things up, then:
#   - ComfyUI with ZenTrainer, as the unprivileged user zen, listening on 127.0.0.1:8189 only
#   - Caddy on :8188, the one way in from outside: a password on every request (basic auth)
#   - sshd on :22 when PUBLIC_KEY is set: key login as zen, no passwords, no root
# Everything the user makes (datasets, runs, LoRAs, downloaded models) lives in $ZT_DATA, on the
# volume, so a new image or a restarted container keeps it.
set -euo pipefail

DATA=${ZT_DATA:-/workspace/zentrainer}
ZT=/opt/zentrainer
ZT_USER=${ZT_USER:-zentrainer}

as_zen() { setpriv --reuid=zen --regid=zen --init-groups env HOME=/home/zen USER=zen "$@"; }
die() { echo "ZenTrainer: $*" >&2; exit 1; }

# ---- 1. the GPU matches this image's PyTorch ------------------------------------------------------
"$ZT/cli/.venv/bin/python" -I - "$ZT_TORCH" <<'EOF' || exit 1
import sys
from zt import comfy, repo

variant = sys.argv[1]
need = repo.lock()["torch"]["variants"].get(variant, {}).get("min_driver_cuda")
if need:
    have = comfy.driver_cuda()
    if not have:
        sys.exit("ZenTrainer: no NVIDIA GPU is visible in this container. Run it with a GPU (docker run --gpus all, or a GPU pod).")
    if have < tuple(int(x) for x in need.split(".")):
        sys.exit(f"ZenTrainer: this image's PyTorch ({variant}) needs an NVIDIA driver for CUDA {need} or newer, "
                 f"and this machine's supports {have[0]}.{have[1]}. Pick a machine with a newer driver (on RunPod: "
                 f"'Allowed CUDA versions' {need} and up), or use an image built for an older CUDA.")
for i, (name, mem) in enumerate(comfy.gpus()):
    print(f"ZenTrainer: gpu {i}: {name}, {mem / 2**30:.0f} GB")
EOF

# ---- 2. the user's data, on the volume ------------------------------------------------------------
read -ra folders < <("$ZT/cli/.venv/bin/python" -I -c 'from zt import repo; print(*repo.MODEL_FOLDERS)')
mkdir -p "$DATA"/{input,output,user,insightface} "${folders[@]/#/$DATA/models/}"
# InsightFace downloads its face models to ~/.insightface: on the volume, they download once
ln -sfnT "$DATA/insightface" /home/zen/.insightface
if [[ $(findmnt -n -o TARGET -T "$DATA") == / ]]; then
  echo "ZenTrainer: WARNING: $DATA is not on a volume, so everything in it is lost when the container is deleted." \
       "Mount a volume at /workspace (on RunPod: a volume or network volume)." >&2
fi
# files copied in as root (a cloud file browser, an older container) must stay usable by ComfyUI
find "$DATA" -xdev -path "$DATA/.password" -prune -o \( ! -user zen -o ! -group zen \) -exec chown -h zen:zen {} +

# models on the volume: ComfyUI loads comfyui/extra_model_paths.yaml by itself, and ZenTrainer's
# downloads go to the first writable folder outside ComfyUI's own models/, which is this one
{
  echo "# written by the container entrypoint at every start"
  echo "zentrainer_volume:"
  echo "  base_path: $DATA/models"
  echo "  is_default: true"
  for f in "${folders[@]}"; do echo "  $f: $f/"; done
} > "$ZT/comfyui/extra_model_paths.yaml"

# ---- 3. the password --------------------------------------------------------------------------------
if [[ -n ${ZT_PASSWORD:-} ]]; then
  (( ${#ZT_PASSWORD} >= 12 )) || die "ZT_PASSWORD is shorter than 12 characters; pick a longer one."
  password=$ZT_PASSWORD
else
  # none given: one made for this volume, kept so it survives restarts, shown in the container's logs
  if [[ ! -s $DATA/.password ]]; then
    (umask 077; head -c 18 /dev/urandom | base64 | tr '+/' 'xy' > "$DATA/.password")
  fi
  password=$(<"$DATA/.password")
  echo "ZenTrainer: no ZT_PASSWORD set, so this one was made (kept in $DATA/.password):"
  echo "ZenTrainer:   user $ZT_USER, password $password"
fi
hash=$(printf '%s\n' "$password" | caddy hash-password)  # stdin, so it never shows in ps
unset ZT_PASSWORD password

# ---- 4. SSH, key only ------------------------------------------------------------------------------
if [[ -n ${PUBLIC_KEY:-} ]]; then
  install -d -m 700 -o zen -g zen /home/zen/.ssh
  printf '%s\n' "$PUBLIC_KEY" > /home/zen/.ssh/authorized_keys
  chown zen:zen /home/zen/.ssh/authorized_keys
  chmod 600 /home/zen/.ssh/authorized_keys
  ssh-keygen -A >/dev/null  # this container's own host keys, never ones baked into the image
  cat > /etc/ssh/sshd_config.d/zentrainer.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
AllowUsers zen
EOF
  mkdir -p /run/sshd
  /usr/sbin/sshd
  echo "ZenTrainer: SSH on :22, as zen with your key"
fi

# ---- 5. run: the password proxy, and ComfyUI (started again if it dies) ----------------------------
as_zen env ZT_USER="$ZT_USER" ZT_PASSWORD_HASH="$hash" \
  caddy run --config /etc/zentrainer/Caddyfile --adapter caddyfile &
caddy=$!

read -ra extra <<< "${ZT_COMFY_ARGS:-}"  # more ComfyUI flags, e.g. --reserve-vram 2
start_comfy() {
  as_zen zt start --port 8189 -- \
    --input-directory "$DATA/input" --output-directory "$DATA/output" --user-directory "$DATA/user" "${extra[@]}" &
  comfy=$!
}

stopping=
# a run cut off by a stop is marked interrupted at the next start and resumes from its last checkpoint
trap 'stopping=1; kill -TERM "$comfy" "$caddy" 2>/dev/null' TERM INT
start_comfy
if [[ -n ${RUNPOD_POD_ID:-} ]]; then
  echo "ZenTrainer: open https://$RUNPOD_POD_ID-8188.proxy.runpod.net/ (user $ZT_USER)"
else
  echo "ZenTrainer: open port 8188 (user $ZT_USER)"
fi

while true; do
  wait -n "$caddy" "$comfy" || true
  [[ -n $stopping ]] && break
  if ! kill -0 "$caddy" 2>/dev/null; then
    echo "ZenTrainer: the password proxy stopped, so ComfyUI stops too" >&2
    kill -TERM "$comfy" 2>/dev/null
    wait "$comfy" || true
    exit 1
  fi
  echo "ZenTrainer: ComfyUI exited; starting it again in 5 seconds" >&2
  sleep 5
  start_comfy
done
wait "$comfy" || true
