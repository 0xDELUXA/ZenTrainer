<script setup lang="ts">
// Pick a model as one thing (Krea 2, Z-Image, SDXL…) instead of three files. Families come
// from ComfyUI's own templates plus the builds in each family's Hugging Face repo; every part
// shows whether it's installed and downloads in place if not. "Custom" keeps the manual
// file-by-file picker for anything the catalog doesn't know.
import { ZenButton, ZenCombo } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { engine, type ModelFamily, type ModelFile, type ModelFolder, type ModelProfile, type SampleDefaults } from '@/api/engine'
import { fmtAdded, fmtBytes, finished, hfToken, isNew, startDownloadSync, targets } from '@/stores/downloads'
import FileState from './FileState.vue'
import StarButton from './StarButton.vue'
import { starredFirst } from '@/stores/favorites'
import type { ModelValue } from './ModelPicker.vue'

export interface SampleSuggestion extends SampleDefaults {
  family: string
  folder: ModelFolder
  size: number | null
  /** the name ComfyUI lists it by, once installed */
  installed: string | null
}

const model = defineModel<ModelValue>({ required: true })
const emit = defineEmits<{
  /** files still to download before this model can train (0 = ready) */
  missing: [number]
  /** the family's suggested sampling model + settings, when it differs from the trained one */
  sampleSuggestion: [SampleSuggestion | null]
  custom: [boolean]
  /** the family's hand-picked settings (null = none); picked = the user chose this family just now */
  profile: [ModelProfile | null, boolean]
  /** the picked family's id (null: none, or custom files) */
  family: [string | null]
}>()

const CUSTOM = '__custom'
const families = ref<ModelFamily[]>([])
const famId = ref<string>('')
let userPicked = false
const fam = ref<ModelFamily | null>(null)
const loadingRepo = ref(false)
const error = ref<string | null>(null)

const PARTS: Record<'split' | 'checkpoint', { folder: ModelFolder; label: string; key: keyof ModelValue }[]> = {
  split: [
    { folder: 'diffusion_models', label: 'Diffusion model', key: 'unet_name' },
    { folder: 'text_encoders', label: 'Text encoder', key: 'clip_name' },
    { folder: 'vae', label: 'VAE', key: 'vae_name' },
  ],
  checkpoint: [{ folder: 'checkpoints', label: 'Checkpoint', key: 'ckpt_name' }],
}
const parts = computed(() => (fam.value ? PARTS[fam.value.kind] : []))

const base = (n?: string | null) => (n ?? '').replace(/\\/g, '/').split('/').pop() ?? ''
const fileOf = (folder: ModelFolder, name?: string | null) => fam.value?.files[folder]?.find((f) => f.name === base(name))

// which family the current spec already uses (a duplicated run, a preset)
function detect(list: ModelFamily[]): string {
  const m = model.value
  const key = base(m.kind === 'checkpoint' ? m.ckpt_name : m.unet_name)
  if (!key) return ''
  const hit = list.find((f) => (f.files[m.kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'] ?? []).some((x) => x.name === key))
  return hit?.id ?? CUSTOM
}

async function loadFamilies() {
  try {
    families.value = await engine.families()
    redetect()
  } catch (e) {
    error.value = String(e)
  }
}
// the spec's files changed from outside (a duplicated run loading, a preset): follow them,
// unless they still belong to the family already showing
function redetect() {
  if (!families.value.length) return
  const m = model.value
  const key = base(m.kind === 'checkpoint' ? m.ckpt_name : m.unet_name)
  const cur = families.value.find((f) => f.id === famId.value)
  if (cur && (cur.files[m.kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'] ?? []).some((x) => x.name === key)) return
  if (!key && famId.value) return
  const next = detect(families.value)
  if (next !== famId.value) {
    userPicked = false
    famId.value = next
  }
}
watch(() => [model.value.kind, model.value.ckpt_name, model.value.unet_name], redetect)

async function open(id: string, refresh = false) {
  if (!id || id === CUSTOM) {
    fam.value = null
    return
  }
  fam.value = families.value.find((f) => f.id === id) ?? null
  loadingRepo.value = true
  try {
    const full = await engine.family(id, true, refresh)
    if (famId.value === id) fam.value = full
  } catch {
    // offline: the template files are still there
  } finally {
    loadingRepo.value = false
  }
}

// choosing a family fills the spec with its recommended files
function apply(f: ModelFamily) {
  const rec = f.recommended.train
  const name = (folder: ModelFolder) => {
    const n = rec[folder]
    const file = n ? f.files[folder]?.find((x) => x.name === n) : undefined
    return file ? (file.installed ?? file.name) : null
  }
  model.value =
    f.kind === 'checkpoint'
      ? { ...model.value, kind: 'checkpoint', ckpt_name: name('checkpoints') }
      : {
          ...model.value,
          kind: 'split',
          unet_name: name('diffusion_models'),
          clip_name: name('text_encoders'),
          clip_type: f.clip_type,
          vae_name: name('vae'),
        }
}

watch(famId, async (id, old) => {
  emit('custom', id === CUSTOM)
  await open(id)
  // a family picked by hand fills the files; one detected from the spec keeps them
  const picked = !!fam.value && userPicked && id !== old
  if (picked) apply(fam.value!)
  emit('profile', fam.value?.profile ?? null, picked)
  emit('family', fam.value?.id ?? null)
})
const pick = (id: string | number) => {
  userPicked = true
  famId.value = String(id)
}

function setFile(folder: ModelFolder, key: keyof ModelValue, name: string | number) {
  const f = fileOf(folder, String(name))
  model.value = { ...model.value, [key]: f ? (f.installed ?? f.name) : String(name) }
}

// install state changes as downloads land: refresh the family and swap in the local names
watch(finished, async () => {
  if (!fam.value) return
  const full = await engine.family(fam.value.id, true).catch(() => null)
  if (!full) return
  fam.value = full
  for (const p of parts.value) {
    const f = fileOf(p.folder, model.value[p.key] as string)
    if (f?.installed && model.value[p.key] !== f.installed) model.value = { ...model.value, [p.key]: f.installed }
  }
  families.value = families.value.map((x) => (x.id === full.id ? { ...x, missing: full.missing } : x))
})

const missingParts = computed(() => parts.value.filter((p) => !fileOf(p.folder, model.value[p.key] as string)?.installed))
watch(
  [missingParts, fam],
  () => emit('missing', fam.value ? missingParts.value.length : 0),
  { immediate: true },
)
// the suggestion only makes sense while training on a different (base) build
watch(
  [fam, () => model.value.unet_name, () => model.value.ckpt_name],
  () => {
    const f = fam.value
    const s = f?.recommended.sample ?? null
    if (!f || !s) return emit('sampleSuggestion', null)
    const folder: ModelFolder = f.kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'
    const trained = base(f.kind === 'checkpoint' ? model.value.ckpt_name : model.value.unet_name)
    // any installed distilled build will do (the template's precision first); else the template's
    const tmpl = fileOf(folder, s.model)
    const prec = (n: string) => n.match(/bf16|fp16|fp8\w*|int8\w*|nvfp4|mxfp8/)?.[0] ?? ''
    const distilled = (f.files[folder] ?? []).filter((x) => x.distilled && x.name !== trained)
    const pick =
      distilled.find((x) => x.installed && prec(x.name) === prec(s.model)) ?? distilled.find((x) => x.installed) ?? tmpl
    if (!pick || pick.name === trained) return emit('sampleSuggestion', null)
    emit('sampleSuggestion', { ...s, model: pick.name, family: f.id, folder, size: pick.size, installed: pick.installed })
  },
  { immediate: true },
)

async function download(folder: ModelFolder, name: string) {
  if (!fam.value) return
  error.value = null
  try {
    await engine.download(fam.value.id, folder, name)
  } catch (e) {
    error.value = String(e)
  }
}
async function downloadAll() {
  for (const p of missingParts.value) {
    const f = fileOf(p.folder, model.value[p.key] as string)
    if (f) await download(p.folder, f.name)
  }
}
const missingBytes = computed(() =>
  missingParts.value.reduce((n, p) => n + (fileOf(p.folder, model.value[p.key] as string)?.size ?? 0), 0),
)

const familyItems = computed(() => [
  ...starredFirst(families.value, (f) => f.id).map((f) => ({
    value: f.id,
    label: f.name,
    keywords: `${f.name} ${f.clip_type ?? ''} ${f.repos.join(' ')}`,
    f,
  })),
  { value: CUSTOM, label: 'Custom: pick the files yourself', keywords: 'custom manual files', f: null },
])
const fileItems = (folder: ModelFolder) =>
  (fam.value?.files[folder] ?? []).map((f) => ({ value: f.name, label: f.name, keywords: f.name, file: f }))
const statusOf = (f: ModelFamily) => {
  if (!f.missing.length) return 'ready'
  const bytes = f.missing.reduce((n, m) => n + (m.size ?? 0), 0)
  return `${f.missing.length} to download${bytes ? ` · ${fmtBytes(bytes)}` : ''}`
}

onMounted(() => {
  startDownloadSync()
  loadFamilies()
})
defineExpose({ refresh: () => fam.value && open(fam.value.id, true) })
</script>

<template>
  <div class="fp">
    <ZenCombo
      :model-value="famId"
      :items="familyItems"
      placeholder="Choose a model…"
      :sorted="false"
      :item-height="40"
      :menu-width="460"
      @update:model-value="pick"
    >
      <template #option="{ item }">
        <div class="fam-opt">
          <StarButton v-if="item.f" :id="(item.f as ModelFamily).id" :name="item.label" />
          <b>{{ item.label }}</b>
          <i
            v-if="item.f && (item.f as ModelFamily).profile"
            class="mdi tick"
            :class="(item.f as ModelFamily).profile!.status === 'verified' ? 'mdi-check-decagram' : 'mdi-check-circle-outline'"
            :title="(item.f as ModelFamily).profile!.status === 'verified' ? 'Tuned: trained and checked with hand-picked settings' : 'Hand-picked settings from the model makers'"
          />
          <span v-if="item.f && isNew((item.f as ModelFamily).added)" class="tag new" :title="`Added ${fmtAdded((item.f as ModelFamily).added)}`">new</span>
          <span v-if="item.f && (item.f as ModelFamily).source === 'repo'" class="tag" title="ComfyUI has no example workflow for this model yet, so its files come straight from its Hugging Face repo, and there are no suggested sampler settings for previews">from repo</span>
          <span v-if="item.f" class="fam-status" :class="{ ready: !(item.f as ModelFamily).missing.length }">
            <i :class="(item.f as ModelFamily).missing.length ? 'mdi mdi-download-outline' : 'mdi mdi-check-circle'" />
            {{ statusOf(item.f as ModelFamily) }}
          </span>
        </div>
      </template>
    </ZenCombo>

    <div v-if="fam" class="parts">
      <div v-for="p in parts" :key="p.folder" class="part">
        <span class="p-label">{{ p.label }}</span>
        <ZenCombo
          :model-value="base(model[p.key] as string)"
          :items="fileItems(p.folder)"
          :item-height="34"
          :menu-width="440"
          @update:model-value="setFile(p.folder, p.key, $event)"
        >
          <template #option="{ item }">
            <div class="file-opt">
              <i :class="(item.file as ModelFile).installed ? 'mdi mdi-check-circle ok' : 'mdi mdi-cloud-download-outline muted'" />
              <span class="f-name">{{ item.label }}</span>
              <span v-if="(item.file as ModelFile).distilled" class="tag" title="Distilled (turbo/lightning): fast to sample, not a good base to train on">turbo</span>
              <span class="f-size mono">{{ fmtBytes((item.file as ModelFile).size) }}</span>
            </div>
          </template>
        </ZenCombo>
        <!-- fixed-width status column: installed / download / progress, without shifting the row -->
        <div class="p-state">
          <template v-for="f in [fileOf(p.folder, model[p.key] as string)]" :key="p.folder">
            <FileState v-if="f && fam" :family="fam.id" :file="f" @error="error = $event" />
            <span v-else class="muted">not in the catalog</span>
          </template>
        </div>
      </div>

      <div class="fp-foot">
        <template v-if="missingParts.length > 1">
          <ZenButton class="dl-all" size="sm" icon="mdi mdi-download-multiple" @click="downloadAll">Download all · {{ fmtBytes(missingBytes) }}</ZenButton>
        </template>
        <span class="muted small ellipsis" :title="targets[parts[0]?.folder ?? 'diffusion_models']">
          <template v-if="missingParts.length">Downloads go to {{ targets[parts[0]?.folder ?? 'diffusion_models'] ?? 'your models folder' }}</template>
          <template v-else-if="fam.repos.length">From {{ fam.repos.join(', ') }}</template>
        </span>
        <span class="spacer" />
        <span v-if="loadingRepo" class="muted small"><i class="mdi mdi-loading mdi-spin" /> listing builds…</span>
        <button v-else class="link small" title="Re-list the builds on Hugging Face" @click="open(fam.id, true)"><i class="mdi mdi-refresh" /></button>
      </div>
    </div>
    <div v-if="fam?.gated && missingParts.length" class="gate small">
      <i class="mdi mdi-lock-outline" />
      <span>
        Gated: accept the license on
        <a :href="`https://huggingface.co/${fam.repos[0]}`" target="_blank" rel="noopener">Hugging Face</a>
        <template v-if="!hfToken">, then set <span class="mono">HF_TOKEN</span> (a read token) and restart ZenTrainer</template>
        <template v-else> with the account your <span class="mono">HF_TOKEN</span> belongs to</template>.
      </span>
    </div>
    <div v-if="error" class="err small mono">{{ error }}</div>
  </div>
</template>

<style scoped>
.fp {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.fam-opt {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
}
.fam-opt b {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
}
.fam-status {
  font-size: 11px;
  color: var(--zen-muted);
  white-space: nowrap;
}
.fam-status.ready,
.ok {
  color: var(--zen-ok);
}
.parts {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: var(--r-lg);
  background: color-mix(in srgb, var(--zen-surface-2) 50%, transparent);
}
.part {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr) 150px;
  align-items: center;
  gap: 10px;
}
.p-label {
  font-size: 12px;
  color: var(--zen-muted);
}
.p-state {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  font-size: 12px;
  white-space: nowrap;
  min-width: 0;
}
.file-opt {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-width: 0;
}
.f-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.f-size {
  font-size: 11px;
  color: var(--zen-muted);
}
.gate {
  display: flex;
  gap: 8px;
  align-items: baseline;
  padding: 8px 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-warn) 10%, transparent);
}
.gate i {
  color: var(--zen-warn);
}
.gate a {
  color: var(--zen-accent);
}
.tick {
  color: var(--zen-accent);
  font-size: 14px;
  margin-left: -4px;
}
.tag.new {
  background: color-mix(in srgb, var(--zen-ok) 18%, transparent);
  color: var(--zen-ok);
}
.tag {
  font-size: 10px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-warn) 18%, transparent);
  color: var(--zen-warn);
}
.fp-foot {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 26px;
  margin-top: 2px;
}
/* one line: the folder path beside it truncates instead */
.dl-all {
  flex-shrink: 0;
  white-space: nowrap;
}
.small {
  font-size: 11px;
}
.ellipsis {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  font: inherit;
}
.link:hover {
  color: var(--zen-accent);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 700px) {
  .part {
    grid-template-columns: 1fr auto;
  }
  .p-label {
    grid-column: 1 / -1;
  }
}
</style>
