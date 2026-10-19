<script setup lang="ts">
// The run form: presets on top (they layer), four numbered sections with the settings
// most runs touch, everything else under Advanced, and a sticky summary that says what
// you're about to do (and what looks off) next to the Start buttons. Model files, folders,
// samplers and schedulers come from ComfyUI's object_info; the rest is the harness's.
import { ZenButton, ZenCombo, ZenField, ZenInput, ZenNumber, ZenSwitch, ZenTip } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, reactive, ref, toRaw, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { inputChoices } from '@/api/comfy'
import AdapterPicker from '@/components/AdapterPicker.vue'
import SamplePromptList from '@/components/SamplePromptList.vue'
import FamilyPicker, { type SampleSuggestion } from '@/components/FamilyPicker.vue'
import { downloadFor, fmtBytes } from '@/downloads'
import { setSetting, settings } from '@/settings'
import {
  EngineError,
  defaultSpec,
  engine,
  promptText,
  thumbUrl,
  type DatasetInfo,
  type DatasetItem,
  type OptimizerInfo,
  type ModelProfile,
  type Preset,
  type TrainSpec,
} from '@/api/engine'

const router = useRouter()
const route = useRoute()
const spec = reactive(defaultSpec())
const submitting = ref(false)
const error = ref<string | null>(null)
const advanced = reactive({ training: false, samples: false, memory: false })

// ---- where checkpoints go: one of ComfyUI's loras folders ---------------------------------
const loraPlaces = ref<Awaited<ReturnType<typeof engine.loraPlaces>>>([])
const loraDir = computed({
  get: () => spec.output.lora_dir ?? loraPlaces.value.find((p) => p.default)?.path ?? '',
  set: (v: string) => {
    const place = loraPlaces.value.find((p) => p.path === v)
    spec.output.lora_dir = !place || place.default ? null : v
    setSetting('loraFolder', spec.output.lora_dir ?? '')
  },
})
const loraPlaceItems = computed(() =>
  loraPlaces.value.map((p) => ({ value: p.path, label: p.path, keywords: p.path, free: `${fmtBytes(p.free)} free` })),
)
const loraPlaceHint = computed(() => {
  const p = loraPlaces.value.find((x) => x.path === loraDir.value)
  if (!p) return 'One of ComfyUI’s LoRA folders'
  return p.default ? 'Inside this install. Your model library’s loras folder keeps them with the rest, visible to your other ComfyUI' : `In a zentrainer/ subfolder · ${fmtBytes(p.free)} free`
})
watch(() => spec.output.keep_states, (v) => setSetting('keepStates', v))

// ---- choices from the server ------------------------------------------------------------
const choices = reactive<Record<string, string[]>>({})
// the model picker: files still to download, custom (file-by-file) mode, the family's sampling suggestion
const modelMissing = ref(0)
const customModel = ref(false)
const sampleSuggestion = ref<SampleSuggestion | null>(null)
const usingSuggestion = computed(
  () => !!sampleSuggestion.value && spec.samples.model_name === sampleSuggestion.value.installed && spec.samples.steps === sampleSuggestion.value.steps,
)
// the family's hand-picked settings: applied when the family is picked, re-appliable after edits
const profile = ref<ModelProfile | null>(null)
const familyId = ref<string | null>(null)
function applyProfile(p: ModelProfile) {
  merge(spec as unknown as Record<string, unknown>, p.train)
  merge(spec.samples as unknown as Record<string, unknown>, p.samples as Record<string, unknown>)
}
function onProfile(p: ModelProfile | null, picked: boolean) {
  profile.value = p
  if (p && picked) applyProfile(p)
}
// does the form still hold every value the profile sets?
const profileApplied = computed(() => {
  const p = profile.value
  if (!p) return false
  const same = (want: Record<string, unknown>, have: Record<string, unknown>): boolean =>
    Object.entries(want).every(([k, v]) =>
      v && typeof v === 'object' && !Array.isArray(v)
        ? same(v as Record<string, unknown>, (have[k] ?? {}) as Record<string, unknown>)
        : JSON.stringify(v) === JSON.stringify(have[k]),
    )
  return same(p.train, spec as unknown as Record<string, unknown>) && same(p.samples as Record<string, unknown>, spec.samples as unknown as Record<string, unknown>)
})
const profileOpen = ref(false)
const slgLayersText = computed({
  get: () => spec.samples.slg_layers.join(', '),
  set: (v: string) => (spec.samples.slg_layers = (v.match(/\d+/g) ?? []).map(Number)),
})
function useSuggestion() {
  const s = sampleSuggestion.value
  if (!s?.installed) return
  Object.assign(spec.samples, { model_name: s.installed, steps: s.steps, cfg: s.cfg, sampler: s.sampler, scheduler: s.scheduler })
}
const want: [key: string, node: string, input: string][] = [
  ['ckpt', 'CheckpointLoaderSimple', 'ckpt_name'],
  ['unet', 'UNETLoader', 'unet_name'],
  ['unet_dtype', 'UNETLoader', 'weight_dtype'],
  ['clip', 'CLIPLoader', 'clip_name'],
  ['clip_type', 'CLIPLoader', 'type'],
  ['vae', 'VAELoader', 'vae_name'],
  ['loras', 'LoraLoader', 'lora_name'],
  ['sampler', 'KSampler', 'sampler_name'],
  ['scheduler', 'KSampler', 'scheduler'],
]
const datasets = ref<DatasetInfo[]>([])
const optimizers = ref<OptimizerInfo[]>([])
const trainAttention = ref<string[]>(['pytorch'])
const runPresets = ref<Preset<Partial<TrainSpec>>[]>([])
const promptSets = ref<Preset<Partial<TrainSpec['samples']>>[]>([])

// ---- edit pairs: the chosen dataset's <name>_control pictures ---------------------------------
const NK2E_TEMPLATE =
  'Make the following edit to the reference image, changing only what is requested and keeping everything else identical: {}'
const pairCount = computed(() => datasets.value.find((d) => d.folder === spec.dataset.folder)?.pairs ?? 0)
const pairItems = ref<DatasetItem[]>([])
watch(
  () => [spec.dataset.folder, spec.dataset.edit] as const,
  async ([folder, edit]) => {
    pairItems.value = []
    if (!folder || !edit) return
    try {
      pairItems.value = (await engine.dataset(folder)).items.filter((it) => it.controls?.length)
    } catch {
      // the dataset moved: nothing to pick from
    }
  },
  { immediate: true },
)
// a dataset of pairs is an edit run unless turned off
watch(pairCount, (n, before) => {
  if (n && !before && !resuming.value) spec.dataset.edit = true
})
const controlOptions = computed(() =>
  pairItems.value.flatMap((it) => (it.controls ?? []).map((c) => ({ value: c, thumb: thumbUrl(spec.dataset.folder, c, 128) }))),
)
function samplesFromPairs() {
  spec.samples.prompts = pairItems.value.slice(0, 2).map((it) => ({ prompt: it.caption, control: it.controls![0] }))
}

const items = (vals: string[] | undefined, labels?: Record<string, string>) =>
  (vals ?? []).map((v) => ({ value: v, label: labels?.[v] ?? v, keywords: v }))

onMounted(async () => {
  await Promise.all(want.map(async ([k, n, i]) => (choices[k] = await inputChoices(n, i))))
  try {
    const [ds, health, rp, ps] = await Promise.all([
      engine.datasets(),
      engine.health(),
      engine.presets<Partial<TrainSpec>>('run'),
      engine.presets<Partial<TrainSpec['samples']>>('prompts'),
    ])
    datasets.value = ds
    optimizers.value = health.capabilities.optimizers ?? []
    trainAttention.value = health.capabilities.train_attention ?? ['pytorch']
    runPresets.value = rp
    promptSets.value = ps
  } catch {
    // engine unreachable; the submit will say so
  }
  const from = route.query.from
  if (typeof from === 'string' && from) {
    try {
      const src = (await engine.run(from)).spec
      merge(spec as unknown as Record<string, unknown>, structuredClone(src) as unknown as Record<string, unknown>)
      spec.name = nextName(src.name)
    } catch {
      // run gone; keep defaults
    }
  }
  // editing a queued run: its settings as they are, saved back over it
  const edit = route.query.edit
  if (typeof edit === 'string' && edit) {
    try {
      const src = (await engine.run(edit)).spec
      merge(spec as unknown as Record<string, unknown>, structuredClone(src) as unknown as Record<string, unknown>)
      editing.value = edit
      const r = await engine.run(edit)
      // trained already: only what its LoRA and optimizer state don't depend on can change,
      // then it resumes where it stopped
      resuming.value = r.run.status !== 'queued' && (r.checkpoints.length > 0 || r.metrics.length > 0)
      resumeStep.value = r.metrics.at(-1)?.step ?? 0
      retrying.value = r.run.status !== 'queued' && !resuming.value
    } catch {
      error.value = "That run can't be loaded to edit."
    }
  }
  // "Train more" on a checkpoint: continue from that LoRA
  if (typeof route.query.init === 'string' && route.query.init) spec.network.init_from = route.query.init
  // a new run (not a duplicate) starts from the remembered saving choices
  loraPlaces.value = await engine.loraPlaces().catch(() => [])
  if (typeof from !== 'string' || !from) {
    const folder = settings.loraFolder
    spec.output.lora_dir = folder && loraPlaces.value.some((p) => p.path === folder) ? folder : null
    spec.output.keep_states = settings.keepStates
  }
  const q = route.query.dataset
  spec.dataset.folder = (typeof q === 'string' && q) || spec.dataset.folder || datasets.value[0]?.folder || ''
})

// ---- presets: deep-merge a partial spec onto the form (so they stack) -----------------
const applied = ref<string[]>([])
function merge(target: Record<string, unknown>, src: Record<string, unknown>) {
  for (const [k, v] of Object.entries(src)) {
    if (v && typeof v === 'object' && !Array.isArray(v) && target[k] && typeof target[k] === 'object')
      merge(target[k] as Record<string, unknown>, v as Record<string, unknown>)
    else target[k] = structuredClone(toRaw(v)) // presets live in refs: clone the plain object, not the proxy
  }
}
function applyPreset(p: Preset<Partial<TrainSpec>>) {
  merge(spec as unknown as Record<string, unknown>, p.data as Record<string, unknown>)
  applied.value = [...applied.value.filter((x) => x !== p.name), p.name]
}
async function savePreset() {
  const name = prompt('Save these settings as a preset. Name:')
  if (!name) return
  const slug = name.replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase() // spec.py slugify
  const same = runPresets.value.find((p) => p.id === slug || p.name.toLowerCase() === name.trim().toLowerCase())
  if (same && !confirm(`Replace your preset “${same.name}” with these settings?`)) return
  const { model: _m, dataset, name: _n, ...rest } = structuredClone(toRaw(spec)) as TrainSpec
  const { folder: _f, ...ds } = dataset // presets are model- and dataset-agnostic
  const saved = await engine.savePreset('run', name, { ...rest, dataset: ds })
  runPresets.value = [...runPresets.value.filter((p) => p.id !== saved.preset.id), saved.preset as Preset<Partial<TrainSpec>>]
}

// ---- prompt sets ---------------------------------------------------------------------------
// Picking a set loads it into the form to edit; once edited it shows as modified and can be
// saved back over itself (a built-in gets a user copy that takes its place), saved as a new
// set, or reverted.
const SET_KEYS = ['prompts', 'negative', 'width', 'height', 'steps', 'cfg', 'sampler', 'scheduler'] as const
const promptSet = ref<string>('')
const currentSet = computed(() => promptSets.value.find((p) => p.id === promptSet.value) ?? null)
const loadSet = () => currentSet.value && merge(spec.samples as unknown as Record<string, unknown>, currentSet.value.data as Record<string, unknown>)
watch(promptSet, loadSet)
const setModified = computed(() => {
  const set = currentSet.value
  if (!set) return false
  const d = set.data as Record<string, unknown>
  const now = spec.samples as unknown as Record<string, unknown>
  return Object.keys(d).some((k) => JSON.stringify(d[k]) !== JSON.stringify(now[k]))
})
const naming = ref(false)
const newSetName = ref('')
const savingSet = ref(false)
const nameBox = ref<HTMLElement | null>(null)
async function startNaming() {
  newSetName.value = currentSet.value ? `${currentSet.value.name} (copy)` : ''
  naming.value = true
  await nextTick()
  nameBox.value?.querySelector('input')?.select()
}
async function savePromptSet(name: string) {
  if (!name.trim()) return
  savingSet.value = true
  try {
    const data = JSON.parse(JSON.stringify(Object.fromEntries(SET_KEYS.map((k) => [k, spec.samples[k]])))) // reactive arrays don't structuredClone
    const saved = await engine.savePreset('prompts', name.trim(), data)
    const row = saved.preset as Preset<Partial<TrainSpec['samples']>>
    promptSets.value = [...promptSets.value.filter((p) => p.id !== row.id), row]
    promptSet.value = row.id
    naming.value = false
  } finally {
    savingSet.value = false
  }
}

// ---- bound helpers ---------------------------------------------------------------------------
// video models get a frames field per sample (and the run's own)
const isVideoModel = computed(() => familyId.value === 'minimax-h3' || (spec.samples.frames ?? 1) > 1)
const alphaAuto = computed({
  get: () => spec.network.alpha == null,
  set: (v: boolean) => (spec.network.alpha = v ? null : spec.network.rank),
})
const alphaValue = computed({
  get: () => spec.network.alpha ?? spec.network.rank,
  set: (v: number) => (spec.network.alpha = v),
})
const initFrom = computed({
  get: () => spec.network.init_from ?? '',
  set: (v: string) => (spec.network.init_from = v || null),
})
const trainingAdapter = computed({
  get: () => spec.network.training_adapter ?? '',
  set: (v: string) => (spec.network.training_adapter = v || null),
})
const baseLora = computed({
  get: () => spec.network.base_lora ?? '',
  set: (v: string) => (spec.network.base_lora = v || null),
})
const sampleModel = computed({
  get: () => spec.samples.model_name ?? '',
  set: (v: string) => (spec.samples.model_name = v || null),
})
const targets = computed({
  get: () => (typeof spec.network.targets === 'string' ? spec.network.targets : 'attn_mlp'),
  set: (v: string) => (spec.network.targets = v as typeof spec.network.targets),
})
// one set = caption_set; more than one = caption_mix (random per step)
function toggleSet(st: string) {
  const current = spec.dataset.caption_mix.length ? [...spec.dataset.caption_mix] : [spec.dataset.caption_set]
  const next = current.includes(st) ? current.filter((x) => x !== st) : [...current, st]
  if (!next.length) return
  spec.dataset.caption_set = next[0]!
  spec.dataset.caption_mix = next.length > 1 ? next : []
}
const insertTrigger = () => {
  spec.samples.prompts = [...spec.samples.prompts, '[trigger], ']
}
const firstThumb = (d: DatasetInfo) => {
  const t = d.thumbs[0]
  return t ? thumbUrl(d.folder, [...t.subfolder.split('/').slice(1), t.filename].join('/'), 256) : ''
}

// ---- optimizer defaults: picking one sets its usual LR (and schedule), unless you'd changed it
const OPT_DEFAULTS: Record<string, { lr: number; scheduler?: TrainSpec['optim']['scheduler'] }> = {
  Automagic: { lr: 1e-6, scheduler: 'constant' }, // a start: it tunes each weight's LR itself
  // v3 moves its LR only on a consistent trend, which noisy small-batch gradients rarely show: from 1e-6
  // it barely climbs (measured on Krea 2, batch 1: 1.0e-6 -> 1.3e-6 in 280 steps). Start near a normal
  // LoRA LR and let it adjust from there
  Automagic3: { lr: 1e-4, scheduler: 'constant' },
  Prodigy: { lr: 1 },
  Lion: { lr: 3e-5 },
  Lion8bit: { lr: 3e-5 },
  SGD: { lr: 1e-3 },
  Adafactor: { lr: 1e-4 },
  AdamWScheduleFree: { lr: 2.5e-4, scheduler: 'constant' },
}
const optDefault = (id: string) => OPT_DEFAULTS[id] ?? { lr: 1e-4 }
watch(
  () => spec.optim.optimizer,
  (id, old) => {
    if (!old || id === old) return
    const was = optDefault(old)
    const next = optDefault(id)
    if (Math.abs(spec.optim.learning_rate - was.lr) < was.lr * 1e-6) spec.optim.learning_rate = next.lr
    if (next.scheduler) spec.optim.scheduler = next.scheduler
  },
)

// ---- summary & checks -----------------------------------------------------------------------
const dataset = computed(() => datasets.value.find((d) => d.folder === spec.dataset.folder))
// a mask says "only this part": picking a dataset that has some turns Masked training on
watch(
  () => spec.dataset.folder,
  (folder, was) => {
    if (was !== undefined && folder !== was) spec.dataset.masked_loss = (dataset.value?.masked ?? 0) > 0
  },
)
const VIDEO_MODES: [TrainSpec['dataset']['video_mode'], string][] = [
  ['uniform', 'Spread'],
  ['head', 'Start'],
  ['chunks', 'Back to back'],
]
const opt = computed(() => optimizers.value.find((o) => o.id === spec.optim.optimizer))
const epochs = computed(() => {
  const per = dataset.value?.steps_per_epoch
  return per ? (spec.optim.steps * spec.optim.batch_size * spec.optim.grad_accumulation_steps) / per : null
})
const modelLabel = computed(() => (spec.model.kind === 'checkpoint' ? spec.model.ckpt_name : spec.model.unet_name))
const warnings = computed(() => {
  const w: string[] = []
  if (!spec.dataset.trigger) w.push('No trigger word: the likeness has nothing specific to attach to.')
  const masked = dataset.value?.masked ?? 0
  if (masked && !spec.dataset.masked_loss)
    w.push(`${masked} picture${masked === 1 ? ' has a mask' : 's have masks'}, but Masked training is off: the parts the masks leave out train too.`)
  if (!spec.samples.prompts.length) w.push('No sample prompts: you will only see the loss curves.')
  else if (!spec.samples.every) w.push('Samples are off (Every 0): only Sample now renders them, so checkpoints get no samples or likeness. Set Every to the save interval.')
  else if (spec.save_every && spec.save_every % spec.samples.every) w.push(`Samples every ${spec.samples.every} don't land on the saves every ${spec.save_every}: checkpoints get no samples or likeness to compare.`)
  else if (spec.dataset.trigger && !spec.samples.prompts.map(promptText).some((p) => p.includes('[trigger]') || p.includes(spec.dataset.trigger)))
    w.push("Sample prompts don't use [trigger].")
  if (opt.value?.id === 'Prodigy' && spec.optim.learning_rate < 0.5) w.push('Prodigy sets its own step size: use LR 1.0.')
  if (opt.value && opt.value.id !== 'Prodigy' && spec.optim.learning_rate >= 0.01) w.push('That LR is very high for this optimizer.')
  const automagic = opt.value?.id === 'Automagic' || opt.value?.id === 'Automagic3'
  if (opt.value?.id === 'Automagic' && spec.optim.learning_rate > 1e-4) w.push("Automagic's LR is only where each weight starts: use 1e-6 and let it climb.")
  if (opt.value?.id === 'Automagic3' && spec.optim.learning_rate < 2e-5) w.push('Automagic v3 climbs only on a steady trend, which small batches rarely give: from a low start it can stay near it the whole run. Start around 1e-4.')
  if (spec.network.lora_dtype === 'bf16' && !automagic) w.push('bf16 LoRA weights lose most small steps with this optimizer: use fp32.')
  if (opt.value && !opt.value.available) w.push(`${opt.value.label} isn't installed (scripts/zt extras optimizers).`)
  if (epochs.value != null && epochs.value < 3) w.push(`Only ${epochs.value.toFixed(1)} epochs: probably too few steps.`)
  return w
})
const missing = computed(() => {
  const m = spec.model
  if (modelMissing.value) return modelMissing.value === 1 ? 'Download the missing model file' : `Download ${modelMissing.value} model files`
  if (!spec.dataset.folder) return 'Pick a dataset'
  if (m.kind === 'checkpoint' && !m.ckpt_name) return 'Pick a checkpoint'
  if (m.kind === 'split' && !(m.unet_name && m.clip_name && m.clip_type && m.vae_name)) return 'Pick all model parts'
  return null
})

function nextName(name: string): string {
  const m = name.match(/^(.*)-(\d+)$/)
  return m ? `${m[1]}-${Number(m[2]) + 1}` : `${name}-2`
}

const editing = ref<string | null>(null)
const retrying = ref(false) // a run that ended before training: saving requeues it
const resuming = ref(false) // a stopped run that has trained: saving resumes it with the new settings
const resumeStep = ref(0)
async function saveEdit() {
  if (!editing.value) return
  submitting.value = true
  error.value = null
  try {
    await engine.editRun(editing.value, spec)
    if (resuming.value) await engine.resume(editing.value)
    router.push(retrying.value || resuming.value ? `/runs/${editing.value}` : '/jobs')
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    submitting.value = false
  }
}

async function start(position: 'next' | 'end' = 'next') {
  submitting.value = true
  error.value = null
  try {
    const run = await engine.start(spec, position)
    router.push(`/runs/${run.run_id}`)
  } catch (e) {
    const comfy = e instanceof EngineError ? (e.body as { comfy?: unknown })?.comfy : null
    error.value = comfy ? `${(e as Error).message}\n${JSON.stringify(comfy, null, 2)}` : String((e as Error).message ?? e)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="page wide">
    <div class="page-head">
      <div class="titles">
        <h1>{{ editing ? (resuming ? 'Edit & resume' : retrying ? 'Edit & retry' : 'Edit queued run') : 'New run' }}</h1>
        <p v-if="editing" class="lede">
          {{ resuming ? `carries on from step ${resumeStep} with the new settings` : retrying ? 'goes back to the front of the queue' : 'keeps its place in the queue' }}
        </p>
      </div>
    </div>

    <div class="presets">
      <span class="eyebrow">Start from</span>
      <button v-for="p in runPresets" :key="p.id" class="preset" :class="{ on: applied.includes(p.name) }" :title="p.description" @click="applyPreset(p)">
        <i v-if="!p.builtin" class="mdi mdi-account-outline" /> {{ p.name }}
      </button>
      <button class="preset ghost" title="Save the current settings (not the model or dataset) as a preset" @click="savePreset">
        <i class="mdi mdi-content-save-outline" /> Save as preset
      </button>
    </div>

    <div class="layout">
      <div class="form">
        <div v-if="resuming" class="card locked-note small">
          <i class="mdi mdi-lock-outline" /> It has trained, so its model, dataset and LoRA shape (algorithm, rank, layers, blocks) and
          optimizer type stay as they were. Memory, samples, learning rate, schedule and length can change.
        </div>
        <section class="card sec" :class="{ locked: resuming }">
          <h2><span class="num">1</span> Model</h2>
          <FamilyPicker
            v-model="spec.model"
            @missing="modelMissing = $event"
            @sample-suggestion="sampleSuggestion = $event"
            @custom="customModel = $event"
            @profile="onProfile"
            @family="familyId = $event"
          />
          <div v-if="profile" class="profile" :class="{ open: profileOpen }">
            <div class="profile-head">
              <i :class="profile.status === 'verified' ? 'mdi mdi-check-decagram' : 'mdi mdi-check-circle-outline'" />
              <span class="profile-sum">
                <b>{{ profile.status === 'verified' ? 'Tuned' : 'Recipe' }}</b> · {{ profile.summary }}
              </span>
              <span class="profile-state">
                <span v-if="profileApplied" class="muted small"><i class="mdi mdi-check" /> applied</span>
                <button v-else class="chip-btn" title="Set the form back to this model's recommended values" @click="applyProfile(profile)">Apply</button>
              </span>
              <button class="link small" @click="profileOpen = !profileOpen">{{ profileOpen ? 'Less' : 'Details' }}</button>
            </div>
            <div v-if="profileOpen" class="profile-body">
              <ul v-if="profile.notes.length" class="profile-notes">
                <li v-for="n in profile.notes" :key="n">{{ n }}</li>
              </ul>
              <div class="profile-src">
                <a v-for="src in profile.sources" :key="src.url" :href="src.url" target="_blank" rel="noopener" class="link small"><i class="mdi mdi-open-in-new" /> {{ src.label }}</a>
              </div>
            </div>
          </div>
          <div v-if="customModel" class="custom-model">
            <div class="seg">
              <button :class="{ on: spec.model.kind === 'checkpoint' }" @click="spec.model.kind = 'checkpoint'">One checkpoint file</button>
              <button :class="{ on: spec.model.kind === 'split' }" @click="spec.model.kind = 'split'">Diffusion model + text encoder + VAE</button>
            </div>
            <div class="grid-2">
              <ZenField v-if="spec.model.kind === 'checkpoint'" label="Checkpoint" stack class="wide-f">
                <ZenCombo v-model="spec.model.ckpt_name as string" :items="items(choices.ckpt)" placeholder="Search checkpoints…" :item-height="30" />
              </ZenField>
              <template v-else>
                <ZenField label="Diffusion model" stack class="wide-f">
                  <ZenCombo v-model="spec.model.unet_name as string" :items="items(choices.unet)" placeholder="Search diffusion models…" :item-height="30" />
                </ZenField>
                <ZenField label="Text encoder" stack>
                  <ZenCombo v-model="spec.model.clip_name as string" :items="items(choices.clip)" placeholder="Search text encoders…" :item-height="30" />
                </ZenField>
                <ZenField label="Text encoder type" stack tip="The model family">
                  <ZenCombo v-model="spec.model.clip_type as string" :items="items(choices.clip_type)" placeholder="Family…" />
                </ZenField>
                <ZenField label="VAE" stack>
                  <ZenCombo v-model="spec.model.vae_name as string" :items="items(choices.vae)" placeholder="Search VAEs…" :item-height="30" />
                </ZenField>
                <ZenField label="Weight dtype" stack tip="fp8 + quantized backward is the low-VRAM path">
                  <ZenCombo v-model="spec.model.unet_weight_dtype as string" :items="items(choices.unet_dtype ?? ['default'])" :searchable="false" />
                </ZenField>
              </template>
            </div>
          </div>
        </section>

        <section class="card sec" :class="{ locked: resuming }">
          <h2><span class="num">2</span> Dataset</h2>
          <ZenField label="Folder" stack>
            <ZenCombo
              v-model="spec.dataset.folder"
              :items="datasets.map((d) => ({ value: d.folder, label: d.folder, keywords: d.folder, d }))"
              placeholder="Pick a dataset…"
              grid
              :grid-min="150"
              :menu-width="560"
            >
              <template #option="{ item }">
                <div class="ds-opt">
                  <img v-if="firstThumb(item.d as DatasetInfo)" :src="firstThumb(item.d as DatasetInfo)" alt="" />
                  <b>{{ item.label }}</b>
                  <span class="muted">{{ (item.d as DatasetInfo).images }} images</span>
                </div>
              </template>
            </ZenCombo>
          </ZenField>
          <div class="grid-2">
            <ZenField label="Captions" stack :hint="spec.dataset.caption_mix.length > 1 ? 'Each step picks one of these sets at random' : 'Which caption set to train on; click more to mix them'">
              <div class="chips">
                <button
                  v-for="st in dataset?.sets ?? ['default']"
                  :key="st"
                  class="chip-btn"
                  :class="{ on: spec.dataset.caption_mix.length ? spec.dataset.caption_mix.includes(st) : spec.dataset.caption_set === st }"
                  @click="toggleSet(st)"
                >
                  {{ st }}
                </button>
              </div>
            </ZenField>
            <ZenField label="Trigger word" stack tip="Replaces [trigger] in captions and prompts">
              <ZenInput v-model="spec.dataset.trigger" placeholder="e.g. ohwx" />
            </ZenField>
            <ZenField label="Add it to every caption" stack tip="Prepend it where a caption doesn't mention it"><div class="sw-row"><ZenSwitch v-model="spec.dataset.trigger_prepend" :disabled="!spec.dataset.trigger" /><span class="sw-state">{{ spec.dataset.trigger_prepend ? 'On' : 'Off' }}</span></div></ZenField>
            <ZenField label="Resolution" stack>
              <div class="chips">
                <button v-for="r in [512, 768, 1024, 1280]" :key="r" class="chip-btn" :class="{ on: spec.dataset.resolution === r }" @click="spec.dataset.resolution = r">
                  {{ r }}
                </button>
              </div>
            </ZenField>
            <ZenField label="Caption dropout" stack tip="Share of steps with an empty caption">
              <ZenNumber v-model="spec.dataset.caption_dropout" :min="0" :max="1" :step="0.05" :precision="2" />
            </ZenField>
            <ZenField label="Edit training" stack :tip="`Each image's <name>_control picture is the input the model edits from; the caption is the instruction. For models that take reference images (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit)${pairCount ? `. This dataset has ${pairCount} pairs` : ''}`">
              <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.edit" @update:model-value="spec.dataset.edit = $event" /><span class="sw-state">{{ spec.dataset.edit ? `On · ${pairCount} pairs` : 'Off' }}</span></div>
            </ZenField>
            <template v-if="spec.dataset.edit">
              <ZenField label="Instruction template" stack tip="Wraps every instruction ({} is the caption), in training and samples alike. Training on top of NK2E? Use its template: its LoRA learned wrapped instructions">
                <div class="tpl-row">
                  <ZenInput :model-value="spec.dataset.edit_template ?? ''" placeholder="None: the caption as it is" size="sm" @update:model-value="spec.dataset.edit_template = String($event)" />
                  <button class="chip-btn" :class="{ on: spec.dataset.edit_template === NK2E_TEMPLATE }" title="NK2E's instruction template" @click="spec.dataset.edit_template = spec.dataset.edit_template === NK2E_TEMPLATE ? '' : NK2E_TEMPLATE">NK2E</button>
                </div>
              </ZenField>
              <ZenField label="Encoder sees the input" stack tip="The text encoder looks at the input picture with the instruction (Qwen3-VL encoders: Krea 2, Qwen Image), as krea2edit's identity edits do: better at keeping who and what is in the picture. Each pair's text is encoded on its own, so caching takes longer">
                <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.edit_grounded" @update:model-value="spec.dataset.edit_grounded = $event" /><span class="sw-state">{{ spec.dataset.edit_grounded ? 'On' : 'Off' }}</span></div>
              </ZenField>
            </template>
            <ZenField label="Masked training" stack tip="Masked images learn only where it's white"><div class="sw-row"><ZenSwitch v-model="spec.dataset.masked_loss" /><span class="sw-state">{{ spec.dataset.masked_loss ? 'On' : 'Off' }}</span></div></ZenField>
            <ZenField v-if="spec.dataset.masked_loss" label="Unmasked weight" stack tip="0 ignores the rest; 0.1 learns it lightly">
              <ZenNumber v-model="spec.dataset.unmasked_weight" :min="0" :max="1" :step="0.05" :precision="2" />
            </ZenField>
          </div>
          <!-- only for datasets with videos: how they turn into training clips -->
          <div v-if="dataset?.videos" class="video-opts">
            <div class="sub-head"><i class="mdi mdi-filmstrip" /> Videos <span class="muted small">{{ dataset.videos }} in this dataset</span></div>
            <div class="grid-2">
              <ZenField label="Frames per clip" stack tip="The longest clip; it's cut to a length the model's VAE encodes exactly (49 → 39 for MiniMax H3, 49 for Wan)">
                <ZenNumber v-model="spec.dataset.video_frames" :min="1" :max="257" :step="8" :precision="0" />
              </ZenField>
              <ZenField label="Frame rate" stack tip="Resample clips to this many frames a second; 0 keeps each video's own">
                <ZenNumber v-model="spec.dataset.video_fps" :min="0" :max="60" :step="1" :precision="0" />
              </ZenField>
              <ZenField label="Which part" stack tip="Spread: clips spaced over the whole video. Start: the opening only. Back to back: consecutive clips from the start">
                <div class="chips">
                  <button v-for="[m, l] in VIDEO_MODES" :key="m" class="chip-btn" :class="{ on: spec.dataset.video_mode === m }" @click="spec.dataset.video_mode = m">{{ l }}</button>
                </div>
              </ZenField>
              <ZenField label="Clips per video" stack tip="Each clip trains as its own sample">
                <ZenNumber v-model="spec.dataset.video_clips" :min="1" :max="32" :step="1" :precision="0" :disabled="spec.dataset.video_mode === 'head'" />
              </ZenField>
              <ZenField label="Video resolution" stack tip="A clip costs frames × pixels, so videos train smaller than images">
                <div class="chips">
                  <button v-for="r in [256, 384, 512, 640, 768]" :key="r" class="chip-btn" :class="{ on: spec.dataset.video_resolution === r }" @click="spec.dataset.video_resolution = r">{{ r }}</button>
                </div>
              </ZenField>
              <ZenField label="Stay inside a shot" stack tip="Finds hard cuts and keeps every clip within one shot. Leave off for edits where the cutting is the point (music videos)">
                <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.video_avoid_cuts" @update:model-value="spec.dataset.video_avoid_cuts = $event" /><span class="sw-state">{{ spec.dataset.video_avoid_cuts ? 'Split at cuts' : 'Clips may cross cuts' }}</span></div>
              </ZenField>
              <ZenField label="Image to video" stack tip="MiniMax H3 fl2va: each clip is conditioned on its own first frame (or first and last), as the model is at inference when you give it a start image. Trains the LoRA for image-to-video; samples use their input picture as the first frame">
                <ZenCombo
                  :model-value="spec.dataset.keyframes ?? ''"
                  :items="items(['', 'first', 'first_last'], { '': 'Off: text to video', first: 'First frame', first_last: 'First and last frame' })"
                  :searchable="false"
                  :sorted="false"
                  @update:model-value="spec.dataset.keyframes = String($event)"
                />
              </ZenField>
              <ZenField label="Clip stills" stack tip="Each clip also trains its sharpest frame as a photo, with the clip's caption. Stills run whole (never TREAD-routed) and anchor the LoRA's detail; Fizgig turns this on for H3">
                <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.clip_stills" @update:model-value="spec.dataset.clip_stills = $event" /><span class="sw-state">{{ spec.dataset.clip_stills ? 'One per clip' : 'Off' }}</span></div>
              </ZenField>
              <ZenField label="Train on sound" stack tip="Joint audio-video models (MiniMax H3): each clip's soundtrack joins the loss, so motion and cuts can learn the beat. Needs the model's audio VAE">
                <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.video_audio" @update:model-value="spec.dataset.video_audio = $event" /><span class="sw-state">{{ spec.dataset.video_audio ? 'On' : 'Off' }}</span></div>
              </ZenField>
              <ZenField v-if="spec.dataset.video_audio" label="Audio VAE" stack tip="The model's audio VAE (MiniMax H3: minimax_h3_audio_vae)">
                <ZenCombo :model-value="spec.model.audio_vae_name ?? ''" :items="items(choices.vae ?? [])" placeholder="Pick the audio VAE…" :item-height="30" @update:model-value="spec.model.audio_vae_name = String($event) || null" />
              </ZenField>
            </div>
          </div>
        </section>

        <section class="card sec">
          <h2><span class="num">3</span> Training</h2>
          <div class="grid-2">
            <ZenField label="Steps" stack :hint="epochs != null ? `≈ ${epochs.toFixed(1)} epochs of ${dataset?.steps_per_epoch} images` : ''">
              <ZenNumber v-model="spec.optim.steps" :min="1" :step="100" :precision="0" />
            </ZenField>
            <ZenField label="Optimizer" stack :tip="opt ? `${opt.note} Suggested LR ${opt.lr_hint}.`.trim() : ''">
              <ZenCombo
                v-model="spec.optim.optimizer"
                :items="optimizers.map((o) => ({ value: o.id, label: o.label, keywords: o.note, o }))"
                :sorted="false"
                :pick-guard="(v: string | number) => !optimizers.find((o) => o.id === v)?.available"
                :menu-width="360"
                :item-height="34"
              >
                <!-- what each one is for, while choosing: an ⓘ per option (clicking it doesn't pick) -->
                <template #option="{ item }">
                  <div class="opt-row" :class="{ off: !(item.o as OptimizerInfo).available }">
                    <span class="opt-name">{{ item.label }}</span>
                    <span v-if="(item.o as OptimizerInfo).note" class="opt-tip" @click.stop @mousedown.stop @pointerdown.stop>
                      <ZenTip :text="(item.o as OptimizerInfo).note" :label="`About ${item.label}`" />
                    </span>
                    <span class="spacer" />
                    <span class="muted mono small">{{ (item.o as OptimizerInfo).available ? `lr ${(item.o as OptimizerInfo).lr_hint}` : 'not installed' }}</span>
                  </div>
                </template>
              </ZenCombo>
            </ZenField>
            <ZenField label="Learning rate" stack>
              <ZenNumber v-model="spec.optim.learning_rate" :min="0" :step="0.00001" :precision="6" />
            </ZenField>
            <ZenField label="Rank" stack tip="8–16 for a character, 16–32 for a style">
              <ZenNumber v-model="spec.network.rank" :min="1" :max="256" :step="1" :precision="0" />
            </ZenField>
            <ZenField label="Schedule" stack>
              <ZenCombo v-model="spec.optim.scheduler" :items="items(['constant', 'linear', 'cosine'])" :searchable="false" :sorted="false" />
            </ZenField>
            <ZenField label="Save every" stack tip="0 = only the final LoRA">
              <ZenNumber v-model="spec.save_every" :min="0" :step="50" :precision="0" />
            </ZenField>
            <ZenField label="Save LoRAs to" stack class="span-2" :tip="loraPlaceHint">
              <ZenCombo v-model="loraDir" :items="loraPlaceItems" :sorted="false" :searchable="false" :item-height="40" :menu-width="520">
                <template #option="{ item }">
                  <div class="place-opt">
                    <span class="mono ellipsis">{{ item.label }}</span>
                    <span class="spacer" />
                    <span class="muted small mono">{{ item.free }}</span>
                  </div>
                </template>
              </ZenCombo>
            </ZenField>
            <ZenField label="Resume states" stack tip="Optimizer state for resuming a stopped run; often bigger than the LoRA">
              <ZenCombo
                v-model="spec.output.keep_states"
                :items="[{ value: 'latest', label: 'Latest only' }, { value: 'all', label: 'Every checkpoint' }, { value: 'none', label: 'None (no resuming)' }]"
                :sorted="false"
                :searchable="false"
              />
            </ZenField>
          </div>
          <button class="more" @click="advanced.training = !advanced.training">
            <i :class="advanced.training ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Advanced training
          </button>
          <div v-if="advanced.training" class="adv">
            <div class="subgroup">
              <div class="sub-title">Optimizer</div>
              <div class="grid-2">
                <ZenField label="Warmup steps" stack><ZenNumber v-model="spec.optim.warmup_steps" :min="0" :step="10" :precision="0" /></ZenField>
                <ZenField label="Batch size" stack><ZenNumber v-model="spec.optim.batch_size" :min="1" :step="1" :precision="0" /></ZenField>
                <ZenField label="Grad accumulation" stack><ZenNumber v-model="spec.optim.grad_accumulation_steps" :min="1" :step="1" :precision="0" /></ZenField>
                <ZenField label="LoRA+" stack tip="The LoRA's up matrices learn this many times faster than the down ones: trains faster at the same stability. 4 is a safe start, the paper uses 16. Off for LoHa/LoKr/OFT">
                  <div class="sw-row">
                    <ZenSwitch :model-value="(spec.optim.lora_plus ?? 0) > 1" @update:model-value="spec.optim.lora_plus = $event ? 4 : 0" />
                    <ZenNumber v-if="(spec.optim.lora_plus ?? 0) > 1" :model-value="spec.optim.lora_plus ?? 4" :min="1.5" :max="32" :step="1" :precision="1" @update:model-value="spec.optim.lora_plus = $event" />
                    <span v-else class="sw-state">Off</span>
                  </div>
                </ZenField>
                <ZenField label="Weight averaging" stack tip="Checkpoints, samples and the saved LoRA use a running average of the recent steps (EMA) instead of whichever step the save landed on: steadier likeness, less wobble between checkpoints. 0.98 ≈ the last 50 steps">
                  <div class="sw-row">
                    <ZenSwitch :model-value="(spec.optim.ema ?? 0) > 0" @update:model-value="spec.optim.ema = $event ? 0.98 : 0" />
                    <ZenNumber v-if="(spec.optim.ema ?? 0) > 0" :model-value="spec.optim.ema ?? 0.98" :min="0.5" :max="0.999" :step="0.005" :precision="3" @update:model-value="spec.optim.ema = $event" />
                    <span v-else class="sw-state">Off</span>
                  </div>
                </ZenField>
                <ZenField label="Weight decay" stack><ZenNumber v-model="spec.optim.weight_decay" :min="0" :step="0.001" :precision="4" /></ZenField>
                <ZenField label="Grad clip" stack tip="Max grad norm; 0 = off"><ZenNumber v-model="spec.optim.max_grad_norm" :min="0" :step="0.1" :precision="2" /></ZenField>
              </div>
            </div>
            <div class="subgroup" :class="{ locked: resuming }">
              <div class="sub-title">Network</div>
              <div class="grid-2">
                <ZenField label="Algorithm" stack><ZenCombo v-model="spec.network.algorithm" :items="items(['LoRA', 'LoHa', 'LoKr', 'OFT'])" :searchable="false" :sorted="false" /></ZenField>
                <ZenField label="Layers" stack>
                  <ZenCombo v-model="targets" :items="items(['attn', 'attn_mlp', 'all'], { attn: 'Attention', attn_mlp: 'Attention + MLP', all: 'Every layer' })" :searchable="false" :sorted="false" />
                </ZenField>
                <ZenField label="Alpha" stack :tip="alphaAuto ? 'Auto = rank (scale 1.0)' : 'Scale is alpha / rank'">
                  <div class="sw-row">
                    <ZenSwitch v-model="alphaAuto" />
                    <ZenNumber v-if="!alphaAuto" v-model="alphaValue" :min="0.01" :step="1" :precision="2" />
                    <span v-else class="sw-state">Auto (= rank)</span>
                  </div>
                </ZenField>
                <ZenField label="LoRA precision" stack tip="fp32: small steps land. bf16 rounds most AdamW steps away; use it only with Automagic (stochastic rounding)">
                  <ZenCombo v-model="spec.network.lora_dtype" :items="items(['fp32', 'bf16'], { fp32: 'fp32 (recommended)', bf16: 'bf16' })" :searchable="false" :sorted="false" />
                </ZenField>
                <ZenField label="Continue from" stack>
                  <ZenCombo v-model="initFrom" :items="[{ value: '', label: 'New LoRA' }, ...items(choices.loras)]" :item-height="30" />
                </ZenField>
                <ZenField label="Training adapter" stack class="span-2" tip="A LoRA applied while training only, never in samples or the saved file. Guidance-distilled models (MiniMax H3) train much better on a de-distillation adapter">
                  <AdapterPicker v-model="trainingAdapter" :family="familyId" />
                </ZenField>
                <ZenField label="Base LoRA" stack class="span-2" tip="A LoRA the new one sits on everywhere: training, samples and use. For a fix the model always needs (MOSS SFT's destripe) or an edit LoRA to build on. The new LoRA learns only what's left, so load the base beside it when you use it">
                  <AdapterPicker v-model="baseLora" :family="familyId" kind="base" />
                </ZenField>
                <ZenField v-if="spec.network.base_lora" label="Base strength" stack tip="The strength the base LoRA runs at, in training and samples. Use it at the same strength">
                  <ZenNumber :model-value="spec.network.base_strength ?? 1" :min="-4" :max="4" :step="0.05" :precision="2" @update:model-value="spec.network.base_strength = $event" />
                </ZenField>
                <ZenField label="Blocks" stack tip="Only these transformer blocks get adapters, e.g. 20-49: the backward stops at the first one, so steps are quicker, and the early blocks (layout, motion) stay untouched. Empty = every block">
                  <ZenInput :model-value="spec.network.blocks ?? ''" placeholder="all" @update:model-value="spec.network.blocks = String($event).trim()" />
                </ZenField>
                <ZenField label="Train biases & norms" stack><div class="sw-row"><ZenSwitch v-model="spec.network.train_bias_norm" /><span class="sw-state">{{ spec.network.train_bias_norm ? 'On' : 'Off' }}</span></div></ZenField>
              </div>
            </div>
            <div class="subgroup">
              <div class="sub-title">Objective</div>
              <div class="grid-2">
                <ZenField label="Target" stack tip="Native: ε for SD, velocity for flow models">
                  <ZenCombo v-model="spec.objective.target" :items="items(['native', 'x0'], { native: 'Model native', x0: 'Denoised (x0)' })" :searchable="false" :sorted="false" />
                </ZenField>
                <ZenField label="Timesteps" stack tip="Logit-normal favours mid noise (flow models)">
                  <ZenCombo v-model="spec.objective.timesteps" :items="items(['uniform', 'logit_normal'], { logit_normal: 'Logit-normal' })" :searchable="false" :sorted="false" />
                </ZenField>
                <ZenField v-if="spec.objective.timesteps === 'logit_normal'" label="Centre / spread" stack tip="Logit-normal's mean and standard deviation. Mean 0 centres on mid noise; positive leans to heavy noise (composition), negative to light noise (detail). MOSS pretrained at 1.09 / 0.95">
                  <div class="inline">
                    <ZenNumber v-model="spec.objective.logit_mean" :min="-3" :max="3" :step="0.1" :precision="2" />
                    <ZenNumber v-model="spec.objective.logit_std" :min="0.1" :max="3" :step="0.1" :precision="2" />
                  </div>
                </ZenField>
                <ZenField v-if="spec.objective.timesteps === 'logit_normal'" label="Uniform mix (%)" stack tip="Share of steps drawn uniformly instead: a little training at the extremes logit-normal rarely visits. 1% is plenty">
                  <ZenNumber :model-value="Math.round((spec.objective.uniform_mix ?? 0) * 100)" :min="0" :max="50" :step="1" :precision="0" @update:model-value="spec.objective.uniform_mix = $event / 100" />
                </ZenField>
                <ZenField label="Timestep shift" stack tip="Flow models. 0 = the model's own (MOSS 5: composition, colour). 1–3 trains more on fine detail like faces">
                  <ZenNumber v-model="spec.objective.shift" :min="0" :max="10" :step="0.5" :precision="1" />
                </ZenField>
                <ZenField label="Loss" stack><ZenCombo v-model="spec.objective.loss" :items="items(['mse', 'l1', 'huber', 'smooth_l1'])" :searchable="false" :sorted="false" /></ZenField>
                <ZenField label="Validation every" stack tip="Fixed-noise loss for a smooth curve; 0 = off">
                  <ZenNumber v-model="spec.validation.every" :min="0" :step="50" :precision="0" />
                </ZenField>
              </div>
            </div>
          </div>
        </section>

        <section class="card sec">
          <h2><span class="num">4</span> Samples</h2>
          <!-- one fixed-height row: the set picker and its actions, or the name field when saving -->
          <div v-if="naming" class="sample-head" @keydown.enter.prevent="savePromptSet(newSetName)" @keydown.esc="naming = false">
            <div ref="nameBox"><ZenInput v-model="newSetName" size="sm" placeholder="Name this prompt set" /></div>
            <ZenButton size="sm" variant="primary" icon="mdi mdi-content-save-outline" :disabled="!newSetName.trim() || savingSet" @click="savePromptSet(newSetName)">Save</ZenButton>
            <ZenButton size="sm" @click="naming = false">Cancel</ZenButton>
          </div>
          <div v-else class="sample-head">
            <div class="set-pick">
              <ZenCombo
                v-model="promptSet"
                :items="promptSets.map((p) => ({ value: p.id, label: p.name }))"
                placeholder="Saved prompt sets…"
                empty-text="No saved sets yet"
                size="sm"
              />
              <span v-if="setModified" class="set-dot" title="Edited since it was loaded" />
            </div>
            <template v-if="currentSet && setModified">
              <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="savingSet" :title="currentSet.builtin ? 'Save your edits as your own version of this built-in set' : `Overwrite “${currentSet.name}”`" @click="savePromptSet(currentSet.name)">Save</ZenButton>
              <ZenButton size="sm" icon="mdi mdi-content-save-plus-outline" @click="startNaming">Save as new</ZenButton>
              <ZenButton size="sm" icon="mdi mdi-undo-variant" title="Back to the saved prompts" @click="loadSet">Revert</ZenButton>
            </template>
            <ZenButton v-else-if="!currentSet" size="sm" icon="mdi mdi-content-save-outline" :disabled="!spec.samples.prompts.length" @click="startNaming">Save as set</ZenButton>
            <span class="spacer" />
            <ZenButton v-if="spec.dataset.edit && pairItems.length" size="sm" icon="mdi mdi-swap-horizontal" title="Sample two of the dataset's pairs: their instruction on their input picture" @click="samplesFromPairs">From pairs</ZenButton>
            <ZenButton size="sm" icon="mdi mdi-tag-plus-outline" @click="insertTrigger">Insert [trigger]</ZenButton>
          </div>
          <SamplePromptList
            v-model="spec.samples.prompts"
            :width="spec.samples.width"
            :height="spec.samples.height"
            :frames="spec.samples.frames ?? 1"
            :video="isVideoModel"
            :trigger="spec.dataset.trigger"
            :controls="spec.dataset.edit ? controlOptions : undefined"
          />
          <div v-if="sampleSuggestion && !usingSuggestion" class="suggest">
            <i class="mdi mdi-lightning-bolt-outline" />
            <span>
              Preview faster on <b>{{ sampleSuggestion.model }}</b>: {{ sampleSuggestion.steps }} steps, CFG {{ sampleSuggestion.cfg }},
              {{ sampleSuggestion.sampler }} / {{ sampleSuggestion.scheduler }}, the settings ComfyUI's template uses with it.
            </span>
            <button v-if="sampleSuggestion.installed" class="chip-btn" @click="useSuggestion">Use</button>
            <template v-else-if="['queued', 'downloading'].includes(downloadFor(sampleSuggestion.folder, sampleSuggestion.model)?.status ?? '')">
              <span class="mono small">{{ Math.floor(((downloadFor(sampleSuggestion.folder, sampleSuggestion.model)!.done) / (downloadFor(sampleSuggestion.folder, sampleSuggestion.model)!.total || 1)) * 100) }}%</span>
            </template>
            <button v-else class="chip-btn" @click="engine.download(sampleSuggestion.family, sampleSuggestion.folder, sampleSuggestion.model)">
              <i class="mdi mdi-download" /> {{ fmtBytes(sampleSuggestion.size) || 'Download' }}
            </button>
          </div>
          <div class="grid-2">
            <ZenField label="Every" stack tip="Steps between samples; 0 = off (only Sample now). The save interval puts samples, and a likeness score, on every checkpoint">
              <ZenNumber v-model="spec.samples.every" :min="0" :step="50" :precision="0" />
            </ZenField>
            <ZenField label="Baseline before training" stack tip="Render once with the untouched model"><div class="sw-row"><ZenSwitch v-model="spec.samples.baseline" /><span class="sw-state">{{ spec.samples.baseline ? 'On' : 'Off' }}</span></div></ZenField>
            <ZenField label="Sample with another model" stack tip="E.g. train on raw, preview on turbo (same family)">
              <ZenCombo
                v-model="sampleModel"
                :items="[{ value: '', label: 'The model being trained' }, ...items(spec.model.kind === 'split' ? choices.unet : choices.ckpt)]"
                :item-height="30"
              />
            </ZenField>
            <ZenField v-if="spec.samples.model_name" label="LoRA strength" stack>
              <ZenNumber v-model="spec.samples.lora_strength" :min="0" :max="2" :step="0.05" :precision="2" />
            </ZenField>
          </div>
          <button class="more" @click="advanced.samples = !advanced.samples">
            <i :class="advanced.samples ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Sampler settings
          </button>
          <div v-if="advanced.samples" class="grid-2">
            <ZenField label="Size" stack>
              <div class="inline">
                <ZenNumber v-model="spec.samples.width" :min="256" :step="64" :precision="0" />
                <span class="muted">×</span>
                <ZenNumber v-model="spec.samples.height" :min="256" :step="64" :precision="0" />
              </div>
            </ZenField>
            <ZenField v-if="isVideoModel" label="Frames / fps" stack tip="Video samples: frames per sample (snapped to a length the VAE decodes; 1 = a still) and the frame rate they're saved at">
              <div class="inline">
                <ZenNumber :model-value="spec.samples.frames ?? 1" :min="1" :max="361" :step="17" :precision="0" @update:model-value="spec.samples.frames = $event" />
                <ZenNumber :model-value="spec.samples.fps ?? 24" :min="1" :max="60" :step="1" :precision="0" @update:model-value="spec.samples.fps = $event" />
              </div>
            </ZenField>
            <ZenField label="Turbo LoRA" stack tip="Rides under every sample (never training or the saved LoRA) so previews take a few steps: MiniMax H3's minimax_h3_turbo_v4_step600_ema at 0.75, 6 steps, CFG 1">
              <div class="inline">
                <ZenCombo :model-value="spec.samples.turbo_lora ?? ''" :items="[{ value: '', label: 'None' }, ...items(choices.loras)]" :item-height="30" @update:model-value="spec.samples.turbo_lora = String($event) || null" />
                <ZenNumber v-if="spec.samples.turbo_lora" :model-value="spec.samples.turbo_strength ?? 0.75" :min="0" :max="1.5" :step="0.05" :precision="2" @update:model-value="spec.samples.turbo_strength = $event" />
              </div>
            </ZenField>
            <ZenField label="Steps / CFG" stack>
              <div class="inline">
                <ZenNumber v-model="spec.samples.steps" :min="1" :step="1" :precision="0" />
                <ZenNumber v-model="spec.samples.cfg" :min="0" :step="0.5" :precision="1" />
              </div>
            </ZenField>
            <ZenField label="Sampler" stack><ZenCombo v-model="spec.samples.sampler" :items="items(choices.sampler ?? ['euler'])" /></ZenField>
            <ZenField label="Scheduler" stack><ZenCombo v-model="spec.samples.scheduler" :items="items(choices.scheduler ?? ['normal'])" /></ZenField>
            <ZenField label="Seed" stack><ZenNumber v-model="spec.samples.seed" :min="0" :step="1" :precision="0" /></ZenField>
            <ZenField label="Skip-layer guidance" stack tip="Negative pass: skip layers in the unconditional pass (free). Extra pass: one more forward per step">
              <div class="seg">
                <button v-for="m in (['off', 'uncond', 'standard'] as const)" :key="m" :class="{ on: spec.samples.slg_mode === m }" @click="spec.samples.slg_mode = m">
                  {{ { off: 'Off', uncond: 'Negative pass', standard: 'Extra pass' }[m] }}
                </button>
              </div>
            </ZenField>
            <ZenField v-if="spec.samples.slg_mode !== 'off'" label="Skipped layers" stack tip="Block indices, from 0">
              <ZenInput v-model="slgLayersText" placeholder="22" />
            </ZenField>
            <template v-if="spec.samples.slg_mode === 'standard'">
              <ZenField label="SLG scale" stack><ZenNumber v-model="spec.samples.slg_scale" :min="0" :step="0.5" :precision="1" /></ZenField>
              <ZenField label="SLG range" stack tip="Fraction of the steps it runs over">
                <div class="inline">
                  <ZenNumber v-model="spec.samples.slg_start" :min="0" :max="1" :step="0.01" :precision="2" />
                  <ZenNumber v-model="spec.samples.slg_end" :min="0" :max="1" :step="0.01" :precision="2" />
                </div>
              </ZenField>
            </template>
            <ZenField label="Negative" stack class="wide-f"><ZenInput v-model="spec.samples.negative" /></ZenField>
          </div>
        </section>

        <section class="card sec">
          <button class="more head" @click="advanced.memory = !advanced.memory">
            <i :class="advanced.memory ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Memory & speed
            <span class="muted small">{{ spec.memory.training_dtype }} · checkpointing {{ spec.memory.gradient_checkpointing ? 'on' : 'off' }}{{ spec.memory.offloading ? ' · offload' : '' }}</span>
          </button>
          <div v-if="advanced.memory" class="grid-2">
            <ZenField label="Training dtype" stack><ZenCombo v-model="spec.memory.training_dtype" :items="items(['bf16', 'fp16', 'fp32'])" :searchable="false" :sorted="false" /></ZenField>
            <ZenField label="Attention" stack tip="Only backends with a backward pass"><ZenCombo v-model="spec.memory.attention" :items="items(trainAttention)" :searchable="false" /></ZenField>
            <ZenField label="Gradient checkpointing" stack tip="Recompute each block in backward"><div class="sw-row"><ZenSwitch v-model="spec.memory.gradient_checkpointing" /><span class="sw-state">{{ spec.memory.gradient_checkpointing ? 'On' : 'Off' }}</span></div></ZenField>
            <ZenField v-if="spec.memory.gradient_checkpointing" label="Keep last blocks" stack tip="checkpoint_skip: the last N blocks skip checkpointing: they keep their activations (more VRAM) and skip the recompute (faster). Spare VRAM spent on speed">
              <ZenNumber v-model="spec.memory.checkpoint_skip" :min="0" :max="64" :step="1" :precision="0" />
            </ZenField>
            <ZenField label="Offload weights" stack tip="Stream weights from RAM"><div class="sw-row"><ZenSwitch v-model="spec.memory.offloading" /><span class="sw-state">{{ spec.memory.offloading ? 'On' : 'Off' }}</span></div></ZenField>
            <ZenField label="Quantized backward" stack tip="FP8 matmuls in backward (fp8 bases)"><div class="sw-row"><ZenSwitch v-model="spec.memory.quantized_backward" /><span class="sw-state">{{ spec.memory.quantized_backward ? 'On' : 'Off' }}</span></div></ZenField>
            <ZenField label="TREAD (video)" stack tip="On video steps this share of the clip's tokens skips the middle blocks and rejoins unchanged: those blocks do about half the work at 50%. Stills, samples and validation run whole. Best with some stills in the dataset: they anchor the LoRA. MiniMax H3">
              <div class="sw-row">
                <ZenSwitch :model-value="(spec.memory.tread ?? 0) > 0" @update:model-value="spec.memory.tread = $event ? 0.5 : 0" />
                <ZenNumber v-if="(spec.memory.tread ?? 0) > 0" :model-value="Math.round((spec.memory.tread ?? 0.5) * 100)" :min="10" :max="90" :step="5" :precision="0" @update:model-value="spec.memory.tread = $event / 100" />
                <span v-else class="sw-state">Off</span>
              </div>
            </ZenField>
            <ZenField v-if="(spec.memory.tread ?? 0) > 0" label="Share of steps routed" stack tip="TREAD on only this share of clip steps; the rest run whole. Routing every step of a dataset with no stills lets the LoRA learn to make up for the routing (measured on H3: over-sharpened samples). Whole steps keep it working on the model it'll actually be used on">
              <ZenNumber :model-value="Math.round((spec.memory.tread_steps ?? 1) * 100)" :min="0" :max="100" :step="10" :precision="0" @update:model-value="spec.memory.tread_steps = $event / 100" />
            </ZenField>
            <ZenField label="Compile blocks" stack tip="torch.compile on the model's repeated transformer block: the small operations around the big matmuls (norms, modulation, activations, the LoRA add) fuse into fewer kernels. The first steps and the first sample are slower while it compiles">
              <div class="sw-row"><ZenSwitch :model-value="!!spec.memory.compile_blocks" @update:model-value="spec.memory.compile_blocks = $event" /><span class="sw-state">{{ spec.memory.compile_blocks ? 'On' : 'Off' }}</span></div>
            </ZenField>
            <ZenField label="int8 forward" stack tip="int8 ConvRot bases (MiniMax H3, Krea 2 int8): the frozen layers run ComfyUI's int8 kernel forward, activations quantized, instead of rebuilding a bf16 weight for each matmul. Off is slower: for comparing quality">
              <div class="sw-row"><ZenSwitch :model-value="spec.memory.int8_forward !== false" @update:model-value="spec.memory.int8_forward = $event" /><span class="sw-state">{{ spec.memory.int8_forward !== false ? 'On' : 'Off' }}</span></div>
            </ZenField>
            <ZenField label="int8 backward (W8A8)" stack tip="int8 ConvRot bases (MiniMax H3, Krea 2 int8): the gradient is quantized to int8 too, so each frozen layer's input gradient runs on int8 tensor cores (about 2× the bf16 backward on an RTX 5090). Rounded stochastically so it stays unbiased; noisier per step than 16-bit">
              <div class="sw-row"><ZenSwitch :model-value="!!spec.memory.w8a8_backward" @update:model-value="spec.memory.w8a8_backward = $event" /><span class="sw-state">{{ spec.memory.w8a8_backward ? 'On' : 'Off' }}</span></div>
            </ZenField>
            <ZenField label="Fused int8 backward" stack tip="int8 ConvRot bases (MiniMax H3, Krea 2 int8): each frozen layer's input gradient from a fused Triton kernel instead of a rebuilt bf16 weight (Fizgig's, by @mabseyuk). Saves the rebuilt weight's VRAM; not faster on an RTX 5090 (cuBLAS wins)">
              <div class="sw-row"><ZenSwitch :model-value="!!spec.memory.fused_int8_backward" @update:model-value="spec.memory.fused_int8_backward = $event" /><span class="sw-state">{{ spec.memory.fused_int8_backward ? 'On' : 'Off' }}</span></div>
            </ZenField>
            <ZenField label="Bypass mode" stack tip="Forward hooks; automatic for quantized bases"><div class="sw-row"><ZenSwitch v-model="spec.memory.bypass_mode" /><span class="sw-state">{{ spec.memory.bypass_mode ? 'On' : 'Off' }}</span></div></ZenField>
          </div>
        </section>
      </div>

      <aside class="summary card">
        <ZenField label="Run name" stack><ZenInput v-model="spec.name" /></ZenField>
        <dl>
          <dt>Model</dt><dd class="ellipsis" :title="modelLabel ?? ''">{{ modelLabel || '–' }}</dd>
          <dt>Dataset</dt><dd>{{ spec.dataset.folder || '–' }}<span v-if="dataset" class="muted"> · {{ dataset.images }} imgs</span></dd>
          <dt>Captions</dt><dd>{{ spec.dataset.caption_mix.length ? spec.dataset.caption_mix.join(' + ') : spec.dataset.caption_set }}</dd>
          <dt>Trigger</dt><dd>{{ spec.dataset.trigger || '–' }}</dd>
          <dt>Steps</dt><dd class="mono">{{ spec.optim.steps }}<span v-if="epochs != null" class="muted"> · {{ epochs.toFixed(1) }} ep</span></dd>
          <dt>Optimizer</dt><dd>{{ opt?.label ?? spec.optim.optimizer }} <span class="mono muted">{{ spec.optim.learning_rate }}</span></dd>
          <dt>Network</dt><dd class="mono">{{ spec.network.algorithm }} r{{ spec.network.rank }}</dd>
          <dt>Resolution</dt><dd class="mono">{{ spec.dataset.resolution }}</dd>
          <dt>Samples</dt><dd>{{ spec.samples.prompts.length }} prompt{{ spec.samples.prompts.length === 1 ? '' : 's' }}{{ spec.samples.every ? ` / ${spec.samples.every} steps` : '' }}</dd>
        </dl>
        <div v-if="applied.length" class="muted small">Presets: {{ applied.join(' + ') }}</div>
        <ul v-if="warnings.length" class="warns">
          <li v-for="w in warnings" :key="w"><i class="mdi mdi-alert-outline" /> {{ w }}</li>
        </ul>
        <div v-if="error" class="err mono">{{ error }}</div>
        <template v-if="editing">
          <ZenButton variant="primary" icon="mdi mdi-content-save-outline" block :disabled="submitting || !!missing" :title="missing ?? (resuming ? 'Saves these settings and resumes it from its last state' : retrying ? 'Saves these settings and queues it to run next' : 'Keeps its place in the queue')" @click="saveEdit">
            {{ missing ?? (resuming ? 'Save & resume' : retrying ? 'Save & retry' : 'Save changes') }}
          </ZenButton>
          <ZenButton icon="mdi mdi-close" block @click="router.push('/jobs')">Cancel</ZenButton>
        </template>
        <template v-else>
          <ZenButton variant="primary" icon="mdi mdi-play" block :disabled="submitting || !!missing" :title="missing ?? 'Starts now, or right after the current run'" @click="start('next')">
            {{ missing ?? 'Start training' }}
          </ZenButton>
          <ZenButton icon="mdi mdi-tray-plus" block :disabled="submitting || !!missing" title="Line it up behind everything queued" @click="start('end')">Add to queue</ZenButton>
        </template>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.wide {
  max-width: 1280px;
}
.presets {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.preset {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 30px;
  padding: 0 12px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 80%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 50%, transparent);
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.preset:hover,
.preset.on {
  border-color: var(--zen-accent);
}
.preset.on {
  background: color-mix(in srgb, var(--zen-accent) 16%, transparent);
}
.preset.ghost {
  color: var(--zen-muted);
  border-style: dashed;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 18px;
  align-items: start;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.sec {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px;
}
.video-opts {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.sub-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 13px;
}
h2 {
  margin: 0;
  font-size: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.num {
  width: 22px;
  height: 22px;
  border-radius: var(--r-lg);
  display: grid;
  place-items: center;
  font-size: 12px;
  background: var(--zen-accent);
  color: var(--zen-accent-text);
}
.seg {
  display: inline-flex;
  align-self: flex-start;
  flex-wrap: wrap;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.locked {
  opacity: 0.5;
  pointer-events: none;
}
.locked-note {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  color: var(--zen-muted);
  line-height: 1.5;
}
.seg button,
.chip-btn {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.seg button.on,
.chip-btn.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.chip-btn {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.chips {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}
.wide-f {
  grid-column: 1 / -1;
}
.span-2 {
  grid-column: span 2;
}
@media (max-width: 760px) {
  .span-2 {
    grid-column: 1 / -1;
  }
}
/* a switch field: label above like the rest, the switch on a control-height row beside its state */
.tpl-row {
  display: flex;
  gap: 6px;
  align-items: center;
}
.tpl-row > :first-child {
  flex: 1;
  min-width: 0;
}
.sw-row {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 32px;
}
.sw-row > :deep(.zen-number) {
  flex: 1;
}
.sw-state {
  font-size: 12px;
  color: var(--zen-muted);
}
.adv {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.subgroup {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sub-title {
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted);
  padding-bottom: 6px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.inline {
  display: flex;
  align-items: center;
  gap: 8px;
}
.inline > * {
  flex: 1;
}
.inline > .muted,
.alpha > :first-child {
  flex: none;
}
.more {
  align-self: flex-start;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  padding: 0;
}
.more:hover {
  color: var(--zen-text);
}
.more.head {
  font-size: 14px;
  font-weight: 600;
  color: var(--zen-text);
}
.small {
  font-size: 11px;
  font-weight: 400;
}
.place-opt {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
}
.place-opt .ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sample-head {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.sample-head {
  min-height: 32px;
}
.sample-head > :first-child {
  width: 240px;
}
.set-pick {
  position: relative;
}
.set-dot {
  position: absolute;
  top: 50%;
  right: 30px;
  width: 7px;
  height: 7px;
  margin-top: -3.5px;
  border-radius: var(--r-sm);
  background: var(--zen-warn);
  pointer-events: none;
}
.spacer {
  flex: 1;
}
.ds-opt {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}
.ds-opt img {
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.opt-row {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  min-width: 0;
  font-size: 12px;
}
.opt-tip {
  display: inline-flex;
}
.opt-row.off {
  opacity: 0.5;
}
.summary {
  position: sticky;
  top: 8px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
}
dl {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 6px 12px;
  margin: 0;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
}
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.custom-model {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
}
.link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: none;
  background: none;
  padding: 0;
  color: var(--zen-muted);
  font: inherit;
  cursor: pointer;
  text-decoration: none;
}
.link:hover {
  color: var(--zen-accent);
}
.small {
  font-size: 11px;
}
.profile {
  margin-top: 10px;
  padding: 8px 12px;
  border-radius: var(--r-lg);
  font-size: 12px;
  background: color-mix(in srgb, var(--zen-accent) 7%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 25%, transparent);
}
.profile-head {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 26px;
}
.profile-head > i {
  font-size: 16px;
  color: var(--zen-accent);
}
.profile-sum {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.open .profile-sum {
  white-space: normal;
}
.profile-state {
  min-width: 64px;
  display: inline-flex;
  justify-content: flex-end;
}
.profile-body {
  padding: 6px 0 2px 26px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.profile-notes {
  margin: 0;
  padding-left: 16px;
  line-height: 1.5;
  color: var(--zen-muted);
}
.profile-src {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}
.suggest {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: var(--r-lg);
  font-size: 12px;
  line-height: 1.45;
  background: color-mix(in srgb, var(--zen-accent) 9%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 30%, transparent);
}
.suggest > i {
  font-size: 18px;
  color: var(--zen-accent);
}
.suggest > span:not(.muted) {
  flex: 1;
}
.warns {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 12px;
}
.warns li {
  display: flex;
  gap: 6px;
  color: var(--zen-warn);
}
.err {
  font-size: 11px;
  white-space: pre-wrap;
  color: var(--zen-danger);
}
@media (max-width: 1000px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .summary {
    position: static;
  }
}
</style>
