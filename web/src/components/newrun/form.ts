// The New run form as its sections see it: the spec they all edit, what the server offers
// (datasets, optimizers, presets, model files), the run being edited if any, and what's
// derived across sections (the dataset, the optimizer, epochs, the model's noise schedule).
// NewRunView makes it; the sections take it with useRunForm().
import { computed, inject, onMounted, reactive, ref, toRaw, watch, type InjectionKey } from 'vue'
import { useRoute } from 'vue-router'
import { inputChoices } from '@/api/comfy'
import type { SampleSuggestion } from '@/components/FamilyPicker.vue'
import { setSetting, settings } from '@/settings'
import {
  defaultSpec,
  engine,
  type DatasetInfo,
  type DatasetItem,
  type ModelProfile,
  type NoiseProfile,
  type OptimizerInfo,
  type Preset,
  type TrainSpec,
} from '@/api/engine'

// combo items from plain values, with optional labels
export const items = (vals: string[] | undefined, labels?: Record<string, string>) =>
  (vals ?? []).map((v) => ({ value: v, label: labels?.[v] ?? v, keywords: v }))

// presets: deep-merge a partial spec onto the form (so they stack)
export function merge(target: Record<string, unknown>, src: Record<string, unknown>) {
  for (const [k, v] of Object.entries(src)) {
    if (v && typeof v === 'object' && !Array.isArray(v) && target[k] && typeof target[k] === 'object')
      merge(target[k] as Record<string, unknown>, v as Record<string, unknown>)
    else target[k] = structuredClone(toRaw(v)) // presets live in refs: clone the plain object, not the proxy
  }
}

function nextName(name: string): string {
  const m = name.match(/^(.*)-(\d+)$/)
  return m ? `${m[1]}-${Number(m[2]) + 1}` : `${name}-2`
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

export function useRunForm() {
  const route = useRoute()
  const spec = reactive(defaultSpec())

  // ---- where checkpoints go: one of ComfyUI's loras folders ---------------------------------
  const loraPlaces = ref<Awaited<ReturnType<typeof engine.loraPlaces>>>([])
  watch(() => spec.output.keep_states, (v) => setSetting('keepStates', v))

  // ---- choices from the server ------------------------------------------------------------
  const choices = reactive<Record<string, string[]>>({})
  // the model picker: files still to download, the family's sampling suggestion
  const modelMissing = ref(0)
  const sampleSuggestion = ref<SampleSuggestion | null>(null)
  // the family's hand-picked settings: applied when the family is picked, re-appliable after edits
  const profile = ref<ModelProfile | null>(null)
  const familyId = ref<string | null>(null)
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

  // ---- editing a run that exists ------------------------------------------------------------
  const editing = ref<string | null>(null)
  const retrying = ref(false) // a run that ended before training: saving requeues it
  const resuming = ref(false) // a stopped run that has trained: saving resumes it with the new settings
  const resumeStep = ref(0)
  const error = ref<string | null>(null)

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

  // ---- presets ---------------------------------------------------------------------------------
  const applied = ref<string[]>([])
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

  // video models get a frames field per sample (and the run's own)
  const isVideoModel = computed(() => familyId.value === 'minimax-h3' || (spec.samples.frames ?? 1) > 1)
  // start lower: one earlier stage at a lower resolution (the engine takes several; the form edits one)
  const stage0 = computed(() => spec.dataset.stages?.[0])
  watch(
    () => spec.dataset.resolution,
    (r) => {
      if (spec.dataset.stages?.some((st) => st.resolution >= r)) spec.dataset.stages = spec.dataset.stages.filter((st) => st.resolution < r)
    },
  )
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

  const dataset = computed(() => datasets.value.find((d) => d.folder === spec.dataset.folder))
  // a mask says "only this part": picking a dataset that has some turns Masked training on
  watch(
    () => spec.dataset.folder,
    (folder, was) => {
      if (was !== undefined && folder !== was) spec.dataset.masked_loss = (dataset.value?.masked ?? 0) > 0
    },
  )
  const opt = computed(() => optimizers.value.find((o) => o.id === spec.optim.optimizer))
  const epochs = computed(() => {
    const per = dataset.value?.steps_per_epoch
    return per ? (spec.optim.steps * spec.optim.batch_size * spec.optim.grad_accumulation_steps) / per : null
  })
  const modelLabel = computed(() => (spec.model.kind === 'checkpoint' ? spec.model.ckpt_name : spec.model.unet_name))
  // the picked model's own noise schedule (read from its file's header), for the Focus control
  const noise = ref<NoiseProfile>({ known: false })
  watch(
    () => [spec.model.kind, modelLabel.value] as const,
    async ([kind, name]) => {
      noise.value = { known: false }
      if (!name) return
      const folder = kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'
      try {
        const p = await engine.noise(folder, name)
        if (modelLabel.value === name) noise.value = p
      } catch {
        /* no profile: the plain shift field stays */
      }
    },
    { immediate: true },
  )

  return {
    spec, loraPlaces, choices, modelMissing, sampleSuggestion, profile, familyId,
    datasets, optimizers, trainAttention, runPresets, promptSets, pairCount, pairItems,
    editing, retrying, resuming, resumeStep, error,
    applied, applyPreset, savePreset, isVideoModel, stage0, dataset, opt, epochs, modelLabel, noise,
  }
}

export type RunForm = ReturnType<typeof useRunForm>
export const RUN_FORM: InjectionKey<RunForm> = Symbol('runForm')
export const useRunFormContext = () => inject(RUN_FORM)!
