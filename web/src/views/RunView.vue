<script setup lang="ts">
// One run. Header with the controls; progress beside the latest samples; the loss and
// validation curves; then samples as a prompt × step grid, so a prompt's row reads as
// its training progression (click a cell to page through that row in the lightbox).
import { ZenButton, ZenCombo, ZenCompare, ZenLightbox, ZenModal, ZenNumber, ZenSlider, ZenSwitch } from '@nynxz/zenkit-ui'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { loadedModels, watchLoaded } from '@/loaded'
import { gb } from '@/system'
import { vAutosize } from '@/autosize'
import { onComfyMessage } from '@/api/comfy'
import {
  ACTIVE,
  EVENTS,
  engine,
  promptText as specPrompt,
  viewUrl,
  type CheckpointRow,
  type Command,
  type MetricRow,
  type RunDetail,
  type RunStatus,
  type SampleProgress,
  type SampleRow,
  type ValidationRow,
} from '@/api/engine'
import LossChart from '@/components/LossChart.vue'
import ScrollArea from '@/components/ScrollArea.vue'
import { fmtBytes } from '@/downloads'
import StatusPill from '@/components/StatusPill.vue'

const props = defineProps<{ id: string }>()
const router = useRouter()

const detail = ref<RunDetail | null>(null)
const error = ref<string | null>(null)
const busy = ref<string | null>(null)
const lrInput = ref(0)
const lrOpen = ref(false)
const sampling = ref<SampleProgress | null>(null)
const now = ref(Date.now() / 1000)
let clock: ReturnType<typeof setInterval> | undefined

async function load() {
  try {
    detail.value = await engine.run(props.id)
    lrInput.value = detail.value.metrics.at(-1)?.lr ?? detail.value.spec.optim.learning_rate
  } catch (e) {
    error.value = String(e)
  }
}

let off = () => {}
onMounted(() => {
  load()
  clock = setInterval(() => (now.value = Date.now() / 1000), 1000)
  off = onComfyMessage((m) => {
    const d = detail.value
    if (!d || m.data.run_id !== props.id) return
    if (m.type === EVENTS.metrics) {
      const row = m.data as unknown as MetricRow
      if (!d.metrics.length || d.metrics.at(-1)!.step < row.step) d.metrics.push(row)
    } else if (m.type === EVENTS.checkpoint) {
      d.checkpoints.push(m.data as unknown as CheckpointRow)
    } else if (m.type === EVENTS.sample) {
      const row = m.data as unknown as SampleRow
      d.samples.push(row)
      if (row.manual && !row.base) queuedTries.value = queuedTries.value.slice(1) // its render landed
    } else if (m.type === EVENTS.validation) {
      d.validation.push(m.data as unknown as ValidationRow)
    } else if (m.type === EVENTS.sampleProgress) {
      const p = m.data as unknown as SampleProgress
      if (p.phase === 'done') setTimeout(() => (sampling.value = null), 600)
      else
        sampling.value = {
          ...sampling.value,
          ...p,
          preview: p.preview ?? (p.index === sampling.value?.index ? sampling.value?.preview : undefined),
        }
    } else if (m.type === EVENTS.status) {
      d.run.status = m.data.status as RunStatus
      d.run.message = (m.data.message as string | null) ?? undefined
      d.live = ACTIVE.includes(d.run.status)
    }
  })
})
onUnmounted(() => {
  off()
  clearInterval(clock)
})
watch(() => props.id, load)

// ---- progress -----------------------------------------------------------------------------
const last = computed(() => detail.value?.metrics.at(-1))
const total = computed(() => detail.value?.spec.optim.steps ?? 0)
const step = computed(() => last.value?.step ?? 0)
const progress = computed(() => (total.value ? step.value / total.value : 0))
const status = computed(() => detail.value?.run.status)
const running = computed(() => !!status.value && ACTIVE.includes(status.value))
const dur = (s: number | null | undefined) => {
  if (s == null || !isFinite(s)) return '–'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60)
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : m ? `${m}m ${String(sec).padStart(2, '0')}s` : `${sec}s`
}
const eta = computed(() => {
  const l = last.value
  if (!l?.it_s || status.value !== 'training') return null
  return (total.value - l.step) / l.it_s
})
// what the run is doing when it isn't stepping: shown in the step bar's place (same size), with
// its own progress where there is one (a sample's steps, the dataset's "(12/120)")
const phase = computed(() => {
  const d = detail.value
  if (!d || !running.value) return null
  const sm = sampling.value
  if (sm && sm.phase !== 'done') {
    const baseline = step.value === 0 && d.spec.samples.baseline
    const st = sm.s_total ?? 0
    const s0 = sm.s_step ?? 0
    return {
      icon: 'mdi mdi-image-filter-center-focus-weak',
      label: baseline ? 'Baseline' : 'Sampling',
      text: (baseline ? 'The model before training · ' : '') + (sm.prompt ?? ''),
      frac: st ? s0 / st : 0,
      right: `${sm.count ? `${(sm.index ?? 0) + 1}/${sm.count} · ` : ''}${st && s0 >= st ? 'decoding' : st ? `${s0}/${st}` : ''}`,
    }
  }
  if (status.value === 'queued') return { icon: 'mdi mdi-clock-outline', label: 'Queued', text: 'Waiting for its turn', frac: 0, right: '' }
  if (status.value === 'loading') {
    const msg = d.run.message || 'Loading the model and the dataset'
    const m = msg.match(/\((\d+)\/(\d+)\)\s*$/)
    return {
      icon: 'mdi mdi-database-arrow-down-outline',
      label: 'Preparing',
      text: m ? msg.slice(0, m.index).trim() : msg,
      frac: m ? Number(m[1]) / Number(m[2]) : 0,
      right: m ? `${m[1]}/${m[2]}` : '',
    }
  }
  if (status.value === 'training' && !last.value) return { icon: 'mdi mdi-progress-helper', label: 'Starting', text: d.run.message || 'First steps', frac: 0, right: '' }
  return null
})
const wall = computed(() => {
  const d = detail.value
  if (!d) return null
  return (running.value ? now.value : d.run.updated) - d.run.created
})
const marks = computed(() => {
  const d = detail.value
  if (!d) return []
  const out: { step: number; kind: 'ckpt' | 'sample'; done: boolean }[] = []
  const saved = new Set(d.checkpoints.map((c) => c.step))
  const sampled = new Set(d.samples.filter((x) => !x.manual).map((x) => x.step))
  const every = d.spec.save_every
  if (every) for (let s = every; s < total.value; s += every) out.push({ step: s, kind: 'ckpt', done: saved.has(s) })
  const se = d.spec.samples.every
  if (se) for (let s = se; s <= total.value; s += se) out.push({ step: s, kind: 'sample', done: sampled.has(s) })
  return out
})
const nextSave = computed(() => {
  const every = detail.value?.spec.save_every
  if (!every || !last.value?.it_s) return null
  const next = (Math.floor(step.value / every) + 1) * every
  if (next >= total.value) return null
  return { step: next, in: dur((next - step.value) / last.value.it_s) }
})
const nextSample = computed(() => {
  const every = detail.value?.spec.samples.every
  if (!every || !last.value?.it_s) return null
  const next = (Math.floor(step.value / every) + 1) * every
  if (next > total.value) return null
  return { step: next, in: dur((next - step.value) / last.value.it_s) }
})
const lastCkpt = computed(() => detail.value?.checkpoints.at(-1))
const valLast = computed(() => detail.value?.validation.at(-1))
const valFirst = computed(() => detail.value?.validation[0])

// ---- samples: prompt × step grid ---------------------------------------------------------------
// the run's own samples (its prompts at its steps) vs one-off "Try a prompt" renders
const runSamples = computed(() => (detail.value?.samples ?? []).filter((x) => !x.manual))
// each try is a card: its no-LoRA render (if asked for) and every step it was rendered at
const manualGroups = computed(() => {
  const by = new Map<number, { index: number; base: SampleRow | null; steps: SampleRow[] }>()
  for (const x of detail.value?.samples ?? []) {
    if (!x.manual) continue
    const g = by.get(x.index) ?? { index: x.index, base: null, steps: [] }
    if (x.base) g.base = x
    else g.steps.push(x)
    by.set(x.index, g)
  }
  for (const g of by.values()) g.steps.sort((a, b) => a.step - b.step)
  return [...by.values()].filter((g) => g.steps.length || g.base).sort((a, b) => b.index - a.index)
})
const manualCount = computed(() => manualGroups.value.length)
const pickedStep = ref<Record<number, number>>({}) // try -> which step the slider compares
const shownStep = (g: { index: number; steps: SampleRow[] }) =>
  g.steps.find((x) => x.step === pickedStep.value[g.index]) ?? g.steps.at(-1) ?? null
async function again(g: { index: number; base: SampleRow | null; steps: SampleRow[] }) {
  const x = g.steps.at(-1) ?? g.base
  if (!x) return
  sampleView.value = 'manual'
  delete pickedStep.value[g.index]
  await queueTry({ prompt: x.prompt, width: x.width, height: x.height, frames: x.frames, seed: x.seed, base: false, index: g.index }, x.prompt, g.index)
}

// manual tries waiting their turn: the engine runs sample commands one after another at step
// boundaries, so any number can be queued; each leaves this list when its render lands
const queuedTries = ref<{ id: number; prompt: string; again?: number }[]>([])
let tryIds = 0
async function queueTry(args: Record<string, unknown>, prompt: string, again?: number) {
  const id = ++tryIds
  queuedTries.value = [...queuedTries.value, { id, prompt, again }]
  try {
    await engine.control(props.id, 'sample', args)
  } catch (e) {
    queuedTries.value = queuedTries.value.filter((q) => q.id !== id)
    error.value = String((e as Error).message ?? e)
  }
}
const steps = computed(() => [...new Set(runSamples.value.map((x) => x.step))].sort((a, b) => a - b))
const prompts = computed(() => {
  const by = new Map<number, string>()
  for (const x of runSamples.value) by.set(x.index, x.prompt)
  return [...by.entries()].sort((a, b) => a[0] - b[0])
})
const cell = (index: number, s: number) => runSamples.value.find((x) => x.index === index && x.step === s)
const latest = computed(() => {
  const s = steps.value.at(-1)
  if (s == null) return []
  const byIndex = new Map<number, SampleRow>()
  for (const x of runSamples.value) if (x.step === s) byIndex.set(x.index, x)
  return [...byIndex.values()].sort((a, b) => a.index - b.index)
})
const promptCount = computed(() => Math.max(detail.value?.spec.samples.prompts.length ?? 0, prompts.value.length))
// each prompt's newest sample: mid-pass, the prompts already rendered show the new step
// while the rest keep the previous one until their turn
const latestByIndex = computed(() => {
  const by = new Map<number, SampleRow>()
  for (const x of runSamples.value) if ((by.get(x.index)?.step ?? -1) <= x.step) by.set(x.index, x)
  return by
})
const latestFor = (i: number) => latestByIndex.value.get(i)
const promptText = (i: number) => latestFor(i)?.prompt ?? prompts.value.find(([k]) => k === i)?.[1] ?? (detail.value?.spec.samples.prompts[i] != null ? specPrompt(detail.value.spec.samples.prompts[i]!) : '')
const specOpen = ref(false)
const sampleView = ref<'latest' | 'grid' | 'manual'>('latest')
// "Try a prompt": one render at the current step, apart from the run's samples
const tryPrompt = ref('')
const trySize = ref('')
const tryFrames = ref<number | null>(null)
const tryBase = ref(true)
async function renderTry() {
  // one try per line: several prompts queue at once
  const lines = tryPrompt.value.split('\n').map((l) => l.trim()).filter(Boolean)
  if (!lines.length) return
  const [w, h] = trySize.value ? trySize.value.split('x').map(Number) : [undefined, undefined]
  tryPrompt.value = ''
  for (const prompt of lines)
    await queueTry({ prompt, width: w, height: h, frames: tryFrames.value ?? undefined, base: tryBase.value }, prompt)
}
const trySizes = computed(() => {
  const sp = detail.value?.spec.samples
  return [
    { value: '', label: sp ? `Run size (${sp.width}×${sp.height})` : 'Run size' },
    ...[[1024, 1024], [832, 1216], [1216, 832], [480, 832], [832, 480], [512, 512]].map(([w, h]) => ({ value: `${w}x${h}`, label: `${w}×${h}` })),
  ]
})
const fmtLr = (v?: number | null) => (v == null ? '–' : v === 0 ? '0' : v.toExponential(1))
interface Stat {
  k: string
  v: string
  mono?: boolean
  delta?: { good: boolean; text: string } | null
}
const statGroups = computed((): { title: string; items: Stat[] }[] => {
  const d = detail.value
  if (!d) return []
  const l = last.value
  const valDelta =
    valLast.value && valFirst.value && valLast.value !== valFirst.value
      ? {
          good: valLast.value.val_loss < valFirst.value.val_loss,
          text: ` ${valLast.value.val_loss < valFirst.value.val_loss ? '▼' : '▲'}${Math.abs(((valLast.value.val_loss - valFirst.value.val_loss) / valFirst.value.val_loss) * 100).toFixed(0)}%`,
        }
      : null
  return [
    {
      title: 'Time',
      items: [
        { k: 'Training', v: dur(l?.train_time) },
        { k: 'Total', v: dur(wall.value) }, // start to now, loading and sampling included
        { k: 'Speed', v: `${l?.it_s?.toFixed(2) ?? '–'} it/s` },
        { k: 'Epoch', v: String(l?.epoch ?? '–') },
        ...(l?.vram_peak ? [{ k: 'Peak VRAM', v: `${gb(l.vram_peak)} GB` }] : []),
      ],
    },
    {
      title: 'Learning',
      items: [
        { k: 'Learning rate', v: fmtLr(l?.lr) },
        { k: 'Grad norm', v: l?.grad_norm?.toFixed(3) ?? '–' },
        { k: 'Validation', v: valLast.value ? valLast.value.val_loss.toFixed(4) : '–', delta: valDelta },
        { k: 'Loss', v: l?.loss?.toFixed(4) ?? '–' },
      ],
    },
    {
      title: 'Setup',
      items: [
        { k: 'Optimizer', v: `${d.spec.optim.optimizer} · ${d.spec.optim.scheduler}`, mono: false },
        { k: 'Network', v: `${d.spec.network.algorithm} r${d.spec.network.rank}` },
      ],
    },
  ]
})
// validation loss at a checkpoint: the latest evaluation at or before its step
const ckptVal = (stepAt: number) => {
  let v: number | undefined
  for (const r of detail.value?.validation ?? []) if (r.step <= stepAt) v = r.val_loss
  return v ?? null
}
const bestStep = computed(() => {
  const cks = (detail.value?.checkpoints ?? []).filter((c) => ckptVal(c.step) != null)
  if (cks.length < 2) return null
  return cks.reduce((a, b) => (ckptVal(b.step)! < ckptVal(a.step)! ? b : a)).step
})
// how far into the run it was saved
const ckptAt = (stepAt: number) => {
  const row = detail.value?.metrics.find((m) => m.step >= stepAt)
  return row?.elapsed != null ? `${dur(row.elapsed)} in` : ''
}
const saveLine = computed(() => {
  const n = detail.value?.checkpoints.length ?? 0
  const next = nextSave.value
  const plan = detail.value?.spec.save_every ? `every ${detail.value.spec.save_every}` : 'final only'
  if (running.value && next) return `${n ? `${n} · ` : ''}next ${next.step} · ${next.in}`
  return n ? `${n} · ${plan}` : plan
})
const etaLine = computed(() => {
  const pct = `${Math.round(progress.value * 100)}%`
  if (eta.value != null) return `${pct} · ${dur(eta.value)} left`
  if (status.value === 'done') return `${pct} · in ${dur(last.value?.train_time)}`
  return pct
})
const samplesLine = computed(() => {
  const sp = sampling.value
  if (sp?.phase === 'requested') return 'requested · after this training step'
  if (sp?.phase === 'sampling') return `rendering ${(sp.index ?? 0) + 1}/${sp.count} at step ${sp.step}`
  const x = latest.value[0]
  const at = !x ? 'none yet' : x.step === 0 ? 'baseline' : `step ${x.step}`
  const next = nextSample.value
  return running.value && next ? `${at} · next ${next.step} · ${next.in}` : at
})
const liveIndex = computed(() => (sampling.value?.phase === 'sampling' && !sampling.value.manual ? (sampling.value.index ?? -1) : -1))
const liveManual = computed(() => (sampling.value?.manual ? sampling.value : null))
// Latest: one column per prompt (2-4), filling the card
const latestCols = computed(() => Math.min(4, Math.max(2, promptCount.value)))
// Progression: thumbnails sized so every step fits the card, until the slider picks a size
const samplesCard = ref<HTMLElement | null>(null)
const cardWidth = ref(800)
let cardRO: ResizeObserver | undefined
watch(samplesCard, (el) => {
  cardRO?.disconnect()
  if (!el) return
  cardRO = new ResizeObserver(() => (cardWidth.value = el.clientWidth - 36)) // minus the card's padding
  cardRO.observe(el)
})
onUnmounted(() => cardRO?.disconnect())
const thumbPick = ref<number | null>(null)
const fitSize = computed(() => {
  const n = Math.max(1, steps.value.length)
  return Math.max(64, Math.min(200, Math.floor((cardWidth.value - (n - 1) * 6) / n)))
})
const gridSize = computed(() => thumbPick.value ?? fitSize.value)
// a step that isn't on the sampling schedule: someone pressed Sample now
const isManual = (s: number) => {
  const every = detail.value?.spec.samples.every ?? 0
  return s !== 0 && s !== total.value && (!every || s % every !== 0)
}
const gridScroll = ref<InstanceType<typeof ScrollArea> | null>(null)
// newest steps in view: when new ones land, and when Progression opens
watch([() => steps.value.length, sampleView], async () => {
  await nextTick()
  gridScroll.value?.scrollToEnd()
})

const lb = ref<{ items: LightboxItem[]; index: number; grid?: number } | null>(null)
// the run's own models among what ComfyUI holds (matched by file), while it's live
const runLoaded = computed(() => {
  const m = detail.value?.spec.model
  if (!m || !running.value) return []
  const files = new Set(
    [m.unet_name, m.ckpt_name, m.clip_name, m.vae_name, (m as { audio_vae_name?: string | null }).audio_vae_name, detail.value?.spec.samples.model_name]
      .filter(Boolean)
      .map((f) => String(f).split('/').pop()),
  )
  return loadedModels.value.filter((x) => x.file && x.file.split(' + ').some((f) => files.has(f)))
})
let stopLoaded = () => {}
onMounted(() => (stopLoaded = watchLoaded()))
onUnmounted(() => stopLoaded())

// tiles take the samples' own shape, so neither stills nor videos are cropped
const sampleAspect = computed(() => {
  const sp = detail.value?.spec.samples
  return sp?.width && sp?.height ? `${sp.width} / ${sp.height}` : '1'
})
function openManual(i: number, row?: SampleRow | null) {
  // the viewer pages through every try (its no-LoRA render first, then each step) and opens on
  // the render that was clicked
  const rows = (g: (typeof manualGroups.value)[number]) => [...(g.base ? [g.base] : []), ...g.steps]
  const flat = manualGroups.value.flatMap(rows)
  const start = manualGroups.value.slice(0, i).reduce((n, g) => n + rows(g).length, 0)
  const own = rows(manualGroups.value[i]!)
  const at = start + Math.max(0, row ? own.indexOf(row) : own.length - 1)
  lb.value = { items: flat.map((x) => ({ ...sampleItem(x), label: x.base ? 'Without the LoRA' : `Step ${x.step}` })), index: at }
}
// a tile shaped like its render (pairs side by side: twice as wide), so nothing letterboxes
const shapeOf = (x: SampleRow | null | undefined, pair = false) =>
  x?.width && x?.height ? { aspectRatio: `${x.width * (pair ? 2 : 1)} / ${x.height}` } : undefined
const isVideo = (x: SampleRow) => /\.(mp4|webm)$/i.test(x.filename)
const sampleItem = (x: SampleRow): LightboxItem => ({
  src: viewUrl(x),
  kind: isVideo(x) ? 'video' : 'image',
  label: `${x.index + 1} · ${x.step === 0 ? 'baseline' : `step ${x.step}`}${x.model ? ` · ${x.model}` : ''}`,
  meta: x.prompt,
})
// The lightbox walks the samples as a grid: a row per prompt, a column per step, so
// left/right (A/D) is time and up/down (W/S) is the prompt. Only steps every prompt has
// rendered make a column; a step still sampling opens as a lone row.
function openGrid(index: number, s: number) {
  const rows = prompts.value.map(([i]) => i)
  const cols = steps.value.filter((st) => rows.every((i) => cell(i, st)))
  if (!cols.includes(s)) {
    const row = steps.value.map((st) => cell(index, st)).filter((x): x is SampleRow => !!x)
    lb.value = { items: row.map(sampleItem), index: Math.max(0, row.findIndex((x) => x.step === s)) }
    return
  }
  lb.value = {
    items: rows.flatMap((i) => cols.map((st) => sampleItem(cell(i, st)!))),
    index: rows.indexOf(index) * cols.length + cols.indexOf(s),
    grid: cols.length,
  }
}
const openLatest = (index: number) => openGrid(index, latestFor(index)!.step)

// ---- actions ------------------------------------------------------------------------------
async function send(cmd: Command, args?: Record<string, unknown>) {
  busy.value = cmd
  if (cmd === 'sample') sampling.value = { run_id: props.id, phase: 'requested' }
  try {
    await engine.control(props.id, cmd, args)
  } catch (e) {
    error.value = String(e)
  } finally {
    setTimeout(() => (busy.value = null), 400)
  }
}
const resumable = computed(() => ['error', 'stopped', 'interrupted', 'rejected'].includes(status.value ?? ''))
// ended before it trained a step: its settings can still change (the backend requeues it)
const retryable = computed(() => resumable.value && !detail.value?.metrics.length && !detail.value?.checkpoints.length)
const resumePoint = computed(() => {
  const c = detail.value?.checkpoints ?? []
  return c.length ? Math.max(...c.map((x) => x.step)) : 0
})
async function resume(checkpoint?: string) {
  busy.value = 'resume'
  error.value = null
  try {
    await engine.resume(props.id, checkpoint)
    await load()
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    busy.value = null
  }
}
const copied = ref<string | null>(null)
async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = text
    setTimeout(() => copied.value === text && (copied.value = null), 1500)
  } catch {
    // clipboard blocked
  }
}
const fileName = (p: string) => p.split(/[\\/]/).at(-1)
</script>

<template>
  <div class="page wide">
    <div v-if="error && !detail" class="card muted">{{ error }}</div>
    <template v-else-if="detail">
      <div class="head">
        <div class="titles">
          <div class="title-row">
            <h1>{{ detail.run.name }}</h1>
            <StatusPill :status="detail.run.status" />
          </div>
          <p class="lede mono">
            {{ detail.run.run_id }} · {{ detail.spec.model.kind === 'split' ? detail.spec.model.unet_name : detail.spec.model.ckpt_name }}
          </p>
        </div>
        <div class="actions">
          <template v-if="running">
            <ZenButton v-if="status === 'paused'" variant="primary" icon="mdi mdi-play" :disabled="!!busy" @click="send('resume')">Resume</ZenButton>
            <ZenButton v-else icon="mdi mdi-pause" :disabled="!!busy || status !== 'training'" @click="send('pause')">Pause</ZenButton>
            <ZenButton icon="mdi mdi-content-save-outline" :disabled="!!busy || status !== 'training'" title="Save a checkpoint now" @click="send('save')">Save</ZenButton>
            <div class="lr-wrap">
              <ZenButton icon="mdi mdi-tune-variant" title="Change the learning rate" @click="lrOpen = !lrOpen">LR</ZenButton>
              <div v-if="lrOpen" class="lr-pop card">
                <span class="eyebrow">Learning rate</span>
                <ZenNumber v-model="lrInput" :min="0" :step="0.00001" :precision="6" size="sm" />
                <ZenButton size="sm" variant="primary" @click="send('set_lr', { lr: lrInput }); lrOpen = false">Apply</ZenButton>
              </div>
            </div>
            <ZenButton variant="danger" icon="mdi mdi-stop" :disabled="!!busy" @click="send('stop')">Stop</ZenButton>
          </template>
          <RouterLink v-if="retryable" :to="`/runs/new?edit=${id}`"><ZenButton icon="mdi mdi-pencil-outline" title="Change its settings, then it goes back in the queue">Edit &amp; retry</ZenButton></RouterLink>
          <ZenButton v-if="resumable" variant="primary" icon="mdi mdi-play-circle-outline" :disabled="!!busy" @click="resume()">
            {{ resumePoint ? `Resume from ${resumePoint}` : 'Restart' }}
          </ZenButton>
          <ZenButton icon="mdi mdi-content-copy" title="New run with these settings" @click="router.push(`/runs/new?from=${id}`)">Duplicate</ZenButton>
        </div>
      </div>
      <div v-if="error" class="card err mono">{{ error }}</div>
      <div v-if="detail.run.message && !running" class="card muted mono msg">{{ detail.run.message }}</div>

      <!-- one slim line: fixed-width text columns either side, so the bar (and its notches) never moves -->
      <div v-if="phase" class="progline card phase">
        <span class="pl-step"><i :class="phase.icon" /> {{ phase.label }}</span>
        <div class="ph-mid">
          <span class="ph-text ellipsis" :title="phase.text">{{ phase.text }}</span>
          <div class="bar thin"><div :style="{ width: `${phase.frac * 100}%` }" /></div>
        </div>
        <span class="pl-eta mono muted">{{ phase.right }}</span>
      </div>
      <div v-else class="progline card">
        <span class="pl-step mono"><b>{{ step }}</b><span class="muted"> / {{ total }}</span></span>
        <div class="track">
          <div class="bar"><div :style="{ width: `${progress * 100}%` }" /></div>
          <span
            v-for="m in marks"
            :key="`${m.kind}${m.step}`"
            class="notch"
            :class="[m.kind, { done: m.done }]"
            :style="{ left: `${(m.step / total) * 100}%` }"
            :title="`${m.kind === 'ckpt' ? 'Checkpoint' : 'Samples'} at step ${m.step}${m.done ? ' (done)' : ''}`"
          />
        </div>
        <span class="pl-eta mono muted">{{ etaLine }}</span>
      </div>

      <div class="layout">
        <div class="main">
          <section class="card">
            <LossChart :rows="detail.metrics" :total="total" :marks="marks" :val="detail.validation" :height="200" />
          </section>
          <section v-if="promptCount" ref="samplesCard" class="card">
            <!-- fixed header: image size (both views), the view switch and Sample now -->
            <header class="sec-head samples-head">
              <span class="sec-title">Samples</span>
              <span class="sec-sub">{{ samplesLine }}</span>
              <span class="spacer" />
              <!-- Progression only; the slot stays so the header never shifts -->
              <label class="zoom" :class="{ off: sampleView !== 'grid' }" title="Thumbnail size">
                <i class="mdi mdi-image-size-select-small" /><ZenSlider :model-value="gridSize" :min="64" :max="260" :step="4" @update:model-value="thumbPick = $event" />
              </label>
              <div class="seg" role="tablist">
                <button :class="{ on: sampleView === 'latest' }" @click="sampleView = 'latest'">Latest</button>
                <button :class="{ on: sampleView === 'grid' }" :disabled="!steps.length" title="Each prompt across training: in the viewer ←→ steps, ↑↓ prompts (or WASD)" @click="sampleView = 'grid'">Progression</button>
                <button :class="{ on: sampleView === 'manual' }" title="Try any prompt at the current step, apart from the run's samples" @click="sampleView = 'manual'">Manual<span v-if="manualCount" class="muted"> {{ manualCount }}</span><i v-if="liveManual" class="mdi mdi-circle live-dot" /></button>
              </div>
              <button
                v-if="running"
                class="chip-btn"
                :disabled="!!busy || status !== 'training' || !detail.spec.samples.prompts.length || !!sampling"
                :title="detail.spec.samples.prompts.length ? 'Render the sample prompts now' : 'This run has no sample prompts'"
                @click="send('sample')"
              >
                <i :class="sampling ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-image-refresh-outline'" /> Sample now
              </button>
            </header>
            <div v-if="sampleView === 'manual'" class="manual">
              <div class="try">
                <textarea v-model="tryPrompt" v-autosize rows="1" class="try-in" placeholder="Try a prompt at this step, e.g. [trigger] reading a book in a cafe (one per line for several)" @keydown.ctrl.enter="renderTry" />
                <div class="try-opts">
                  <ZenCombo v-model="trySize" :items="trySizes" :sorted="false" :searchable="false" size="sm" />
                  <ZenNumber v-if="(detail.spec.samples.frames ?? 1) > 1" :model-value="tryFrames ?? detail.spec.samples.frames ?? 1" :min="1" :max="361" :step="17" :precision="0" size="sm" title="Frames" @update:model-value="tryFrames = $event" />
                  <label class="base-sw small" title="Also render the same prompt and seed without the LoRA, to compare"><ZenSwitch v-model="tryBase" size="sm" /> Compare with base</label>
                  <ZenButton size="sm" variant="primary" icon="mdi mdi-creation-outline" :disabled="status !== 'training' || !tryPrompt.trim()" :title="status !== 'training' ? 'Only while it trains' : 'Queues it (Ctrl+Enter). One prompt per line queues several'" @click="renderTry">Render</ZenButton>
                </div>
              </div>
              <div v-if="liveManual" class="mlive">
                <div class="mmedia live">
                  <img v-if="liveManual.preview" :src="liveManual.preview" alt="" />
                  <i v-else class="mdi mdi-loading mdi-spin" />
                </div>
                <div class="small">
                  <b>{{ liveManual.base ? 'Without the LoRA' : `At step ${liveManual.step ?? step}` }}</b>
                  <span class="muted"> · {{ liveManual.s_total ? ((liveManual.s_step ?? 0) >= liveManual.s_total ? 'decoding' : `${liveManual.s_step ?? 0}/${liveManual.s_total}`) : 'starting' }}</span>
                  <div class="muted mcap">{{ liveManual.prompt }}</div>
                </div>
              </div>
              <div v-if="!manualGroups.length && !liveManual" class="muted small empty-manual">Prompts you try here render with the LoRA as it is now, at the step it's on. They don't join the run's samples.</div>
              <div class="mtiles">
                <div v-for="(q, qi) in queuedTries" :key="q.id" class="mtile">
                  <div class="mmedia queued">
                    <i :class="qi === 0 && liveManual ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-clock-outline'" />
                    <span class="small muted">{{ qi === 0 && liveManual ? 'Rendering' : qi === 0 ? 'Next' : `Queued · ${qi} ahead` }}</span>
                  </div>
                  <span class="mcap">{{ q.again != null ? 'Again: ' : '' }}{{ q.prompt }}</span>
                </div>
                <div v-for="(g, gi) in manualGroups" :key="g.index" class="mtile" :class="{ pair: g.base && shownStep(g) }">
                  <!-- with a base render: images as a before/after slider, videos side by side -->
                  <div v-if="g.base && shownStep(g) && !isVideo(g.base)" class="mmedia" :style="shapeOf(g.base)" @click="openManual(gi, shownStep(g))">
                    <ZenCompare :before="viewUrl(g.base)" :after="viewUrl(shownStep(g)!)" before-label="No LoRA" :after-label="`Step ${shownStep(g)!.step}`" />
                  </div>
                  <div v-else-if="g.base && shownStep(g)" class="mmedia two" :style="shapeOf(g.base, true)">
                    <video v-for="x in [g.base, shownStep(g)!]" :key="x.filename" :src="viewUrl(x)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" @click="openManual(gi, x)" />
                  </div>
                  <div v-else class="mmedia" :style="shapeOf(shownStep(g) ?? g.base)" @click="openManual(gi, shownStep(g) ?? g.base)">
                    <video v-if="isVideo((shownStep(g) ?? g.base)!)" :src="viewUrl((shownStep(g) ?? g.base)!)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
                    <img v-else :src="viewUrl((shownStep(g) ?? g.base)!)" loading="lazy" alt="" />
                  </div>
                  <div class="mfoot">
                    <span v-if="g.steps.length > 1" class="msteps">
                      <button v-for="x in g.steps" :key="x.step" :class="{ on: shownStep(g) === x }" @click="pickedStep[g.index] = x.step">{{ x.step }}</button>
                    </span>
                    <span v-else class="mono small muted">{{ g.base ? 'no LoRA / ' : '' }}step {{ shownStep(g)?.step ?? '–' }}</span>
                    <span class="spacer" />
                    <button
                      class="again"
                      :disabled="status !== 'training' || g.steps.some((x) => x.step === step) || queuedTries.some((q) => q.again === g.index)"
                      :title="g.steps.some((x) => x.step === step) ? 'Already rendered at this step' : 'The same prompt and seed with the LoRA as it is now'"
                      @click="again(g)"
                    >
                      <i class="mdi mdi-refresh" /> Again at {{ step }}
                    </button>
                  </div>
                  <span class="mcap">{{ (g.steps[0] ?? g.base)!.prompt }}</span>
                </div>
              </div>
            </div>
            <div v-else-if="sampleView === 'latest'" class="tiles" :style="{ '--cols': latestCols, '--ar': sampleAspect }">
              <div v-for="i in promptCount" :key="i" class="tile-wrap">
                <div class="tile" :class="{ live: liveIndex === i - 1 }">
                  <template v-if="liveIndex === i - 1">
                    <img v-if="sampling?.preview" :src="sampling.preview" alt="" />
                    <div class="tile-bar"><div :style="{ width: `${sampling?.s_total ? ((sampling.s_step ?? 0) / sampling.s_total) * 100 : 0}%` }" /></div>
                    <span class="tile-badge mono">{{ sampling?.s_step }}/{{ sampling?.s_total }}</span>
                  </template>
                  <button v-else-if="latestFor(i - 1)" class="tile-btn" :class="{ stale: sampling && sampling.phase === 'sampling' && (sampling.index ?? 0) < i - 1 }" @click="openLatest(i - 1)">
                    <video
                      v-if="isVideo(latestFor(i - 1)!)"
                      :src="viewUrl(latestFor(i - 1)!)"
                      muted
                      loop
                      playsinline
                      preload="metadata"
                      @mouseenter="($event.target as HTMLVideoElement).play()"
                      @mouseleave="($event.target as HTMLVideoElement).pause()"
                    />
                    <i v-if="isVideo(latestFor(i - 1)!)" class="mdi mdi-play play-badge" />
                    <img v-else :src="viewUrl(latestFor(i - 1)!)" alt="" />
                  </button>
                  <div v-else class="tile-empty"><i class="mdi mdi-image-outline" /></div>
                  <span class="tile-num mono">{{ i }}</span>
                </div>
                <div class="tile-cap" :title="promptText(i - 1)">{{ promptText(i - 1) }}</div>
              </div>
            </div>
            <ScrollArea v-else ref="gridScroll" axis="x" class="matrix-scroll">
              <div class="matrix" :style="{ '--t': `${gridSize}px` }">
                <div class="mrow mhead">
                  <div
                    v-for="s in steps"
                    :key="s"
                    class="mstep mono"
                    :class="{ manual: isManual(s) }"
                    :title="isManual(s) ? 'Sampled on demand (Sample now)' : undefined"
                  >
                    {{ s === 0 ? 'base' : s }}<i v-if="isManual(s)" class="mdi mdi-gesture-tap" />
                  </div>
                </div>
                <template v-for="[index, prompt] in prompts" :key="index">
                  <div class="mlabel" :title="prompt"><span class="pnum">{{ index + 1 }}</span><span class="ellipsis">{{ prompt }}</span></div>
                  <div class="mrow">
                    <div v-for="s in steps" :key="s" class="mcell">
                      <button v-if="cell(index, s)" @click="openGrid(index, s)">
                        <video
                          v-if="isVideo(cell(index, s)!)"
                          :src="viewUrl(cell(index, s)!)"
                          muted
                          loop
                          playsinline
                          preload="metadata"
                          @mouseenter="($event.target as HTMLVideoElement).play()"
                          @mouseleave="($event.target as HTMLVideoElement).pause()"
                        />
                        <img v-else :src="viewUrl(cell(index, s)!)" loading="lazy" :alt="prompt" />
                      </button>
                    </div>
                  </div>
                </template>
              </div>
            </ScrollArea>
          </section>
        </div>
        <aside class="rail">
          <section class="card">
            <div v-for="g in statGroups" :key="g.title" class="sgroup">
              <div class="sg-title">{{ g.title }}</div>
              <dl class="sgrid">
                <div v-for="it in g.items" :key="it.k" class="scell">
                  <dt>{{ it.k }}</dt>
                  <dd :class="{ mono: it.mono !== false }">
                    {{ it.v }}<span v-if="it.delta" class="delta" :class="it.delta.good ? 'good' : 'bad'">{{ it.delta.text }}</span>
                  </dd>
                </div>
              </dl>
            </div>
            <button class="link spec-link" @click="specOpen = true"><i class="mdi mdi-code-json" /> Full spec</button>
          </section>
          <section v-if="runLoaded.length" class="card loaded-card">
            <div class="eyebrow">Loaded now</div>
            <div v-for="m in runLoaded" :key="m.file ?? m.name" class="lrow" :title="`${m.file}: ${gb(m.vram)} of ${gb(m.bytes)} GB in VRAM`">
              <span class="lname ellipsis">{{ (m.file ?? m.name).replace(/\.safetensors$/, '') }}</span>
              <span class="mono small muted">{{ gb(m.vram) }}/{{ gb(m.bytes) }} GB</span>
              <div class="lbar"><span :style="{ width: `${m.bytes ? (m.vram / m.bytes) * 100 : 0}%` }" /></div>
            </div>
          </section>

          <section class="card">
            <header class="sec-head">
              <span class="sec-title">Checkpoints</span>
              <span class="sec-sub">{{ saveLine }}</span>
            </header>
            <div v-if="!detail.checkpoints.length" class="muted small">
              {{ running ? `The first save lands at step ${nextSave?.step ?? total}.` : 'None saved.' }}
            </div>
            <div class="ckpts">
              <div v-for="c in [...detail.checkpoints].reverse()" :key="c.path" class="ckpt" :class="{ best: c.step === bestStep }" :title="c.path">
                <div class="ck-main">
                  <span class="ck-step mono">{{ c.step }}</span>
                  <span v-if="c.final" class="chip">final</span>
                  <span v-if="c.step === bestStep" class="chip best-chip" title="Lowest validation loss of the saved checkpoints">best</span>
                  <span class="spacer" />
                  <span class="ck-size mono" :class="{ gone: c.size === null }">{{ [ckptAt(c.step), c.size ? fmtBytes(c.size) : c.size === null ? 'file missing' : ''].filter(Boolean).join(' · ') }}</span>
                </div>
                <div class="ck-acts">
                  <span class="ck-sub mono">{{ ckptVal(c.step) != null ? `val ${ckptVal(c.step)!.toFixed(4)}` : 'no val' }}</span>
                  <span class="spacer" />
                  <RouterLink class="mini-btn" :to="`/apps/test-lora?run=${id}&lora=${encodeURIComponent(c.lora_name)}`" title="Try it in Test LoRA, on this run's model">Test</RouterLink>
                  <button v-if="resumable && !c.final" class="mini-btn" title="Resume this run from here, optimizer state and all" @click="resume(c.path)">Resume</button>
                  <RouterLink v-else class="mini-btn" :to="`/runs/new?from=${id}&init=${encodeURIComponent(c.lora_name)}`" title="A new run with these settings that starts from this LoRA">Train more</RouterLink>
                  <button class="ib" :title="copied === c.lora_name ? 'Copied' : `Copy LoRA name: ${c.lora_name}`" @click="copy(c.lora_name)">
                    <i :class="copied === c.lora_name ? 'mdi mdi-check' : 'mdi mdi-content-copy'" />
                  </button>
                </div>
              </div>
            </div>
          </section>
        </aside>
      </div>

      <ZenModal v-model:open="specOpen" title="Spec" width="760px">
        <pre class="mono spec">{{ JSON.stringify(detail.spec, null, 2) }}</pre>
      </ZenModal>

      <ZenLightbox v-if="lb" v-model:index="lb.index" :items="lb.items" :grid="lb.grid" @close="lb = null" />
    </template>
  </div>
</template>

<style scoped>
.wide {
  max-width: 1320px;
}
.head {
  display: flex;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
}
.titles {
  flex: 1;
  min-width: 0;
}
.title-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.title-row h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 650;
  letter-spacing: -0.01em;
}
.lede {
  margin: 4px 0 0;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}
.lr-wrap {
  position: relative;
}
.lr-pop {
  position: absolute;
  right: 0;
  top: calc(100% + 6px);
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 200px;
  padding: 12px;
  box-shadow: 0 10px 30px rgb(0 0 0 / 0.3);
  background: var(--zen-surface);
}
.err {
  border-color: var(--zen-danger);
  font-size: 12px;
}
.msg {
  font-size: 12px;
  white-space: pre-wrap;
}
.small {
  font-size: 11px;
}
.spacer {
  flex: 1;
}
.loaded-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.lrow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 3px 8px;
  font-size: 12px;
}
.lbar {
  grid-column: span 2;
  height: 3px;
  border-radius: 2px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.lbar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.lname {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.phase .pl-step {
  color: var(--zen-accent);
  font-weight: 600;
}
.ph-mid {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 3px;
  height: 18px; /* the step bar's height: switching between them doesn't move the page */
  min-width: 0;
}
.ph-text {
  font-size: 11px;
  color: var(--zen-muted);
  line-height: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar.thin {
  height: 3px;
}
.progline {
  display: grid;
  grid-template-columns: 11ch minmax(0, 1fr) 20ch;
  align-items: center;
  gap: 16px;
  padding: 12px 18px;
  font-variant-numeric: tabular-nums;
}
.pl-step {
  font-size: 13px;
  white-space: nowrap;
}
.pl-eta {
  font-size: 12px;
  text-align: right;
  white-space: nowrap;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 14px;
  align-items: start;
}
.main,
.rail {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}
.rail {
  position: sticky;
  top: 16px;
}
.sec-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 12px;
  min-height: 20px;
}
.sec-title {
  font-weight: 650;
}
.sec-sub {
  font-size: 11px;
  color: var(--zen-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.track {
  position: relative;
  padding: 5px 0;
}
.bar {
  height: 6px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: var(--zen-accent);
  transition: width 0.3s ease;
}
.notch {
  position: absolute;
  top: 1px;
  width: 2px;
  height: 14px;
  margin-left: -1px;
  border-radius: var(--r-sm);
  background: var(--zen-muted);
  opacity: 0.5;
}
.notch.sample {
  top: 4px;
  width: 8px;
  height: 8px;
  margin-left: -4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface);
  border: 1.5px solid var(--zen-muted);
}
/* done marks in the text colour: the accent would vanish into the filled bar */
.notch.ckpt.done {
  background: var(--zen-text);
  opacity: 0.85;
}
.notch.sample.done {
  background: var(--zen-text);
  border-color: var(--zen-text);
  opacity: 0.85;
}
.sgroup + .sgroup {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.sg-title {
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted);
  margin-bottom: 8px;
}
.sgrid {
  margin: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 14px;
}
.scell {
  min-width: 0;
}
.scell dt {
  font-size: 11px;
}
.scell dd {
  margin: 2px 0 0;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.delta {
  font-size: 11px;
}
dt {
  font-size: 12px;
  color: var(--zen-muted);
  white-space: nowrap;
}
dd {
  margin: 0;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.good {
  color: var(--zen-ok);
  font-size: 11px;
}
.bad {
  color: var(--zen-warn);
  font-size: 11px;
}
/* tile size follows the column width only, never what's inside: no reflow when renders land */
.tiles {
  display: grid;
  grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr));
  gap: 12px;
}
.tile-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.tile {
  position: relative;
  aspect-ratio: var(--ar, 1);
  border-radius: var(--r-lg);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.tile.live {
  box-shadow: 0 0 0 2px var(--zen-accent);
}
.tile img,
.tile video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.manual {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.try {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.try-in {
  display: block;
  width: 100%;
  resize: none;
  overflow-y: auto;
  font: inherit;
  font-size: 13px;
  line-height: 1.5;
  padding: 8px 10px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
  max-height: 10em;
}
.try-in:focus {
  outline: none;
  border-color: var(--zen-accent);
}
.try-opts {
  display: flex;
  gap: 8px;
  align-items: center;
}
.try-opts > :first-child {
  width: 200px;
}
.empty-manual {
  padding: 8px 0;
}
.mtiles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}
.mtile {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.mtile.pair {
  grid-column: span 2;
}
.mmedia {
  aspect-ratio: 1;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
}
.mmedia.two {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2px;
}
.mmedia img,
.mmedia video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.mmedia :deep(.zen-compare img) {
  object-fit: contain;
}
.mlive {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 12px;
  align-items: center;
  padding: 8px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 50%, transparent);
}
.mmedia.queued {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font-size: 20px;
  cursor: default;
}
.mmedia.live {
  display: grid;
  place-items: center;
  color: var(--zen-muted);
}
.mfoot {
  display: flex;
  align-items: center;
  gap: 6px;
}
.msteps {
  display: flex;
  gap: 3px;
  flex-wrap: wrap;
}
.msteps button,
.again {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  padding: 1px 7px;
  border-radius: var(--r-sm);
  cursor: pointer;
}
.msteps button.on {
  color: var(--zen-text);
  border-color: var(--zen-accent);
}
.again:hover:not(:disabled) {
  color: var(--zen-text);
  border-color: var(--zen-accent);
}
.again:disabled {
  opacity: 0.4;
  cursor: default;
}
.live-dot {
  color: var(--zen-accent);
  font-size: 8px;
  margin-left: 4px;
  vertical-align: middle;
}
.base-sw {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
  white-space: nowrap;
}
.mcap {
  font-size: 11px;
  color: var(--zen-muted);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.play-badge {
  position: absolute;
  right: 8px;
  bottom: 8px;
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  font-size: 14px;
  pointer-events: none;
  transition: opacity var(--zen-dur-fast);
}
.tile-btn:hover .play-badge {
  opacity: 0;
}
.tile-btn {
  position: absolute;
  inset: 0;
  padding: 0;
  border: none;
  background: none;
  cursor: zoom-in;
}
.tile-btn.stale img {
  opacity: 0.45;
}
.tile-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: 26px;
  color: var(--zen-muted);
}
.tile-bar {
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: 8px;
  height: 4px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.35);
  overflow: hidden;
}
.tile-bar div {
  height: 100%;
  background: var(--zen-accent);
  transition: width 0.25s ease;
}
.tile-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
}
.tile-cap {
  font-size: 11px;
  line-height: 1.4;
  color: var(--zen-muted);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 2.8em; /* two lines, so tiles line up */
}
.tile-num {
  position: absolute;
  top: 6px;
  left: 6px;
  min-width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  font-size: 10px;
  pointer-events: none;
}
.samples-head {
  align-items: center;
  min-height: 30px;
}
/* the view switch and Sample now share one height */
.samples-head .seg,
.samples-head .chip-btn {
  height: 30px;
  box-sizing: border-box;
}
.samples-head .seg button {
  height: 100%;
  padding-block: 0;
}
.samples-head .chip-btn {
  padding: 0 12px;
  border-radius: var(--r-lg);
}
.seg {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button,
.chip-btn {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}
.chip-btn {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.seg button:hover:not(:disabled),
.chip-btn:hover:not(:disabled) {
  color: var(--zen-text);
}
.seg button:disabled,
.chip-btn:disabled {
  opacity: 0.45;
  cursor: default;
}
.zoom {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
}
.zoom .zen-slider {
  width: 110px;
}
.matrix-scroll {
  height: auto;
}
.matrix {
  padding-bottom: 12px;
}
.mrow {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
  width: max-content;
}
.pnum {
  display: inline-grid;
  place-items: center;
  min-width: 18px;
  height: 18px;
  margin-right: 4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  color: var(--zen-text);
  font-family: var(--zen-mono);
  font-size: 10px;
}
.mlabel {
  position: sticky;
  left: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  width: max-content;
  max-width: min(100%, 720px);
  margin: 8px 0 6px;
  font-size: 12px;
  color: var(--zen-muted);
}
.mlabel .ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mstep.manual {
  color: var(--zen-text);
}
.mstep.manual i {
  margin-left: 3px;
  font-size: 11px;
  color: var(--zen-accent);
}
.zoom.off {
  visibility: hidden;
}
.mstep {
  width: var(--t);
  flex: none;
  font-size: 11px;
  color: var(--zen-muted);
  text-align: center;
}
.mcell {
  width: var(--t);
  height: var(--t);
  flex: none;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.mcell img,
.mcell video {
  width: var(--t);
  height: var(--t);
  object-fit: contain;
  display: block;
  transition: transform 0.15s ease;
}
.mcell button {
  display: block;
  padding: 0;
  border: none;
  background: none;
  cursor: zoom-in;
}
.mcell button:hover img {
  transform: scale(1.04);
}
.ckpts {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ckpt {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 8px;
  margin: 0 -8px;
  border-radius: var(--r-md);
  font-size: 12px;
}
.ckpt:hover {
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.ckpt.best {
  background: color-mix(in srgb, var(--zen-ok) 8%, transparent);
}
.ck-main {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 26px;
}
.ck-step {
  font-weight: 650;
  font-size: 13px;
  min-width: 5ch;
}
.ck-sub {
  font-size: 11px;
  color: var(--zen-muted);
}
.ck-size {
  font-size: 11px;
  color: var(--zen-muted);
}
.ck-size.gone {
  color: var(--zen-warn);
}
.ck-acts {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 2px;
}
.mini-btn {
  height: 24px;
  padding: 0 8px;
  display: inline-flex;
  align-items: center;
  border: 1px solid color-mix(in srgb, var(--zen-border) 80%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 11px;
  text-decoration: none;
  cursor: pointer;
  white-space: nowrap;
}
.mini-btn:hover {
  border-color: var(--zen-accent);
}
.best-chip {
  color: var(--zen-ok);
}
.ib {
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.ib:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.spec-link {
  margin-top: 12px;
  padding: 0;
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  white-space: nowrap;
}
.link:hover {
  color: var(--zen-accent);
}
.spec {
  margin: 0;
  font-size: 12px;
  white-space: pre-wrap;
}
@media (max-width: 1000px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .rail {
    position: static;
  }
  .progline {
    grid-template-columns: auto minmax(0, 1fr);
  }
  .pl-eta {
    grid-column: 1 / -1;
    text-align: left;
  }
}
</style>
