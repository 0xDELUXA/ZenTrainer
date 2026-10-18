<script setup lang="ts">
// One run. Header with the controls; progress beside the latest samples; the loss and
// validation curves; then samples as a prompt × step grid, so a prompt's row reads as
// its training progression (click a cell to page through that row in the lightbox).
import { ZenButton, ZenCombo, ZenLightbox, ZenModal, ZenNumber, ZenSlider, ZenSwitch, ZenTip } from '@nynxz/zenkit-ui'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { playTogether } from '../videoSync'
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
  type ValidationRow, type LikenessRow,
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
    } else if (m.type === EVENTS.likeness) {
      ;(d.likeness ??= []).push(m.data as unknown as LikenessRow)
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
// queued to sample a finished run's checkpoint isn't training (no Pause or Stop)
const queuedToSample = computed(() => status.value === 'queued' && !!detail.value?.run.message?.startsWith('sampling'))
const running = computed(() => !!status.value && ACTIVE.includes(status.value) && !queuedToSample.value)
// finished, with a checkpoint file: Sample now and Manual tries run as a short job on its newest
// checkpoint instead (engine: POST .../sample), and the results land at that step
const sampleCkpt = computed(() => (running.value || sampleBusy.value ? null : (ckRows.value.find((c) => c.size !== null) ?? null)))
const sampleBusy = computed(() => status.value === 'sampling' || queuedToSample.value)
// a finished run's newest checkpoint with its file still there
const newestCkpt = computed(() => (running.value ? null : (ckRows.value.find((c) => c.size !== null) ?? null)))
async function sampleFinished(tries?: Record<string, unknown>[]) {
  if (!sampleCkpt.value) return
  try {
    const r = await engine.sampleRun(props.id, tries ? { checkpoint: sampleCkpt.value.path, run_prompts: false, tries } : { checkpoint: sampleCkpt.value.path })
    if (detail.value) detail.value.run.status = r.run.status
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  }
}
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
async function again(g: { index: number; base: SampleRow | null; steps: SampleRow[] }) {
  const x = g.steps.at(-1) ?? g.base
  if (!x) return
  sampleView.value = 'manual'
  await queueTry({ prompt: x.prompt, width: x.width, height: x.height, frames: x.frames, seed: x.seed, base: false, index: g.index }, x.prompt, g.index)
}

// manual tries waiting their turn: the engine runs sample commands one after another at step
// boundaries, so any number can be queued; each leaves this list when its render lands
const queuedTries = ref<{ id: number; prompt: string; again?: number }[]>([])
// the cards for tries not landed yet: the first one is live while a manual render runs (even
// one this page didn't queue, e.g. after a reload)
// a prompt as samples render it: [trigger] substituted (engine: callbacks.sample_text), so a waiting
// try can find the row its first render (the no-LoRA one) already started
const asRendered = (p: string) => {
  const t = detail.value?.spec.dataset.trigger ?? ''
  return p.replaceAll('[trigger]', t).trim()
}
const waitingCards = computed(() => {
  const live = liveManual.value
  const phase = live
    ? `${live.base ? 'Without the LoRA' : `At step ${live.step ?? step.value}`} · ${
        live.s_total ? ((live.s_step ?? 0) >= live.s_total ? 'decoding' : `${live.s_step ?? 0}/${live.s_total}`) : 'starting'
      }`
    : ''
  const frac = live?.s_total ? Math.min(1, (live.s_step ?? 0) / live.s_total) : 0
  const cards = queuedTries.value.map((q, i) => ({
    key: `q${q.id}`,
    prompt: asRendered(q.prompt),
    again: q.again != null,
    // a new try whose no-LoRA render landed waits in that row, not in a row of its own
    againIndex: q.again ?? manualGroups.value.find((g) => !g.steps.length && rowOf(g).prompt === asRendered(q.prompt))?.index,
    live: i === 0 && !!live,
    // a finished run's job loads the whole model before its first render
    loading: i === 0 && !live && sampleBusy.value,
    frac: i === 0 ? frac : 0,
    state: i === 0 && live ? phase : i === 0 && sampleBusy.value ? 'Loading the model…' : i === (live ? 1 : 0) ? 'Next' : `Queued · ${i - (live ? 1 : 0)} ahead`,
  }))
  if (live && !cards.length)
    cards.push({ key: 'live', prompt: live.prompt ?? '', again: false, againIndex: undefined, live: true, loading: false, frac, state: phase })
  return cards
})
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
  if (sampleCkpt.value) {
    // finished: one job renders them all on the newest checkpoint
    for (const prompt of lines) queuedTries.value = [...queuedTries.value, { id: ++tryIds, prompt }]
    await sampleFinished(lines.map((prompt) => ({ prompt, width: w, height: h, frames: tryFrames.value ?? undefined, base: tryBase.value })))
    return
  }
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
  /** what it means, behind an ⓘ */
  tip?: string
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
        { k: 'Training', v: dur(l?.train_time), tip: 'Time spent in training steps only: no loading, sampling or validation' },
        { k: 'Total', v: dur(wall.value), tip: 'From start to now, with loading, sampling and validation' }, // start to now, loading and sampling included
        { k: 'Speed', v: `${l?.it_s?.toFixed(2) ?? '–'} it/s`, tip: 'Training steps per second, lately. The first steps are slower (caching, compiling)' },
        { k: 'Epoch', v: String(l?.epoch ?? '–'), tip: 'Passes through the dataset so far, repeats included' },
        ...(l?.vram_peak ? [{ k: 'Peak VRAM', v: `${gb(l.vram_peak)} GB`, tip: 'The most GPU memory a training step has used. Sampling can use more, briefly' }] : []),
        ...(l?.phases
          ? [
              { k: 'Forward / back', v: `${l.phases.forward.toFixed(1)} / ${l.phases.backward.toFixed(1)} s`, tip: 'Seconds per step running the model forward, and back for the gradients: nearly all of a step' },
              { k: 'Data / optimizer', v: `${l.phases.data.toFixed(2)} / ${l.phases.optimizer.toFixed(2)} s`, tip: 'Seconds per step fetching the batch, and updating the LoRA. Near 0 is right' },
            ]
          : []),
      ],
    },
    {
      title: 'Learning',
      items: [
        { k: 'Learning rate', v: fmtLr(l?.lr), tip: "The step size now. Automagic and Prodigy set their own as they go; others follow the schedule" },
        { k: 'Grad norm', v: l?.grad_norm?.toFixed(3) ?? '–', tip: 'How strongly this step pulled on the LoRA (before clipping). Spikes are hard pictures; a steady climb can mean it’s going unstable' },
        {
          k: 'Validation',
          v: valLast.value ? valLast.value.val_loss.toFixed(4) : '–',
          delta: valDelta,
          tip: 'Loss on a fixed set: the same pictures at the same noise, every time, so it moves only when the LoRA does (the training loss jumps with each step’s random noise). The arrow: change since before training. It’s drawn from the training pictures, so it shows learning, not overfitting: judge that by the samples',
        },
        { k: 'Loss', v: l?.loss?.toFixed(4) ?? '–', tip: 'This step’s loss. Noisy: each step picks a random picture and noise level. The chart’s smoothed line is the trend' },
        ...(likeLast.value
          ? [{
              k: 'Likeness',
              v: `${likeLast.value.score.toFixed(2)} @ ${likeLast.value.step}`,
              tip: `How much the samples' faces look like the dataset's (mean of the step's samples): 1 is identical, about 0.5 and up reads as the same person, near 0 a stranger. ${likeFirst.value && likeFirst.value.step !== likeLast.value.step ? `It was ${likeFirst.value.score.toFixed(2)} at step ${likeFirst.value.step}.` : ''}`,
            }]
          : []),
      ],
    },
  ]
})
// what the run is, as label/value lines that wrap rather than cut off
const OPT_NAMES: Record<string, string> = { Automagic3: 'Automagic v3', AdamWScheduleFree: 'AdamW schedule-free', AdamW8bit: 'AdamW 8-bit' }
const ADAPTIVE = new Set(['Automagic', 'Automagic3', 'Prodigy']) // they set their own LR: the schedule hardly matters
const setup = computed((): { k: string; v: string; title?: string }[] => {
  const sp = detail.value?.spec
  if (!sp) return []
  const model = sp.model.unet_name ?? sp.model.ckpt_name ?? ''
  const m = sp.memory
  const mem = [
    m.offloading ? 'offload' : '',
    m.int8_forward !== false && /int8/i.test(model) ? 'int8 fwd' : '',
    m.w8a8_backward ? 'W8A8' : m.fused_int8_backward ? 'fused int8' : '',
    m.compile_blocks ? 'compile' : '',
    m.gradient_checkpointing ? (m.checkpoint_skip ? `checkpoint · keep ${m.checkpoint_skip}` : 'checkpoint') : '',
    m.tread ? `TREAD ${Math.round(m.tread * 100)}%` : '',
  ].filter(Boolean)
  const opt = sp.optim
  return [
    { k: 'Model', v: model.replace(/\.safetensors$/, ''), title: model },
    {
      k: 'Dataset',
      v: [sp.dataset.folder, sp.dataset.caption_set && sp.dataset.caption_set !== 'default' ? sp.dataset.caption_set : '', sp.dataset.trigger ? `“${sp.dataset.trigger}”` : ''].filter(Boolean).join(' · '),
    },
    { k: 'Training', v: `${opt.steps} steps · batch ${opt.batch_size} · ${sp.dataset.resolution}px` },
    {
      k: 'Optimizer',
      v: [OPT_NAMES[opt.optimizer] ?? opt.optimizer, `from ${fmtLr(opt.learning_rate)}`, ADAPTIVE.has(opt.optimizer) ? '' : opt.scheduler].filter(Boolean).join(' · '),
    },
    { k: 'Network', v: `${sp.network.algorithm} r${sp.network.rank} · ${Array.isArray(sp.network.targets) ? `${sp.network.targets.length} layer types` : sp.network.targets.replace('_', ' + ')}${sp.dataset.masked_loss ? ' · masked' : ''}` },
    ...(mem.length ? [{ k: 'Memory', v: mem.join(' · ') }] : []),
  ]
})
// faces extra: each sampled step's mean likeness (run samples and Sample now; manual tries left out)
const likeSteps = computed(() => {
  const by = new Map<number, number[]>()
  for (const r of detail.value?.likeness ?? []) if (!r.manual && r.score != null) by.set(r.step, [...(by.get(r.step) ?? []), r.score])
  return [...by].sort((a, b) => a[0] - b[0]).map(([step, xs]) => ({ step, score: xs.reduce((a, b) => a + b, 0) / xs.length }))
})
const likeLast = computed(() => likeSteps.value.at(-1) ?? null)
const likeFirst = computed(() => likeSteps.value[0] ?? null)
// validation loss at a checkpoint: the latest evaluation at or before its step
const ckptVal = (stepAt: number) => {
  let v: number | undefined
  for (const r of detail.value?.validation ?? []) if (r.step <= stepAt) v = r.val_loss
  return v ?? null
}
// faces extra: the run's samples' likeness at a step (their mean; manual tries left out)
const likeAt = (stepAt: number): number | null => {
  const xs = (detail.value?.likeness ?? []).filter((r) => r.step === stepAt && !r.manual && r.score != null).map((r) => r.score!)
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
}
// the best checkpoint: highest likeness when faces are scored (a character LoRA), else lowest validation loss
const bestBy = computed(() => {
  const cks = detail.value?.checkpoints ?? []
  return cks.filter((c) => likeAt(c.step) != null).length >= 2 ? 'likeness' : 'val'
})
const bestStep = computed(() => {
  const cks = (detail.value?.checkpoints ?? []).filter((c) => (bestBy.value === 'likeness' ? likeAt(c.step) : ckptVal(c.step)) != null)
  if (cks.length < 2) return null
  if (bestBy.value === 'likeness') return cks.reduce((a, b) => (likeAt(b.step)! > likeAt(a.step)! ? b : a)).step
  return cks.reduce((a, b) => (ckptVal(b.step)! < ckptVal(a.step)! ? b : a)).step
})
// why a checkpoint was saved, from its name: the schedule (null), by hand, or as the run stopped
const ckKind = (c: { lora_name: string }) => /_(stopped|error|manual)-step\d+/.exec(c.lora_name)?.[1] ?? null
// one row per step, newest first: a run stopped right after a save (or resumed) can hold several
// files of one step, so the final, then the scheduled save, then a manual one, then a stop's stands
const ckRank = (c: { final?: boolean; lora_name: string }) => (c.final ? 0 : { null: 1, manual: 2, stopped: 3, error: 4 }[String(ckKind(c))] ?? 5)
const ckRows = computed(() => {
  const by = new Map<number, NonNullable<typeof detail.value>['checkpoints'][number]>()
  for (const c of detail.value?.checkpoints ?? []) {
    const had = by.get(c.step)
    if (!had || ckRank(c) < ckRank(had)) by.set(c.step, c)
  }
  return [...by.values()].sort((a, b) => b.step - a.step)
})
// every checkpoint of a run is the same size: said once, in the header
const ckSize = computed(() => {
  const sizes = new Set(ckRows.value.map((c) => c.size).filter((v): v is number => !!v))
  return sizes.size ? `${fmtBytes(Math.max(...sizes))} each` : ''
})
const saveLine = computed(() => {
  const n = ckRows.value.length
  const next = nextSave.value
  const plan = detail.value?.spec.save_every ? `every ${detail.value.spec.save_every}` : 'final only'
  if (running.value && next) return `${n ? `${n} · ` : ''}next ${next.step} · ${next.in}`
  return [n ? `${n} · ${plan}` : plan, ckSize.value].filter(Boolean).join(' · ')
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
const rowOf = (g: { base: SampleRow | null; steps: SampleRow[] }) => (g.steps[0] ?? g.base)!
const copyText = (t: string) => navigator.clipboard?.writeText(t).catch(() => {})
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
// a Manual tile: 96px high, but a portrait render grows taller so the tile stays at least
// 80px wide and its "Step N" label fits
const tileOf = (x: SampleRow | null | undefined) => {
  const shape = shapeOf(x)
  if (!x?.width || !x?.height) return shape
  const r = x.width / x.height
  return 96 * r < 80 ? { ...shape, height: `${Math.min(170, Math.round(80 / r))}px` } : shape
}
// a row's videos playing together (Progression and Manual): one row at a time
const syncing = ref<string | null>(null)
let stopSync: (() => void) | null = null
function toggleSync(key: string, row: Element | null | undefined) {
  stopSync?.()
  stopSync = null
  if (syncing.value === key || !row) {
    syncing.value = null
    return
  }
  syncing.value = key
  stopSync = playTogether([...row.querySelectorAll('video')])
}
onUnmounted(() => stopSync?.())
const inSync = (v: HTMLVideoElement) => !!syncing.value && !!v.closest('[data-syncing]')
const hoverPlay = (e: Event) => {
  const v = e.target as HTMLVideoElement
  if (!inSync(v)) void v.play().catch(() => {})
}
const hoverPause = (e: Event) => {
  const v = e.target as HTMLVideoElement
  if (!inSync(v)) v.pause()
}
const rowHasVideo = (xs: (SampleRow | null | undefined)[]) => xs.some((x) => x && isVideo(x))
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
          <RouterLink v-else-if="resumable" :to="`/runs/new?edit=${id}`"><ZenButton icon="mdi mdi-tune-variant" title="Change memory, samples, learning rate or length, then resume from its last state">Edit &amp; resume</ZenButton></RouterLink>
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
            <LossChart :rows="detail.metrics" :total="total" :marks="marks" :val="detail.validation" :like="likeSteps" :height="200" />
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
                <button :class="{ on: sampleView === 'manual' }" title="Try any prompt at the current step, apart from the run's samples" @click="sampleView = 'manual'">Manual<span v-if="manualCount" class="muted"> {{ manualCount }}</span><i class="mdi mdi-circle live-dot" :class="{ off: !liveManual }" /></button>
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
              <button
                v-else-if="newestCkpt && !steps.includes(newestCkpt.step)"
                class="chip-btn"
                :disabled="sampleBusy || !detail.spec.samples.prompts.length"
                :title="sampleCkpt ? `Render the sample prompts with the step ${sampleCkpt.step} checkpoint: they join the samples and the likeness line (replacing any already at that step)` : 'Sampling'"
                @click="sampleFinished()"
              >
                <i :class="sampleBusy || sampling ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-image-refresh-outline'" /> {{ sampleBusy ? 'Sampling…' : `Sample step ${newestCkpt.step}` }}
              </button>
            </header>
            <div v-if="sampleView === 'manual'" class="manual">
              <div class="try">
                <textarea
                  v-model="tryPrompt"
                  v-autosize
                  rows="1"
                  class="try-in"
                  :placeholder="`Try a prompt ${running ? 'at this step' : newestCkpt ? `on the step ${newestCkpt.step} checkpoint` : ''}, e.g. [trigger] reading a book in a cafe (one per line for several)`"
                  @keydown.ctrl.enter="renderTry"
                />
                <div class="try-opts">
                  <ZenCombo v-model="trySize" :items="trySizes" :sorted="false" :searchable="false" size="sm" />
                  <ZenNumber v-if="(detail.spec.samples.frames ?? 1) > 1" :model-value="tryFrames ?? detail.spec.samples.frames ?? 1" :min="1" :max="361" :step="17" :precision="0" size="sm" title="Frames" @update:model-value="tryFrames = $event" />
                  <label class="base-sw small" title="Also render the same prompt and seed without the LoRA, to compare"><ZenSwitch v-model="tryBase" size="sm" /> Compare with base</label>
                  <ZenButton size="sm" variant="primary" icon="mdi mdi-creation-outline" :disabled="(status !== 'training' && !sampleCkpt) || !tryPrompt.trim()" :title="status === 'training' ? 'Queues it (Ctrl+Enter). One prompt per line queues several' : sampleCkpt ? `Renders with the step ${sampleCkpt.step} checkpoint (Ctrl+Enter). One prompt per line for several` : 'Only while it trains, or once it has a checkpoint'" @click="renderTry">Render</ZenButton>
                </div>
              </div>
              <div v-if="!manualGroups.length && !waitingCards.length" class="muted small empty-manual">
                Prompts you try here render with {{ running ? "the LoRA as it is now, at the step it's on" : 'the newest checkpoint' }}. They don't join the run's samples.
              </div>
              <!-- one row per try: its prompt, then its renders across (no LoRA, then each step) -->
              <div class="mrows">
                <div v-for="t in waitingCards.filter((w) => w.againIndex == null)" :key="t.key" class="mrow2">
                  <div class="mprompt"><span class="mtext">{{ t.prompt }}</span></div>
                  <div class="mcells">
                    <div class="tcell pending" :class="{ live: t.live }">
                      <img v-if="t.live && liveManual?.preview" :src="liveManual.preview" alt="" />
                      <i v-else :class="t.live || t.loading ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-clock-outline'" />
                      <div v-if="t.live" class="qbar"><span :style="{ width: `${t.frac * 100}%` }" /></div>
                      <span class="clabel">{{ t.state }}</span>
                    </div>
                  </div>
                </div>
                <div v-for="(g, gi) in manualGroups" :key="g.index" class="mrow2" :data-syncing="syncing === `m${g.index}` || undefined">
                  <div class="mprompt">
                    <span class="mtext" :title="rowOf(g).prompt">{{ rowOf(g).prompt }}</span>
                    <span class="mmeta mono">
                      {{ rowOf(g).width }}×{{ rowOf(g).height }}<template v-if="(rowOf(g).frames ?? 1) > 1"> · {{ rowOf(g).frames }}f</template> · seed {{ rowOf(g).seed }}
                    </span>
                    <span class="macts" :class="{ held: syncing === `m${g.index}` }">
                      <button
                        v-if="rowHasVideo([g.base, ...g.steps])"
                        :class="{ on: syncing === `m${g.index}` }"
                        :title="syncing === `m${g.index}` ? 'Stop' : 'Play the whole row together'"
                        @click="toggleSync(`m${g.index}`, ($event.currentTarget as HTMLElement).closest('.mrow2')?.querySelector('.mcells'))"
                      >
                        <i :class="syncing === `m${g.index}` ? 'mdi mdi-stop' : 'mdi mdi-play'" />
                      </button>
                      <button title="Put this prompt back in the box to change it" @click="tryPrompt = rowOf(g).prompt"><i class="mdi mdi-pencil-outline" /></button>
                      <button title="Copy the prompt" @click="copyText(rowOf(g).prompt)"><i class="mdi mdi-content-copy" /></button>
                    </span>
                  </div>
                  <div class="mcells">
                    <button v-for="x in [...(g.base ? [g.base] : []), ...g.steps]" :key="x.filename" class="tcell" :style="tileOf(x)" @click="openManual(gi, x)">
                      <video v-if="isVideo(x)" :src="viewUrl(x)" muted loop playsinline preload="metadata" @mouseenter="hoverPlay" @mouseleave="hoverPause" />
                      <img v-else :src="viewUrl(x)" loading="lazy" alt="" />
                      <span class="clabel">{{ x.base ? 'No LoRA' : `Step ${x.step}` }}</span>
                    </button>
                    <div v-for="t in waitingCards.filter((w) => w.againIndex === g.index)" :key="t.key" class="tcell pending" :class="{ live: t.live }" :style="tileOf(g.steps.at(-1) ?? g.base)">
                      <img v-if="t.live && liveManual?.preview" :src="liveManual.preview" alt="" />
                      <i v-else :class="t.live ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-clock-outline'" />
                      <div v-if="t.live" class="qbar"><span :style="{ width: `${t.frac * 100}%` }" /></div>
                      <span class="clabel">{{ t.state }}</span>
                    </div>
                    <!-- the next render of this prompt: same seed, the LoRA as it is now -->
                    <button
                      v-if="status === 'training' && !g.steps.some((x) => x.step === step) && !queuedTries.some((q) => q.again === g.index)"
                      class="tcell add"
                      :style="tileOf(g.steps.at(-1) ?? g.base)"
                      title="Render this prompt again (same seed) with the LoRA as it is now"
                      @click="again(g)"
                    >
                      <i class="mdi mdi-plus" />
                      <span>Step {{ step }}</span>
                    </button>
                  </div>
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
                      @mouseenter="hoverPlay"
                      @mouseleave="hoverPause"
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
                  <div class="mlabel" :title="prompt">
                    <span class="pnum">{{ index + 1 }}</span>
                    <button
                      v-if="rowHasVideo(steps.map((s) => cell(index, s)))"
                      class="rowplay"
                      :class="{ on: syncing === `p${index}` }"
                      :title="syncing === `p${index}` ? 'Stop' : 'Play the whole row together'"
                      @click="toggleSync(`p${index}`, ($event.currentTarget as HTMLElement).parentElement?.nextElementSibling)"
                    >
                      <i :class="syncing === `p${index}` ? 'mdi mdi-stop' : 'mdi mdi-play'" />
                    </button>
                    <span class="ellipsis">{{ prompt }}</span>
                  </div>
                  <div class="mrow" :data-syncing="syncing === `p${index}` || undefined">
                    <div v-for="s in steps" :key="s" class="mcell">
                      <button v-if="cell(index, s)" @click="openGrid(index, s)">
                        <video
                          v-if="isVideo(cell(index, s)!)"
                          :src="viewUrl(cell(index, s)!)"
                          muted
                          loop
                          playsinline
                          preload="metadata"
                          @mouseenter="hoverPlay"
                          @mouseleave="hoverPause"
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
          <section v-for="g in statGroups" :key="g.title" class="card">
            <header class="sec-head"><span class="sec-title">{{ g.title }}</span></header>
            <dl class="sgrid">
              <div v-for="it in g.items" :key="it.k" class="scell">
                <dt><ZenTip v-if="it.tip" :text="it.tip">{{ it.k }}</ZenTip><template v-else>{{ it.k }}</template></dt>
                <dd :class="{ mono: it.mono !== false }">
                  {{ it.v }}<span v-if="it.delta" class="delta" :class="it.delta.good ? 'good' : 'bad'">{{ it.delta.text }}</span>
                </dd>
              </div>
            </dl>
          </section>
          <section class="card">
            <header class="sec-head">
              <span class="sec-title">Setup</span>
              <span class="spacer" />
              <button class="link" @click="specOpen = true"><i class="mdi mdi-code-json" /> Full spec</button>
            </header>
            <dl class="kv">
              <div v-for="it in setup" :key="it.k" class="kv-row">
                <dt>{{ it.k }}</dt>
                <dd :title="it.title ?? it.v">{{ it.v }}</dd>
              </div>
            </dl>
          </section>
          <section v-if="runLoaded.length" class="card loaded-card">
            <header class="sec-head">
              <span class="sec-title">In VRAM</span>
              <span class="sec-sub">loaded now</span>
            </header>
            <div v-for="m in runLoaded" :key="m.file ?? m.name" class="lrow" :title="`${m.file}: ${gb(m.vram)} of ${gb(m.bytes)} GB in VRAM`">
              <span class="lname ellipsis">{{ (m.file ?? m.name).replace(/\.safetensors$/, '') }}</span>
              <span class="mono small muted">{{ gb(m.vram) }}/{{ gb(m.bytes) }} GB</span>
              <div class="lbar"><span :style="{ width: `${m.bytes ? (m.vram / m.bytes) * 100 : 0}%` }" /></div>
            </div>
          </section>

          <section class="card">
            <header class="sec-head">
              <span class="sec-title"><ZenTip text="The LoRA saved along the way. Best: the highest likeness when faces are scored (a character), else the lowest validation loss. Hover a row to test it, train more from it, or copy its name">Checkpoints</ZenTip></span>
              <span class="sec-sub">{{ saveLine }}</span>
            </header>
            <div v-if="!ckRows.length" class="muted small">
              {{ running ? `The first save lands at step ${nextSave?.step ?? total}.` : 'None saved.' }}
            </div>
            <div class="ckpts">
              <div
                v-for="c in ckRows"
                :key="c.path"
                class="ckpt"
                :class="{ best: c.step === bestStep }"
                :title="c.size === null ? `File missing: ${c.path}` : c.path"
                tabindex="-1"
              >
                <span class="ck-step mono" :class="{ gone: c.size === null }">{{ c.step }}</span>
                <span v-if="c.final" class="chip">final</span>
                <span v-else-if="ckKind(c)" class="chip muted" :title="`Saved when the run ${ckKind(c) === 'manual' ? 'was asked to' : ckKind(c)}`">{{ ckKind(c) }}</span>
                <span v-if="c.step === bestStep" class="chip best-chip" :title="bestBy === 'likeness' ? 'Its samples look most like the dataset\'s faces' : 'Lowest validation loss of the saved checkpoints'">best</span>
                <span class="spacer" />
                <span class="ck-val mono">
                  <template v-if="likeAt(c.step) != null"><span title="How much its samples' faces look like the dataset's (same person ≳ 0.5)">♥ {{ likeAt(c.step)!.toFixed(2) }}</span> · </template>
                  <span title="Validation loss">{{ ckptVal(c.step) != null ? ckptVal(c.step)!.toFixed(4) : '—' }}</span>
                </span>
                <span class="ck-acts">
                  <RouterLink class="ib" :to="`/apps/test-lora?run=${id}&lora=${encodeURIComponent(c.lora_name)}`" title="Test: try it in Test LoRA, on this run's model"><i class="mdi mdi-flask-outline" /></RouterLink>
                  <button v-if="resumable && !c.final" class="ib" title="Resume this run from here, optimizer state and all" @click="resume(c.path)"><i class="mdi mdi-play-outline" /></button>
                  <RouterLink v-else class="ib" :to="`/runs/new?from=${id}&init=${encodeURIComponent(c.lora_name)}`" title="Train more: a new run with these settings that starts from this LoRA"><i class="mdi mdi-fast-forward-outline" /></RouterLink>
                  <button class="ib" :title="copied === c.lora_name ? 'Copied' : `Copy LoRA name: ${c.lora_name}`" @click="copy(c.lora_name)">
                    <i :class="copied === c.lora_name ? 'mdi mdi-check' : 'mdi mdi-content-copy'" />
                  </button>
                </span>
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
  overflow-wrap: anywhere; /* the run id and model wrap rather than cut off */
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
.loaded-card .sec-head {
  margin-bottom: 4px;
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
/* setup: label / value lines; values wrap instead of cutting off */
.kv {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.kv-row {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr);
  gap: 10px;
  font-size: 12px;
  line-height: 1.4;
}
.kv-row dt {
  color: var(--zen-muted);
}
.kv-row dd {
  margin: 0;
  white-space: normal;
  overflow: visible;
  overflow-wrap: anywhere;
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
  display: flex;
  align-items: center;
  gap: 3px;
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
  /* one row: the prompt grows, its options and Render sit at its right (they wrap below when narrow) */
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 8px;
}
.try-in {
  flex: 1 1 320px;
  min-width: 0;
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
  flex: 0 0 auto;
  min-height: 38px; /* the prompt box's one-line height, so they line up with it */
}
.try-opts > :first-child {
  width: 170px;
}
.empty-manual {
  padding: 8px 0;
}
.mrows {
  display: flex;
  flex-direction: column;
}
.mrow2 {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 12px;
  padding: 8px 6px;
  border-radius: var(--r-md);
}
.mrow2 + .mrow2 {
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 40%, transparent);
}
.mrow2:hover {
  background: color-mix(in srgb, var(--zen-text) 3%, transparent);
}
.mprompt {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.mtext {
  font-size: 11.5px;
  line-height: 1.4;
  color: var(--zen-text);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.mmeta {
  font-size: 10px;
  color: var(--zen-muted);
}
.macts {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.mrow2:hover .macts,
.macts:focus-within,
.macts.held {
  opacity: 1;
}
.macts button.on,
.rowplay.on {
  color: var(--zen-accent);
}
.rowplay {
  flex: none;
  width: 20px;
  height: 20px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 0;
}
.rowplay:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.macts button {
  width: 24px;
  height: 22px;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.macts button:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.tcell.add {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
}
.tcell.add i {
  font-size: 18px;
}
.tcell.add:hover {
  border-color: var(--zen-accent);
  color: var(--zen-accent);
}
.mcells {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.tcell {
  position: relative;
  flex: none;
  height: 96px;
  aspect-ratio: 1;
  border: none;
  padding: 0;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
}
.tcell img,
.tcell video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.tcell.pending {
  display: grid;
  place-items: center;
  cursor: default;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font-size: 20px;
}
.tcell.pending.live {
  border: 1px solid color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
.tcell.pending img {
  position: absolute;
  inset: 0;
}
.clabel {
  position: absolute;
  left: 6px;
  bottom: 6px;
  max-width: calc(100% - 12px);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  pointer-events: none;
}
.qbar {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 3px;
  background: color-mix(in srgb, var(--zen-text) 10%, transparent);
}
.qbar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
  transition: width var(--zen-dur) ease;
}
.again {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: var(--r-sm);
  cursor: pointer;
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
.live-dot.off {
  visibility: hidden; /* keeps its room: the tab doesn't change width when a render starts */
}
.base-sw {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
  white-space: nowrap;
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
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 4.2em; /* three lines, so tiles line up; the whole prompt is its tooltip */
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
  flex-wrap: wrap; /* narrow: the controls drop to a second line rather than spill out */
  row-gap: 8px;
}
.samples-head .sec-sub {
  flex-shrink: 0;
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
  gap: 1px;
}
/* one line per checkpoint; its actions take the numbers' place on hover */
.ckpts .ckpt {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 8px;
  margin: 0 -8px;
  border-radius: var(--r-md);
  font-size: 12px;
}
.ckpts .ckpt:hover,
.ckpts .ckpt:focus-within {
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.ckpts .ckpt.best {
  background: color-mix(in srgb, var(--zen-ok) 8%, transparent);
}
.ckpts .ck-step {
  font-weight: 650;
  font-size: 13px;
  min-width: 4ch;
}
.ckpts .ck-step.gone {
  color: var(--zen-warn);
  text-decoration: line-through;
}
.ckpts .ckpt .chip {
  font-size: 10px;
  padding: 0 6px;
  line-height: 16px;
}
.ckpts .ckpt .chip.muted {
  color: var(--zen-muted);
}
.ckpts .ck-val {
  font-size: 11px;
  color: var(--zen-muted);
  white-space: nowrap;
  transition: opacity 0.12s ease;
}
.ckpts .ck-acts {
  position: absolute;
  right: 4px;
  top: 50%;
  translate: 0 -50%;
  display: flex;
  gap: 2px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
}
.ckpts .ckpt:hover .ck-acts,
.ckpts .ckpt:focus-within .ck-acts {
  opacity: 1;
  pointer-events: auto;
}
.ckpts .ckpt:hover .ck-val,
.ckpts .ckpt:focus-within .ck-val {
  opacity: 0;
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
