<script setup lang="ts">
// A run's curves over optimizer steps, read for their shape: training loss (EMA-smoothed in the
// accent, raw drawn faintly), the fixed-set validation loss, the learning rate when it moves, and,
// with the faces extra, the samples' likeness. No y-axes: each curve has its own scale so its
// trend fills the plot, each ends in a label with its latest value, and hovering gives every
// value at that step (the crosshair). Likeness keeps a fixed 0-1 scale, with 0.5 (a face starts
// to read as the same person) marked, since its level means something on its own.
import { computed, ref } from 'vue'
import type { MetricRow, ValidationRow } from '@/api/engine'

const props = withDefaults(
  defineProps<{
    rows: MetricRow[]
    total?: number
    smoothing?: number
    marks?: { step: number; kind: 'ckpt' | 'sample'; done: boolean }[]
    val?: ValidationRow[]
    /** faces extra: each sampled step's mean likeness */
    like?: { step: number; score: number }[]
    height?: number
  }>(),
  // ~50 steps: diffusion loss is noisy enough step to step that a shorter average still zigzags
  { smoothing: 0.98, marks: () => [], val: () => [], like: () => [], height: 240 },
)

const W = 720
const H = computed(() => props.height)
const pad = { l: 6, r: 52, t: 10, b: 10 } // the right margin holds the end labels

const points = computed(() => props.rows.filter((r) => r.loss != null) as (MetricRow & { loss: number })[])
const ema = computed(() => {
  const out: number[] = []
  let s: number | null = null
  for (const p of points.value) {
    s = s == null ? p.loss : props.smoothing * s + (1 - props.smoothing) * p.loss
    out.push(s)
  }
  return out
})

const xMax = computed(() => Math.max(props.total ?? 0, points.value.at(-1)?.step ?? 1, 1))
const x = (step: number) => pad.l + (step / xMax.value) * (W - pad.l - pad.r)
// a value in [lo, hi] to a height in the plot
const scale = (lo: number, hi: number) => (v: number) =>
  pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H.value - pad.t - pad.b)
const span = (vs: number[], floor0 = false): [number, number] => {
  if (!vs.length) return [0, 1]
  const lo = Math.min(...vs)
  const hi = Math.max(...vs)
  const m = (hi - lo || Math.abs(hi) || 1) * 0.12
  return [floor0 ? Math.max(0, lo - m) : lo - m, hi + m]
}

// loss: scaled to the smoothed curve so it fills the plot; a few spiky raw steps would otherwise
// flatten it (they clip at the top)
const yLoss = computed(() => scale(...span(ema.value, true)))
const path = (pts: { step: number }[], ys: number[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${ys[i]!.toFixed(1)}`).join('')
const rawPath = computed(() => path(points.value, points.value.map((p) => yLoss.value(p.loss))))
const emaPath = computed(() => path(points.value, ema.value.map((v) => yLoss.value(v))))

// validation: its own scale (it moves little next to the noisy training loss)
const yVal = computed(() => scale(...span(props.val.map((r) => r.val_loss))))
const valPath = computed(() => path(props.val, props.val.map((r) => yVal.value(r.val_loss))))
const valAt = (step: number) => [...props.val].reverse().find((r) => r.step <= step)

// learning rate: from 0, drawn only when it actually moves
const lrPoints = computed(() => points.value.filter((p) => p.lr != null) as (MetricRow & { loss: number; lr: number })[])
const lrShown = computed(() => {
  const v = lrPoints.value.map((p) => p.lr)
  return v.length > 1 && Math.max(...v) > Math.min(...v) * 1.001
})
const yLr = computed(() => scale(0, Math.max(...lrPoints.value.map((p) => p.lr), 1e-12) * 1.05))
const lrPath = computed(() => path(lrPoints.value, lrPoints.value.map((p) => yLr.value(p.lr))))

// likeness: a fixed scale (below 0 only if a score is)
const yLike = computed(() => scale(Math.min(0, ...props.like.map((r) => r.score)) - 0.05, 1.05))
const likePath = computed(() => path(props.like, props.like.map((r) => yLike.value(r.score))))
const likeAt = (step: number) => [...props.like].reverse().find((r) => r.step <= step) ?? null

const fmt = (v: number) => (v === 0 || Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(2))
const fmtLr = (v?: number) => (v == null ? '' : v === 0 ? '0' : v.toExponential(1))

// each curve's latest value at its right end, nudged apart so the labels never overlap
const ends = computed(() => {
  const out: { key: string; y: number; text: string }[] = []
  const lastI = points.value.length - 1
  if (lastI >= 0) out.push({ key: 'ema', y: yLoss.value(ema.value[lastI]!), text: fmt(ema.value[lastI]!) })
  const v = props.val.at(-1)
  if (v) out.push({ key: 'val', y: yVal.value(v.val_loss), text: fmt(v.val_loss) })
  const l = props.like.at(-1)
  if (l) out.push({ key: 'like', y: yLike.value(l.score), text: l.score.toFixed(2) })
  out.sort((a, b) => a.y - b.y)
  const GAP = 12 // px: the plot's height in viewBox units is its height in pixels
  for (let i = 1; i < out.length; i++) out[i]!.y = Math.max(out[i]!.y, out[i - 1]!.y + GAP)
  return out
})

const xTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax.value)))

const hover = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const box = (ev.currentTarget as SVGSVGElement).getBoundingClientRect()
  const step = ((((ev.clientX - box.left) / box.width) * W - pad.l) / (W - pad.l - pad.r)) * xMax.value
  let best: number | null = null
  let bestD = Infinity
  points.value.forEach((p, i) => {
    const d = Math.abs(p.step - step)
    if (d < bestD) [best, bestD] = [i, d]
  })
  hover.value = best
}
const hovered = computed(() => (hover.value == null ? null : { p: points.value[hover.value]!, s: ema.value[hover.value]! }))
</script>

<template>
  <figure class="loss">
    <figcaption>
      <span class="legend">
        <span class="key"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-ema" /></svg> Training loss</span>
        <span v-if="val.length" class="key"><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" class="k-val" /><circle cx="9" cy="4" r="3" class="k-val-dot" /></svg> Validation</span>
        <span v-if="like.length" class="key"><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" class="k-like" /><circle cx="9" cy="4" r="3" class="k-like-dot" /></svg> Likeness</span>
        <span v-if="lrShown" class="key muted"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-lr" /></svg> Learning rate</span>
        <span v-if="marks.some((m) => m.kind === 'ckpt')" class="key muted"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-mark" /></svg> Checkpoints</span>
        <span class="muted hint">hover for values</span>
      </span>
    </figcaption>
    <div v-if="!points.length" class="empty muted" :style="{ height: `${height}px` }">Waiting for the first step…</div>
    <div v-else class="plot" :style="{ height: `${height}px` }">
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" @pointermove="onMove" @pointerleave="hover = null">
        <defs>
          <clipPath id="loss-plot"><rect :x="pad.l" :y="0" :width="W - pad.l - pad.r" :height="H" /></clipPath>
        </defs>
        <line class="base" :x1="pad.l" :x2="W - pad.r" :y1="H - pad.b" :y2="H - pad.b" />
        <line
          v-for="m in marks.filter((m) => m.kind === 'ckpt')"
          :key="m.step"
          class="mark"
          :class="{ done: m.done }"
          :x1="x(m.step)"
          :x2="x(m.step)"
          :y1="pad.t"
          :y2="H - pad.b"
        />
        <g clip-path="url(#loss-plot)">
          <line v-if="like.length" class="like-ref" :x1="pad.l" :x2="W - pad.r" :y1="yLike(0.5)" :y2="yLike(0.5)" />
          <path class="raw" :d="rawPath" />
          <path v-if="lrShown" class="lr-line" :d="lrPath" />
          <path class="ema" :d="emaPath" />
          <path v-if="val.length > 1" class="val" :d="valPath" />
          <circle v-for="r in val" :key="`v${r.step}`" class="val-dot" :cx="x(r.step)" :cy="yVal(r.val_loss)" r="3.5" />
          <path v-if="like.length > 1" class="like" :d="likePath" />
          <circle v-for="r in like" :key="`l${r.step}`" class="like-dot" :cx="x(r.step)" :cy="yLike(r.score)" r="3.5" />
        </g>
        <g v-if="hovered">
          <line class="cross" :x1="x(hovered.p.step)" :x2="x(hovered.p.step)" :y1="pad.t" :y2="H - pad.b" />
          <circle class="dot" :cx="x(hovered.p.step)" :cy="yLoss(hovered.s)" r="4" />
        </g>
      </svg>
      <span
        v-for="e in ends"
        :key="e.key"
        class="end mono"
        :class="e.key"
        :style="{ top: `${(e.y / H) * 100}%`, left: `${((W - pad.r + 6) / W) * 100}%` }"
      >{{ e.text }}</span>
      <div
        v-if="hovered"
        class="tip"
        :style="{ left: `${(x(hovered.p.step) / W) * 100}%`, transform: `translateX(${hovered.p.step > xMax / 2 ? '-105%' : '5%'})` }"
      >
        <div class="mono">step {{ hovered.p.step }}</div>
        <div><i class="sw ema" />loss <b class="mono">{{ fmt(hovered.s) }}</b> <span class="muted">· raw {{ fmt(hovered.p.loss) }}</span></div>
        <div v-if="valAt(hovered.p.step)">
          <i class="sw val" />validation <b class="mono">{{ fmt(valAt(hovered.p.step)!.val_loss) }}</b>
          <span class="muted">@{{ valAt(hovered.p.step)!.step }}</span>
        </div>
        <div v-if="likeAt(hovered.p.step)">
          <i class="sw like" />likeness <b class="mono">{{ likeAt(hovered.p.step)!.score.toFixed(2) }}</b>
          <span class="muted">@{{ likeAt(hovered.p.step)!.step }}</span>
        </div>
        <div v-if="hovered.p.lr != null" class="muted"><i class="sw lr" />lr <span class="mono">{{ fmtLr(hovered.p.lr) }}</span></div>
      </div>
    </div>
    <div v-if="points.length" class="x-axis">
      <span v-for="t in xTicks" :key="t" :style="{ left: `${(x(t) / W) * 100}%` }">{{ t }}</span>
    </div>
  </figure>
</template>

<style scoped>
.loss {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  align-items: center;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 14px;
  font-size: 12px;
  color: var(--zen-muted);
}
.legend .hint {
  margin-left: auto;
  font-size: 11px;
  opacity: 0.7;
}
.key {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.empty {
  display: grid;
  place-items: center;
}
.plot {
  position: relative;
}
.plot > svg {
  width: 100%;
  height: 100%;
  display: block;
  touch-action: none;
}
.x-axis {
  position: relative;
  height: 14px;
  margin-top: -2px;
}
.x-axis span {
  position: absolute;
  bottom: 0;
  transform: translateX(-50%);
  font-size: 10px;
  color: var(--zen-muted);
  font-family: var(--zen-mono);
  pointer-events: none;
}
.base {
  stroke: var(--zen-border);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.mark {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 4;
  opacity: 0.45;
  vector-effect: non-scaling-stroke;
}
.mark.done {
  stroke: var(--zen-accent);
  opacity: 0.6;
}
.raw,
.ema {
  fill: none;
  stroke: var(--zen-accent);
  vector-effect: non-scaling-stroke;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.raw {
  stroke-width: 1;
  opacity: 0.1;
}
.ema {
  stroke-width: 2;
}
.lr-line {
  fill: none;
  stroke: var(--zen-muted);
  stroke-width: 1.5;
  opacity: 0.7;
  vector-effect: non-scaling-stroke;
}
.val {
  fill: none;
  stroke: var(--chart-2, var(--zen-info));
  stroke-width: 2;
  stroke-dasharray: 5 4;
  vector-effect: non-scaling-stroke;
}
.val-dot {
  fill: var(--chart-2, var(--zen-info));
  stroke: var(--zen-surface);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.like {
  fill: none;
  stroke: var(--zen-ok);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.like-dot {
  fill: var(--zen-ok);
  stroke: var(--zen-surface);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.like-ref {
  stroke: var(--zen-ok);
  stroke-width: 1;
  stroke-dasharray: 2 4;
  opacity: 0.35;
  vector-effect: non-scaling-stroke;
}
.cross {
  stroke: var(--zen-muted);
  stroke-dasharray: 3 3;
  vector-effect: non-scaling-stroke;
}
.dot {
  fill: var(--zen-accent);
  stroke: var(--zen-surface);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
/* each curve's latest value; a swatch-coloured tick before it carries which curve */
.end {
  position: absolute;
  transform: translateY(-50%);
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: var(--zen-text);
  pointer-events: none;
  white-space: nowrap;
}
.end::before {
  content: '';
  width: 6px;
  height: 2px;
  border-radius: 1px;
}
.end.ema::before {
  background: var(--zen-accent);
}
.end.val::before {
  background: var(--chart-2, var(--zen-info));
}
.end.like::before {
  background: var(--zen-ok);
}
.tip {
  position: absolute;
  top: 8px;
  padding: 6px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  background: var(--zen-surface);
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
  box-shadow: 0 4px 14px rgb(0 0 0 / 0.25);
}
.sw {
  display: inline-block;
  width: 8px;
  height: 2px;
  margin-right: 6px;
  vertical-align: middle;
  border-radius: 1px;
}
.sw.ema {
  background: var(--zen-accent);
}
.sw.val {
  background: var(--chart-2, var(--zen-info));
}
.sw.like {
  background: var(--zen-ok);
}
.sw.lr {
  background: var(--zen-muted);
}
/* legend swatches */
.k-ema {
  stroke: var(--zen-accent);
  stroke-width: 2;
}
.k-val {
  stroke: var(--chart-2, var(--zen-info));
  stroke-width: 2;
  stroke-dasharray: 4 3;
}
.k-val-dot {
  fill: var(--chart-2, var(--zen-info));
  stroke: var(--zen-surface);
  stroke-width: 1.5;
}
.k-like {
  stroke: var(--zen-ok);
  stroke-width: 2;
}
.k-like-dot {
  fill: var(--zen-ok);
  stroke: var(--zen-surface);
  stroke-width: 1.5;
}
.k-lr {
  stroke: var(--zen-muted);
  stroke-width: 1.5;
}
.k-mark {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 3;
}
</style>
