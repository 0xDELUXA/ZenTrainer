<script setup lang="ts">
// Loss over optimizer steps: training loss (EMA-smoothed in the accent, raw drawn faintly
// in the same hue) and, when the run has it, the fixed-set validation loss as a second
// series with markers. Two series, one unit, one axis: a legend names them.
// Crosshair + tooltip on hover. The learning rate, when it changes (warmup, decay, Automagic
// finding its own), rides on a right-hand axis as ai-toolkit draws it: a thin solid line in
// the muted ink, with its own legend key and axis labels, so it never reads as a loss.
import { computed, ref } from 'vue'
import type { MetricRow, ValidationRow } from '@/api/engine'

const props = withDefaults(
  defineProps<{
    rows: MetricRow[]
    total?: number
    smoothing?: number
    marks?: { step: number; kind: 'ckpt' | 'sample'; done: boolean }[]
    val?: ValidationRow[]
    height?: number
  }>(),
  { smoothing: 0.9, marks: () => [], val: () => [], height: 240 },
)

const W = 720
const H = computed(() => props.height)
const pad = computed(() => ({ l: 44, r: lrShown.value ? 46 : 12, t: 10, b: 24 }))

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
const yRange = computed(() => {
  const v = [...points.value.map((p) => p.loss), ...props.val.map((r) => r.val_loss)]
  if (!v.length) return [0, 1] as const
  const lo = Math.min(...v)
  const hi = Math.max(...v)
  const padY = (hi - lo || hi || 1) * 0.08
  return [Math.max(0, lo - padY), hi + padY] as const
})

const x = (step: number) => pad.value.l + (step / xMax.value) * (W - pad.value.l - pad.value.r)
const y = (v: number) => {
  const [lo, hi] = yRange.value
  return pad.value.t + (1 - (v - lo) / (hi - lo || 1)) * (H.value - pad.value.t - pad.value.b)
}

const path = (vals: number[]) =>
  points.value.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(vals[i]!).toFixed(1)}`).join('')

const rawPath = computed(() => path(points.value.map((p) => p.loss)))
const valPath = computed(() =>
  props.val.map((r, i) => `${i ? 'L' : 'M'}${x(r.step).toFixed(1)},${y(r.val_loss).toFixed(1)}`).join(''),
)
const valAt = (step: number) => [...props.val].reverse().find((r) => r.step <= step)
const emaPath = computed(() => path(ema.value))

const yTicks = computed(() => {
  const [lo, hi] = yRange.value
  return [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo))
})
const xTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax.value)))

const hover = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const svg = ev.currentTarget as SVGSVGElement
  const box = svg.getBoundingClientRect()
  const sx = ((ev.clientX - box.left) / box.width) * W
  const step = ((sx - pad.value.l) / (W - pad.value.l - pad.value.r)) * xMax.value
  let best: number | null = null
  let bestD = Infinity
  points.value.forEach((p, i) => {
    const d = Math.abs(p.step - step)
    if (d < bestD) [best, bestD] = [i, d]
  })
  hover.value = best
}
// learning rate on the right axis: linear from 0, drawn only when the LR actually moves
const lrPoints = computed(() => points.value.filter((p) => p.lr != null) as (MetricRow & { loss: number; lr: number })[])
const lrShown = computed(() => {
  const v = lrPoints.value.map((p) => p.lr)
  return v.length > 1 && Math.max(...v) > Math.min(...v) * 1.001
})
const lrMax = computed(() => Math.max(...lrPoints.value.map((p) => p.lr), 1e-12))
const ly = (v: number) => pad.value.t + (1 - v / (lrMax.value * 1.05)) * (H.value - pad.value.t - pad.value.b)
const lrTicks = computed(() => [0, 0.5, 1].map((f) => f * lrMax.value))
const lrPath = computed(() => lrPoints.value.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${ly(p.lr).toFixed(1)}`).join(''))
const lrNow = computed(() => lrPoints.value.at(-1)?.lr)
const fmtLr = (v?: number) => (v == null ? '' : v === 0 ? '0' : v.toExponential(1))

const hovered = computed(() => (hover.value == null ? null : { p: points.value[hover.value]!, s: ema.value[hover.value]! }))
const fmt = (v: number) => (v === 0 || Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(2))
</script>

<template>
  <figure class="loss">
    <figcaption>
      <span class="legend">
        <span class="key"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-ema" /></svg> Training (EMA {{ smoothing }}, raw faint)</span>
        <span v-if="val.length" class="key"><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" class="k-val" /><circle cx="9" cy="4" r="3" class="k-val-dot" /></svg> Validation (fixed set)</span>
        <span v-if="lrShown" class="key muted"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-lr" /></svg> Learning rate (right axis)</span>
        <span v-if="marks.some((m) => m.kind === 'ckpt')" class="key muted"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-mark" /></svg> checkpoints</span>
      </span>
    </figcaption>
    <div v-if="!points.length" class="empty muted" :style="{ height: `${height}px` }">Waiting for the first step…</div>
    <div v-else class="plot" :style="{ height: `${height}px` }">
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" @pointermove="onMove" @pointerleave="hover = null">
        <g class="grid">
          <line v-for="t in yTicks" :key="t" :x1="pad.l" :x2="W - pad.r" :y1="y(t)" :y2="y(t)" />
        </g>
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
        <path class="raw" :d="rawPath" />
        <path class="ema" :d="emaPath" />
        <path v-if="lrShown" class="lr-line" :d="lrPath" />
        <path v-if="val.length > 1" class="val" :d="valPath" />
        <circle v-for="r in val" :key="r.step" class="val-dot" :cx="x(r.step)" :cy="y(r.val_loss)" r="3.5" />
        <g v-if="hovered">
          <line class="cross" :x1="x(hovered.p.step)" :x2="x(hovered.p.step)" :y1="pad.t" :y2="H - pad.b" />
          <circle class="dot" :cx="x(hovered.p.step)" :cy="y(hovered.s)" r="4" />
        </g>
      </svg>
      <div class="y-axis">
        <span v-for="t in yTicks" :key="t" :style="{ top: `${(y(t) / H) * 100}%` }">{{ fmt(t) }}</span>
      </div>
      <div v-if="lrShown" class="y-axis right">
        <span class="lr-head">LR</span>
        <span v-for="t in lrTicks" :key="t" :style="{ top: `${(ly(t) / H) * 100}%` }">{{ fmtLr(t) }}</span>
      </div>
      <div
        v-if="hovered"
        class="tip"
        :style="{ left: `${(x(hovered.p.step) / W) * 100}%`, transform: `translateX(${hovered.p.step > xMax / 2 ? '-105%' : '5%'})` }"
      >
        <div class="mono">step {{ hovered.p.step }}</div>
        <div>loss <b class="mono">{{ fmt(hovered.p.loss) }}</b></div>
        <div>smoothed <b class="mono">{{ fmt(hovered.s) }}</b></div>
        <div v-if="valAt(hovered.p.step)">
          validation <b class="mono">{{ fmt(valAt(hovered.p.step)!.val_loss) }}</b>
          <span class="muted">@{{ valAt(hovered.p.step)!.step }}</span>
        </div>
        <div v-if="hovered.p.lr != null" class="muted">lr <span class="mono">{{ fmtLr(hovered.p.lr) }}</span></div>
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
  font-weight: 600;
}
figcaption .muted {
  font-weight: 400;
  font-size: 12px;
}
.empty {
  display: grid;
  place-items: center;
}
.plot {
  position: relative;
}
.x-axis {
  position: relative;
  height: 14px;
  margin-top: -4px;
}
/* solid and grey: the checkpoint markers are the dotted ones, the losses the coloured ones */
.lr-line {
  fill: none;
  stroke: var(--zen-muted);
  stroke-width: 1.5;
  opacity: 0.85;
  vector-effect: non-scaling-stroke;
}
.k-lr {
  stroke: var(--zen-muted);
  stroke-width: 1.5;
}
.y-axis.right span {
  left: auto;
  right: 0;
}
.y-axis.right .lr-head {
  top: -2px;
  transform: none;
}
/* the plot only: legend swatches are small inline svgs */
.plot > svg {
  width: 100%;
  height: 100%;
  display: block;
  touch-action: none;
}
.grid line {
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
  opacity: 0.25;
}
.ema {
  stroke-width: 2;
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
.y-axis span,
.x-axis span {
  position: absolute;
  font-size: 10px;
  color: var(--zen-muted);
  font-family: var(--zen-mono);
  pointer-events: none;
}
.y-axis span {
  left: 0;
  transform: translateY(-50%);
}
.x-axis span {
  bottom: 0;
  transform: translateX(-50%);
}
.tip {
  position: absolute;
  top: 8px;
  padding: 6px 8px;
  background: var(--zen-surface);
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
  box-shadow: 0 4px 14px rgb(0 0 0 / 0.25);
}
</style>
<style scoped>
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  font-size: 12px;
  font-weight: 400;
  color: var(--zen-muted);
}
.key {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.k-ema {
  stroke: var(--zen-accent);
  stroke-width: 2;
}
.k-val {
  stroke: var(--chart-2, var(--zen-info));
  stroke-width: 2;
  stroke-dasharray: 4 3;
}
.k-val-dot,
.val-dot {
  fill: var(--chart-2, var(--zen-info));
  stroke: var(--zen-surface);
  stroke-width: 1.5;
}
.k-mark {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 3;
}
.val {
  fill: none;
  stroke: var(--chart-2, var(--zen-info));
  stroke-width: 2;
  stroke-dasharray: 5 4;
  vector-effect: non-scaling-stroke;
}
.val-dot {
  vector-effect: non-scaling-stroke;
}
</style>
