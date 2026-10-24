<script setup lang="ts">
// Two (or more) runs on one chart: training loss (smoothed, as on a run's page), validation loss,
// or likeness. Each series keeps its run's own colour and name; values for every series show on
// hover, at the nearest step each has.
import { maxOf, minOf } from '@/lib/extent'
import { computed, ref } from 'vue'
import type { MetricRow, ValidationRow } from '@/api/engine'

export interface CompareSeries {
  label: string
  color: string
  rows: MetricRow[]
  val: ValidationRow[]
  like: { step: number; score: number }[]
}
const props = withDefaults(defineProps<{ series: CompareSeries[]; metric: 'loss' | 'val' | 'like'; height?: number }>(), { height: 240 })

const W = 720
const pad = { l: 46, r: 14, t: 10, b: 24 }

function ema(rows: MetricRow[]) {
  const out: { step: number; v: number }[] = []
  let s: number | null = null
  for (const r of rows) {
    if (r.loss == null) continue
    s = s == null ? r.loss : 0.95 * s + 0.05 * r.loss
    out.push({ step: r.step, v: s })
  }
  return out
}
const lines = computed(() =>
  props.series.map((s) => ({
    ...s,
    pts:
      props.metric === 'loss'
        ? ema(s.rows)
        : props.metric === 'val'
          ? s.val.map((r) => ({ step: r.step, v: r.val_loss }))
          : s.like.map((r) => ({ step: r.step, v: r.score })),
  })),
)
const all = computed(() => lines.value.flatMap((l) => l.pts))
const xMax = computed(() => Math.max(maxOf(all.value.map((p) => p.step), 1), maxOf(props.series.map((s) => s.rows.at(-1)?.step ?? 0))))
const yRange = computed(() => {
  if (props.metric === 'like') return [0, 1] as const
  const v = all.value.map((p) => p.v)
  if (!v.length) return [0, 1] as const
  const lo = minOf(v)
  const hi = maxOf(v)
  const py = (hi - lo || hi || 1) * 0.08
  return [Math.max(0, lo - py), hi + py] as const
})
const x = (step: number) => pad.l + (step / xMax.value) * (W - pad.l - pad.r)
const y = (v: number) => {
  const [lo, hi] = yRange.value
  return pad.t + (1 - (v - lo) / (hi - lo || 1)) * (props.height - pad.t - pad.b)
}
const pathOf = (pts: { step: number; v: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.v).toFixed(1)}`).join('')
const yTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => yRange.value[0] + f * (yRange.value[1] - yRange.value[0])))
const xTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax.value)))
const dots = computed(() => props.metric !== 'loss')
const fmt = (v: number) => (props.metric === 'like' ? v.toFixed(2) : Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(2))

const hoverStep = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const box = (ev.currentTarget as SVGSVGElement).getBoundingClientRect()
  const sx = ((ev.clientX - box.left) / box.width) * W
  hoverStep.value = Math.max(0, Math.min(xMax.value, ((sx - pad.l) / (W - pad.l - pad.r)) * xMax.value))
}
const hovered = computed(() => {
  const h = hoverStep.value
  if (h == null) return null
  return lines.value.map((l) => {
    let best: { step: number; v: number } | null = null
    for (const p of l.pts) if (!best || Math.abs(p.step - h) < Math.abs(best.step - h)) best = p
    return { label: l.label, color: l.color, p: best }
  })
})
</script>

<template>
  <figure class="cmp-chart">
    <figcaption class="legend">
      <span v-for="s in series" :key="s.label" class="key">
        <svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" :stroke="s.color" stroke-width="2" /></svg>
        {{ s.label }}
      </span>
    </figcaption>
    <svg :viewBox="`0 0 ${W} ${height}`" class="plot" @pointermove="onMove" @pointerleave="hoverStep = null">
      <g class="grid">
        <line v-for="t in yTicks" :key="`y${t}`" :x1="pad.l" :x2="W - pad.r" :y1="y(t)" :y2="y(t)" />
      </g>
      <g class="axis">
        <text v-for="t in yTicks" :key="`yl${t}`" :x="pad.l - 6" :y="y(t) + 3" text-anchor="end">{{ fmt(t) }}</text>
        <text v-for="t in xTicks" :key="`xl${t}`" :x="x(t)" :y="height - 6" text-anchor="middle">{{ t }}</text>
      </g>
      <line v-if="metric === 'like'" class="half" :x1="pad.l" :x2="W - pad.r" :y1="y(0.5)" :y2="y(0.5)" />
      <g v-for="l in lines" :key="l.label">
        <path :d="pathOf(l.pts)" fill="none" :stroke="l.color" stroke-width="2" stroke-linejoin="round" />
        <template v-if="dots">
          <circle v-for="p in l.pts" :key="p.step" :cx="x(p.step)" :cy="y(p.v)" r="3.5" :fill="l.color" class="dot" />
        </template>
      </g>
      <line v-if="hoverStep != null" class="cursor" :x1="x(hoverStep)" :x2="x(hoverStep)" :y1="pad.t" :y2="height - pad.b" />
    </svg>
    <div v-if="!all.length" class="empty muted small">
      {{ metric === 'val' ? 'Neither run has validation loss' : metric === 'like' ? 'Neither run has likeness scores' : 'No loss yet' }}
    </div>
    <div class="readout small">
      <template v-if="hovered">
        <span v-for="h in hovered" :key="h.label" class="ro">
          <i class="sw" :style="{ background: h.color }" />{{ h.label }}:
          <b class="mono">{{ h.p ? fmt(h.p.v) : '–' }}</b><span v-if="h.p" class="muted"> @ {{ h.p.step }}</span>
        </span>
      </template>
      <span v-else class="muted">Hover the chart for both runs' values</span>
    </div>
  </figure>
</template>

<style scoped>
.cmp-chart {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  position: relative;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  font-size: 12px;
  color: var(--zen-text);
}
.key {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.plot {
  width: 100%;
  height: auto;
  display: block;
}
.grid line {
  stroke: color-mix(in srgb, var(--zen-border) 60%, transparent);
  stroke-width: 1;
}
.axis text {
  fill: var(--zen-muted);
  font-size: 10px;
  font-family: var(--zen-mono, monospace);
}
.half {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 4;
  opacity: 0.5;
}
.dot {
  stroke: var(--zen-surface);
  stroke-width: 2;
}
.cursor {
  stroke: var(--zen-muted);
  stroke-dasharray: 3 3;
}
.empty {
  position: absolute;
  inset: 30px 0 30px 0;
  display: grid;
  place-items: center;
}
.readout {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  min-height: 18px;
}
.ro {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.sw {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  display: inline-block;
}
.mono {
  font-family: var(--zen-mono, monospace);
}
</style>
