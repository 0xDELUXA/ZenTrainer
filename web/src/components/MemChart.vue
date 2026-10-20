<script setup lang="ts">
// Memory over time, one unit (GB) and one axis. Series are told apart by colour, dash
// and the legend, never colour alone. Hover shows every series at that moment.
// `height` sizes the plot (the log drawer fits it to its own height); `markers` draw thin
// event lines on the time axis (model loads, errors), each with a hover title.
import { computed, ref } from 'vue'
import type { MemSample } from '@/api/engine'

export interface Series {
  key: keyof MemSample
  label: string
  color: string
  dash?: string
}
export interface Marker {
  t: number // seconds, same clock as MemSample.t
  label: string
  kind: 'error' | 'event'
}
const props = withDefaults(
  defineProps<{ samples: MemSample[]; series: Series[]; total?: number; title: string; height?: number; markers?: Marker[] }>(),
  { height: 200, markers: () => [] },
)

const W = 720
const H = computed(() => props.height)
const pad = { l: 40, r: 10, t: 10, b: 22 }
const GB = 1024 ** 3

const present = computed(() => props.series.filter((s) => props.samples.some((x) => x[s.key] != null)))
const t0 = computed(() => props.samples[0]?.t ?? 0)
const t1 = computed(() => Math.max(props.samples.at(-1)?.t ?? 1, t0.value + 60))
const yMax = computed(() => {
  const vals = props.samples.flatMap((x) => present.value.map((s) => (x[s.key] as number | undefined) ?? 0))
  return Math.max(props.total ?? 0, ...vals, 1) / GB
})
const x = (t: number) => pad.l + ((t - t0.value) / (t1.value - t0.value)) * (W - pad.l - pad.r)
const y = (b: number) => pad.t + (1 - b / GB / yMax.value) * (H.value - pad.t - pad.b)
// only the events inside the plotted window
const inRange = computed(() => props.markers.filter((m) => m.t >= t0.value && m.t <= t1.value))
const path = (s: Series) =>
  props.samples
    .filter((p) => p[s.key] != null)
    .map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p[s.key] as number).toFixed(1)}`)
    .join('')
const yTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax.value))
const ago = (t: number) => {
  const s = Math.round(t1.value - t)
  if (s <= 1) return 'now'
  return s >= 60 ? `-${Math.round(s / 60)}m` : `-${s}s`
}
const xTicks = computed(() => [0, 0.5, 1].map((f) => t0.value + f * (t1.value - t0.value)))

const hover = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const box = (ev.currentTarget as SVGSVGElement).getBoundingClientRect()
  const t = t0.value + (((ev.clientX - box.left) / box.width) * W - pad.l) / (W - pad.l - pad.r) * (t1.value - t0.value)
  let best = 0
  props.samples.forEach((p, i) => {
    if (Math.abs(p.t - t) < Math.abs(props.samples[best]!.t - t)) best = i
  })
  hover.value = props.samples.length ? best : null
}
const hs = computed(() => (hover.value == null ? null : props.samples[hover.value]))
</script>

<template>
  <figure class="chart">
    <figcaption>
      <span class="title">{{ title }}</span>
      <span class="legend">
        <span v-for="s in present" :key="s.key" class="key">
          <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" :stroke="s.color" stroke-width="2" :stroke-dasharray="s.dash" /></svg>
          {{ s.label }}
        </span>
      </span>
    </figcaption>
    <div v-if="samples.length < 2" class="empty muted" :style="{ height: `${height}px` }">Collecting samples…</div>
    <div v-else class="plot" :style="{ height: `${height}px` }">
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" @pointermove="onMove" @pointerleave="hover = null">
        <line v-for="v in yTicks" :key="v" class="grid" :x1="pad.l" :x2="W - pad.r" :y1="y(v * GB)" :y2="y(v * GB)" />
        <line v-if="total" class="cap" :x1="pad.l" :x2="W - pad.r" :y1="y(total)" :y2="y(total)" />
        <g v-for="(m, i) in inRange" :key="i" class="marker" :class="m.kind">
          <title>{{ m.label }}</title>
          <line class="mk" :x1="x(m.t)" :x2="x(m.t)" :y1="pad.t" :y2="H - pad.b" />
          <line class="mk-hit" :x1="x(m.t)" :x2="x(m.t)" :y1="pad.t" :y2="H - pad.b" />
        </g>
        <path v-for="s in present" :key="s.key" :d="path(s)" :stroke="s.color" :stroke-dasharray="s.dash" class="line" />
        <line v-if="hs" class="cross" :x1="x(hs.t)" :x2="x(hs.t)" :y1="pad.t" :y2="H - pad.b" />
      </svg>
      <span v-for="v in yTicks" :key="v" class="ytick" :style="{ top: `${(y(v * GB) / H) * 100}%` }">{{ v.toFixed(v < 10 ? 1 : 0) }}</span>
      <span v-for="t in xTicks" :key="t" class="xtick" :style="{ left: `${(x(t) / W) * 100}%` }">{{ ago(t) }}</span>
      <div v-if="hs" class="tip" :style="{ left: `${(x(hs.t) / W) * 100}%`, transform: `translateX(${x(hs.t) > W / 2 ? '-105%' : '5%'})` }">
        <div class="muted small">{{ ago(hs.t) }}</div>
        <div v-for="s in present" :key="s.key" class="tip-row">
          <svg width="14" height="6"><line x1="0" y1="3" x2="14" y2="3" :stroke="s.color" stroke-width="2" :stroke-dasharray="s.dash" /></svg>
          <span>{{ s.label }}</span>
          <b class="mono">{{ hs[s.key] != null ? ((hs[s.key] as number) / GB).toFixed(2) : '–' }} GB</b>
        </div>
      </div>
    </div>
  </figure>
</template>

<style scoped>
.chart {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
figcaption {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px 16px;
}
.title {
  font-weight: 650;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  font-size: 12px;
  color: var(--zen-muted);
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
  padding-bottom: 16px;
}
svg {
  display: block;
}
.plot > svg {
  width: 100%;
  height: 100%;
  touch-action: none;
}
.grid {
  stroke: var(--zen-border);
  vector-effect: non-scaling-stroke;
}
.cap {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 4;
  vector-effect: non-scaling-stroke;
}
.mk {
  stroke: color-mix(in srgb, var(--zen-muted) 55%, transparent);
  stroke-dasharray: 1 3;
  vector-effect: non-scaling-stroke;
}
.marker.error .mk {
  stroke: color-mix(in srgb, var(--zen-danger) 70%, transparent);
  stroke-dasharray: none;
}
.mk-hit {
  stroke: transparent;
  stroke-width: 9;
  vector-effect: non-scaling-stroke;
}
.marker:hover .mk {
  stroke: var(--zen-text);
}
.marker.error:hover .mk {
  stroke: var(--zen-danger);
}
.line {
  fill: none;
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
  stroke-linejoin: round;
}
.cross {
  stroke: var(--zen-muted);
  stroke-dasharray: 3 3;
  vector-effect: non-scaling-stroke;
}
.ytick,
.xtick {
  position: absolute;
  font: 10px var(--zen-mono);
  color: var(--zen-muted);
  pointer-events: none;
}
.ytick {
  left: 0;
  transform: translateY(-50%);
}
.xtick {
  bottom: 0;
  transform: translateX(-50%);
}
.tip {
  position: absolute;
  top: 6px;
  padding: 7px 9px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: 12px;
  white-space: nowrap;
  pointer-events: none;
  background: var(--zen-surface);
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  box-shadow: 0 6px 18px rgb(0 0 0 / 0.25);
}
.tip-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tip-row b {
  margin-left: auto;
  padding-left: 12px;
}
</style>
