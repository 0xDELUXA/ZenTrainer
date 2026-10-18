<script setup lang="ts">
// A dataset's five health scores (0-1, from its last stats snapshot) as a pentagon: a quick
// signature for a card, not a reading instrument; the stats panel carries the real numbers.
// One series, the theme's accent; the grid stays recessive. Axes keep a fixed order so shapes
// compare across datasets.
import { computed } from 'vue'

const props = withDefaults(defineProps<{ scores: Record<string, number>; size?: number; labels?: boolean }>(), {
  size: 40,
  labels: false,
})

const AXES: [string, string][] = [
  ['captions', 'Captions'],
  ['consistency', 'Consistency'],
  ['quality', 'Quality'],
  ['uniqueness', 'Uniqueness'],
  ['faces', 'Faces'],
]

const axes = computed<[string, string][]>(() =>
  AXES.map(([k, label]): [string, string] =>
    k === 'faces' && props.scores.faces == null
      ? ['balance', 'Balance']
      : k === 'quality' && props.scores.quality == null
        ? ['resolution', 'Resolution'] // snapshots from before the quality score
        : [k, label],
  ),
)
const pad = computed(() => (props.labels ? 30 : 3))
const r = computed(() => props.size / 2 - pad.value)
const c = computed(() => props.size / 2)
const at = (i: number, v: number): [number, number] => {
  const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5
  return [c.value + Math.cos(a) * r.value * v, c.value + Math.sin(a) * r.value * v]
}
const ring = (v: number) => axes.value.map((_, i) => at(i, v).join(',')).join(' ')
const shape = computed(() => axes.value.map(([k], i) => at(i, Math.max(0.04, props.scores[k] ?? 0)).join(',')).join(' '))
const tip = computed(() => axes.value.map(([k, label]) => `${label} ${Math.round((props.scores[k] ?? 0) * 100)}`).join(' · '))
</script>

<template>
  <svg class="pent" :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" role="img" :aria-label="`Dataset health: ${tip}`">
    <title>{{ tip }}</title>
    <polygon v-for="v in [1, 0.5]" :key="v" :points="ring(v)" class="grid" />
    <line v-for="(_, i) in axes" :key="i" :x1="c" :y1="c" :x2="at(i, 1)[0]" :y2="at(i, 1)[1]" class="grid" />
    <polygon :points="shape" class="area" />
    <template v-if="labels">
      <text
        v-for="([k, label], i) in axes"
        :key="k"
        :x="at(i, 1.18)[0]"
        :y="at(i, 1.18)[1]"
        class="lbl"
        :text-anchor="Math.abs(at(i, 1)[0] - c) < 2 ? 'middle' : at(i, 1)[0] > c ? 'start' : 'end'"
        dominant-baseline="middle"
      >
        <tspan :x="at(i, 1.18)[0]" dy="-0.55em">{{ label }}</tspan>
        <tspan :x="at(i, 1.18)[0]" dy="1.2em" class="val">{{ Math.round((scores[k] ?? 0) * 100) }}</tspan>
      </text>
    </template>
  </svg>
</template>

<style scoped>
.pent {
  display: block;
  overflow: visible;
}
.grid {
  fill: none;
  stroke: color-mix(in srgb, var(--zen-text) 14%, transparent);
  stroke-width: 1;
}
.area {
  fill: color-mix(in srgb, var(--zen-accent) 24%, transparent);
  stroke: var(--zen-accent);
  stroke-width: 2;
  stroke-linejoin: round;
}
.lbl {
  font-size: 10px;
  fill: var(--zen-muted);
}
.lbl .val {
  fill: var(--zen-text);
  font-weight: 600;
}
</style>
