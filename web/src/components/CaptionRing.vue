<script setup lang="ts">
// A dataset at a glance: the ring is how much of it is captioned (full and green when every image
// has a caption, the share so far in the warning colour while some are missing), the pentagon
// inside it the health of its caption set (when it's been measured).
import { computed } from 'vue'
import HealthPentagon from '@/components/HealthPentagon.vue'

const props = withDefaults(defineProps<{ done: number; total: number; scores?: Record<string, number> | null; size?: number }>(), {
  scores: null,
  size: 18,
})
const SW = 2.5
const R = computed(() => props.size / 2 - SW / 2 - 0.5)
const C = computed(() => 2 * Math.PI * R.value)
const share = computed(() => (props.total ? Math.min(1, props.done / props.total) : 0))
const full = computed(() => props.total > 0 && props.done >= props.total)
</script>

<template>
  <span class="badge" :style="{ width: `${size}px`, height: `${size}px` }">
    <svg class="ring" :class="{ full }" :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" aria-hidden="true">
      <circle class="track" :cx="size / 2" :cy="size / 2" :r="R" :stroke-width="SW" />
      <circle class="fill" :cx="size / 2" :cy="size / 2" :r="R" :stroke-width="SW" :stroke-dasharray="`${share * C} ${C}`" />
    </svg>
    <HealthPentagon v-if="scores" class="pent" :scores="scores" :size="size - 2 * SW - 5" />
  </span>
</template>

<style scoped>
.badge {
  position: relative;
  display: inline-grid;
  place-items: center;
  flex: none;
}
.ring {
  position: absolute;
  inset: 0;
  transform: rotate(-90deg);
}
circle {
  fill: none;
}
.track {
  stroke: var(--zen-surface-2);
}
.fill {
  stroke: var(--zen-warn);
  stroke-linecap: round;
}
.full .fill {
  stroke: var(--zen-ok, var(--zen-accent));
}
.pent {
  position: relative;
}
</style>
