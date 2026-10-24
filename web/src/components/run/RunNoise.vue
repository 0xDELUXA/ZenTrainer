<script setup lang="ts">
// Noise levels: where this run's steps actually landed on the schedule (each step logs its
// samples' σ and plain loss), against where its settings aim them, with each sample's loss as a
// dot and their average per stretch: which noise levels the LoRA finds hard, and whether the
// timesteps went where they were meant to. Flow models (σ 0..1) only; older runs logged nothing.
import { computed, ref, watch } from 'vue'
import { setSetting, settings } from '@/stores/settings'
import { engine, type NoiseProfile } from '@/api/engine'
import NoiseChart, { type NoiseSeries } from '@/components/NoiseChart.vue'
import { draws, FLEX_WEIGHTS, histogram, useSchedule } from '@/lib/timesteps'
import { useRunContext } from './context'

const { detail } = useRunContext()

const points = computed((): [number, number][] =>
  (detail.value?.metrics ?? []).flatMap((r) => (r.drawn ?? []).map(([, s, l]) => [s, l] as [number, number])),
)
const flow = computed(() => points.value.length > 0 && points.value.every(([s]) => s <= 1.0001))

const profile = ref<NoiseProfile>({ known: false })
const model = computed(() => detail.value?.spec.model)
watch(
  () => [model.value?.kind, model.value?.kind === 'checkpoint' ? model.value?.ckpt_name : model.value?.unet_name] as const,
  async ([kind, name]) => {
    profile.value = { known: false }
    if (!name) return
    try {
      profile.value = await engine.noise(kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models', name)
    } catch {
      /* no profile: the actual steps alone */
    }
  },
  { immediate: true },
)

const exact = useSchedule(profile, () => detail.value?.spec.objective)
const series = computed((): NoiseSeries[] => {
  const out: NoiseSeries[] = [{ label: `Trained (${points.value.length})`, hist: histogram(points.value.map(([s]) => [s, 1 / points.value.length])), kind: 'area' }]
  const o = detail.value?.spec.objective
  if (o && profile.value.known && profile.value.flow) out.push({ label: 'As set', hist: histogram(draws(o, profile.value, 20000, exact.value)), kind: 'line' })
  return out
})
// folded by default (it redraws with every step), and remembered once opened
const open = computed({ get: () => settings.noiseChartOpen, set: (v: boolean) => setSetting('noiseChartOpen', v) })
</script>

<template>
  <section v-if="flow" class="card noise">
    <button class="head" @click="open = !open">
      <span class="title">Noise levels</span>
      <span class="muted small">where the steps landed, and what each lost</span>
      <span class="spacer" />
      <i :class="open ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
    </button>
    <NoiseChart v-if="open" :series="series" :points="points" :weights="detail?.spec.objective.weighting === 'flex' ? FLEX_WEIGHTS : null" :height="150" />
  </section>
</template>

<style scoped>
.noise {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 0;
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  cursor: pointer;
  text-align: left;
}
.title {
  font-weight: 600;
}
.spacer {
  flex: 1;
}
</style>
