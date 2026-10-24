<script setup lang="ts">
// The run's samples card: a fixed header (what was sampled last, the view switch, Sample now)
// over one of three views: the latest render of each prompt, each prompt across training
// (a prompt × step grid), and Manual, prompts tried apart from the run's own.
import { ZenLightbox, ZenSlider } from '@nynxz/zenkit-ui'
import { computed, onUnmounted, provide, ref, watch } from 'vue'
import { useRunContext } from './context'
import { SAMPLES, useSamples } from './samples'
import SamplesLatest from './SamplesLatest.vue'
import SamplesManual from './SamplesManual.vue'
import SamplesProgression from './SamplesProgression.vue'

const run = useRunContext()
const { detail, busy, sampling, status, running, sampleCkpt, sampleBusy, newestCkpt, sampleFinished, nextSample, send } = run
const samples = useSamples(run)
provide(SAMPLES, samples)
const { steps, latest, promptCount, liveManual, lb, hasInputs, compare } = samples

const sampleView = ref<'latest' | 'grid' | 'manual'>('latest')
const manualCount = computed(() => {
  const tries = new Set<number>()
  for (const x of detail.value?.samples ?? []) if (x.manual) tries.add(x.index)
  return tries.size
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
</script>

<template>
  <section v-if="promptCount && detail" ref="samplesCard" class="card">
    <!-- fixed header: image size (both views), the view switch and Sample now -->
    <header class="sec-head samples-head">
      <span class="sec-title">Samples</span>
      <span class="sec-sub">{{ samplesLine }}</span>
      <span class="spacer" />
      <!-- Progression only; the slot stays so the header never shifts -->
      <label class="zoom" :class="{ off: sampleView !== 'grid' }" title="Thumbnail size">
        <i class="mdi mdi-image-size-select-small" /><ZenSlider :model-value="gridSize" :min="64" :max="260" :step="4" @update:model-value="thumbPick = $event" />
      </label>
      <button
        v-if="hasInputs && sampleView !== 'manual'"
        class="chip-btn"
        :class="{ on: compare }"
        :title="compare ? 'Showing each edit beside its input picture: hover to wipe between them' : 'Show each edit beside its input picture'"
        @click="compare = !compare"
      >
        <i class="mdi mdi-compare-horizontal" /> Input
      </button>
      <div class="seg" role="tablist">
        <button :class="{ on: sampleView === 'latest' }" @click="sampleView = 'latest'">Latest</button>
        <button :class="{ on: sampleView === 'grid' }" :disabled="!steps.length" title="Each prompt across training: in the viewer ←→ steps, ↑↓ prompts (or WASD)" @click="sampleView = 'grid'">Progression</button>
        <button class="manual-tab" :class="{ on: sampleView === 'manual' }" title="Try any prompt at the current step, apart from the run's samples" @click="sampleView = 'manual'">Manual<span v-if="manualCount" class="muted"> {{ manualCount }}</span><i v-if="liveManual" class="mdi mdi-circle live-dot" title="Rendering a try" /></button>
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
    <SamplesManual v-if="sampleView === 'manual'" />
    <SamplesLatest v-else-if="sampleView === 'latest'" />
    <SamplesProgression v-else :size="gridSize" />
  </section>
  <ZenLightbox v-if="lb" v-model:index="lb.index" :items="lb.items" :grid="lb.grid" @close="lb = null" />
</template>

<style scoped src="./section.css"></style>
<style scoped>
.manual-tab {
  position: relative;
}
/* over the tab's corner, so it takes no room: no gap while idle, no jump when a render starts */
.live-dot {
  position: absolute;
  top: 1px;
  right: 2px;
  color: var(--zen-accent);
  font-size: 7px;
  line-height: 1;
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
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
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
.zoom.off {
  visibility: hidden;
}
</style>
