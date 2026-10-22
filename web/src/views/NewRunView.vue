<script setup lang="ts">
// The run form: presets on top (they layer), four numbered sections with the settings
// most runs touch, everything else under Advanced, and a sticky summary that says what
// you're about to do (and what looks off) next to the Start buttons. Model files, folders,
// samplers and schedulers come from ComfyUI's object_info; the rest is the harness's. The
// sections are in components/newrun/ and share the form through form.ts.
import { provide } from 'vue'
import DatasetSection from '@/components/newrun/DatasetSection.vue'
import { RUN_FORM, useRunForm } from '@/components/newrun/form'
import MemorySection from '@/components/newrun/MemorySection.vue'
import ModelSection from '@/components/newrun/ModelSection.vue'
import RunSummary from '@/components/newrun/RunSummary.vue'
import SamplesSection from '@/components/newrun/SamplesSection.vue'
import TrainingSection from '@/components/newrun/TrainingSection.vue'

const form = useRunForm()
provide(RUN_FORM, form)
const { runPresets, applied, applyPreset, savePreset, editing, retrying, resuming, resumeStep } = form
</script>

<template>
  <div class="page wide">
    <div class="page-head">
      <div class="titles">
        <h1>{{ editing ? (resuming ? 'Edit & resume' : retrying ? 'Edit & retry' : 'Edit queued run') : 'New run' }}</h1>
        <p v-if="editing" class="lede">
          {{ resuming ? `carries on from step ${resumeStep} with the new settings` : retrying ? 'goes back to the front of the queue' : 'keeps its place in the queue' }}
        </p>
      </div>
    </div>

    <div class="presets">
      <span class="eyebrow">Start from</span>
      <button v-for="p in runPresets" :key="p.id" class="preset" :class="{ on: applied.includes(p.name) }" :title="p.description" @click="applyPreset(p)">
        <i v-if="!p.builtin" class="mdi mdi-account-outline" /> {{ p.name }}
      </button>
      <button class="preset ghost" title="Save the current settings (not the model or dataset) as a preset" @click="savePreset">
        <i class="mdi mdi-content-save-outline" /> Save as preset
      </button>
    </div>

    <div class="layout">
      <div class="form">
        <div v-if="resuming" class="card locked-note small">
          <i class="mdi mdi-lock-outline" /> It has trained, so its model, dataset and LoRA shape (algorithm, rank, layers, blocks) and
          optimizer type stay as they were. Memory, samples, learning rate, schedule and length can change.
        </div>
        <ModelSection />
        <DatasetSection />
        <TrainingSection />
        <SamplesSection />
        <MemorySection />
      </div>

      <RunSummary />
    </div>
  </div>
</template>

<style scoped>
.wide {
  max-width: 1280px;
}
.presets {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.preset {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 30px;
  padding: 0 12px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 80%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 50%, transparent);
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.preset:hover,
.preset.on {
  border-color: var(--zen-selected, var(--zen-accent));
}
.preset.on {
  background: var(--zen-selected-bg, color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 16%, transparent));
}
.preset.ghost {
  color: var(--zen-muted);
  border-style: dashed;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 18px;
  align-items: start;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.locked-note {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  color: var(--zen-muted);
  line-height: 1.5;
}
@media (max-width: 1000px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
