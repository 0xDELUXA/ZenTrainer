<script setup lang="ts">
// The run form: presets on top (they layer), four numbered sections with the settings
// most runs touch, everything else under Advanced, and a sticky summary that says what
// you're about to do (and what looks off) next to the Start buttons. Model files, folders,
// samplers and schedulers come from ComfyUI's object_info; the rest is the harness's. The
// sections are in components/newrun/ and share the form through form.ts.
import { onMounted, onUnmounted, provide, ref } from 'vue'
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
const { imported, exportSettings, copySettings, importSettings } = form

// shared settings: a .zentrainer.json from Import, or its text pasted anywhere on the page
const picker = ref<HTMLInputElement | null>(null)
const copied = ref(false)
async function pick(e: Event) {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  input.value = ''
  if (f) await importSettings(await f.text())
}
async function copy() {
  copied.value = await copySettings()
  if (copied.value) setTimeout(() => (copied.value = false), 1500)
}
function onPaste(e: ClipboardEvent) {
  const t = e.target as HTMLElement | null
  if (resuming.value || (t && (t.closest('input, textarea, [contenteditable="true"]')))) return
  const text = e.clipboardData?.getData('text/plain')?.trim() ?? ''
  if (text.startsWith('{') && /"zentrainer-run"|"optim"\s*:/.test(text)) {
    e.preventDefault()
    void importSettings(text)
  }
}
onMounted(() => window.addEventListener('paste', onPaste))
onUnmounted(() => window.removeEventListener('paste', onPaste))
const missing = () => imported.value?.files.filter((f) => f.installed === false) ?? []
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
      <span class="share">
        <button class="preset ghost" title="Download these settings as a file to share (without your dataset or folders)" @click="exportSettings">
          <i class="mdi mdi-export-variant" /> Export
        </button>
        <button class="preset ghost" title="Copy these settings as text, to paste into a chat" @click="copy">
          <i :class="copied ? 'mdi mdi-check' : 'mdi mdi-content-copy'" /> {{ copied ? 'Copied' : 'Copy' }}
        </button>
        <button v-if="!resuming" class="preset ghost" title="Load settings someone shared: a .zentrainer.json file (or paste its text anywhere here)" @click="picker?.click()">
          <i class="mdi mdi-import" /> Import
        </button>
        <input ref="picker" type="file" accept=".json,application/json" hidden @change="pick" />
      </span>
    </div>

    <div v-if="imported" class="card imported small">
      <button class="close" title="Dismiss" @click="imported = null"><i class="mdi mdi-close" /></button>
      <div><i class="mdi mdi-import" /> Loaded <b>{{ imported.name || 'shared settings' }}</b>. Choose {{ imported.local.join(', ') }} here.</div>
      <div v-if="missing().length" class="warn">
        <i class="mdi mdi-alert-outline" /> Not installed here:
        <span v-for="f in missing()" :key="f.field" class="file">{{ f.name }} <span class="muted">({{ f.folder }})</span></span>
      </div>
      <div v-for="w in imported.warnings" :key="w" class="warn"><i class="mdi mdi-information-outline" /> {{ w }}</div>
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
.share {
  display: contents;
}
.imported {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 12px;
  padding-right: 32px;
}
.imported .close {
  position: absolute;
  top: 6px;
  right: 6px;
  background: none;
  border: 0;
  color: inherit;
  cursor: pointer;
  opacity: 0.6;
}
.imported .warn {
  color: var(--zen-warn);
}
.imported .file {
  margin-left: 8px;
  font-family: var(--zen-mono);
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
