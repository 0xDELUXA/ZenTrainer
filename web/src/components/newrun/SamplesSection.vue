<script setup lang="ts">
// 4 Samples: the prompts (and saved prompt sets), how often they render, with which model, and the
// sampler's settings.
import { ZenButton, ZenCombo, ZenField, ZenInput, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, nextTick, reactive, ref, watch } from 'vue'
import { engine, thumbUrl } from '@/api/engine'
import SamplePromptList from '@/components/SamplePromptList.vue'
import { downloadFor, fmtBytes } from '@/stores/downloads'
import { items, merge, useRunFormContext } from './form'

const { spec, choices, sampleSuggestion, promptSets, pairItems, runInputs, isVideoModel } = useRunFormContext()

const advanced = reactive({ samples: false })
const usingSuggestion = computed(
  () => !!sampleSuggestion.value && spec.samples.model_name === sampleSuggestion.value.installed && spec.samples.steps === sampleSuggestion.value.steps,
)
function useSuggestion() {
  const s = sampleSuggestion.value
  if (!s?.installed) return
  Object.assign(spec.samples, { model_name: s.installed, steps: s.steps, cfg: s.cfg, sampler: s.sampler, scheduler: s.scheduler })
}
const slgLayersText = computed({
  get: () => spec.samples.slg_layers.join(', '),
  set: (v: string) => (spec.samples.slg_layers = (v.match(/\d+/g) ?? []).map(Number)),
})
const sampleModel = computed({
  get: () => spec.samples.model_name ?? '',
  set: (v: string) => (spec.samples.model_name = v || null),
})
const insertTrigger = () => {
  spec.samples.prompts = [...spec.samples.prompts, '[trigger], ']
}
// edit pairs: a sample's input picture comes from the dataset's pairs
const controlOptions = computed(() =>
  pairItems.value.flatMap((it) => runInputs(it).map((c) => ({ value: c, thumb: thumbUrl(spec.dataset.folder, c, 128) }))),
)
function samplesFromPairs() {
  spec.samples.prompts = pairItems.value.slice(0, 2).map((it) => ({ prompt: it.caption, control: runInputs(it)[0] }))
}

// ---- prompt sets ---------------------------------------------------------------------------
// Picking a set loads it into the form to edit; once edited it shows as modified and can be
// saved back over itself (a built-in gets a user copy that takes its place), saved as a new
// set, or reverted.
const SET_KEYS = ['prompts', 'negative', 'width', 'height', 'steps', 'cfg', 'sampler', 'scheduler'] as const
const promptSet = ref<string>('')
const currentSet = computed(() => promptSets.value.find((p) => p.id === promptSet.value) ?? null)
const loadSet = () => currentSet.value && merge(spec.samples as unknown as Record<string, unknown>, currentSet.value.data as Record<string, unknown>)
watch(promptSet, loadSet)
const setModified = computed(() => {
  const set = currentSet.value
  if (!set) return false
  const d = set.data as Record<string, unknown>
  const now = spec.samples as unknown as Record<string, unknown>
  return Object.keys(d).some((k) => JSON.stringify(d[k]) !== JSON.stringify(now[k]))
})
const naming = ref(false)
const newSetName = ref('')
const savingSet = ref(false)
const nameBox = ref<HTMLElement | null>(null)
async function startNaming() {
  newSetName.value = currentSet.value ? `${currentSet.value.name} (copy)` : ''
  naming.value = true
  await nextTick()
  nameBox.value?.querySelector('input')?.select()
}
async function savePromptSet(name: string) {
  if (!name.trim()) return
  savingSet.value = true
  try {
    const data = JSON.parse(JSON.stringify(Object.fromEntries(SET_KEYS.map((k) => [k, spec.samples[k]])))) // reactive arrays don't structuredClone
    const saved = await engine.savePreset('prompts', name.trim(), data)
    const row = saved.preset as (typeof promptSets.value)[number]
    promptSets.value = [...promptSets.value.filter((p) => p.id !== row.id), row]
    promptSet.value = row.id
    naming.value = false
  } finally {
    savingSet.value = false
  }
}
</script>

<template>
<section class="card sec">
  <h2><span class="num">4</span> Samples</h2>
  <!-- one fixed-height row: the set picker and its actions, or the name field when saving -->
  <div v-if="naming" class="sample-head" @keydown.enter.prevent="savePromptSet(newSetName)" @keydown.esc="naming = false">
    <div ref="nameBox"><ZenInput v-model="newSetName" size="sm" placeholder="Name this prompt set" /></div>
    <ZenButton size="sm" variant="primary" icon="mdi mdi-content-save-outline" :disabled="!newSetName.trim() || savingSet" @click="savePromptSet(newSetName)">Save</ZenButton>
    <ZenButton size="sm" @click="naming = false">Cancel</ZenButton>
  </div>
  <div v-else class="sample-head">
    <div class="set-pick">
      <ZenCombo
        v-model="promptSet"
        :items="promptSets.map((p) => ({ value: p.id, label: p.name }))"
        placeholder="Saved prompt sets…"
        empty-text="No saved sets yet"
        size="sm"
      />
      <span v-if="setModified" class="set-dot" title="Edited since it was loaded" />
    </div>
    <template v-if="currentSet && setModified">
      <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="savingSet" :title="currentSet.builtin ? 'Save your edits as your own version of this built-in set' : `Overwrite “${currentSet.name}”`" @click="savePromptSet(currentSet.name)">Save</ZenButton>
      <ZenButton size="sm" icon="mdi mdi-content-save-plus-outline" @click="startNaming">Save as new</ZenButton>
      <ZenButton size="sm" icon="mdi mdi-undo-variant" title="Back to the saved prompts" @click="loadSet">Revert</ZenButton>
    </template>
    <ZenButton v-else-if="!currentSet" size="sm" icon="mdi mdi-content-save-outline" :disabled="!spec.samples.prompts.length" @click="startNaming">Save as set</ZenButton>
    <span class="spacer" />
    <ZenButton v-if="spec.dataset.edit && pairItems.length" size="sm" icon="mdi mdi-swap-horizontal" title="Sample two of the dataset's pairs: their instruction on their input picture" @click="samplesFromPairs">From pairs</ZenButton>
    <ZenButton size="sm" icon="mdi mdi-tag-plus-outline" @click="insertTrigger">Insert [trigger]</ZenButton>
  </div>
  <SamplePromptList
    v-model="spec.samples.prompts"
    :width="spec.samples.width"
    :height="spec.samples.height"
    :frames="spec.samples.frames ?? 1"
    :video="isVideoModel"
    :trigger="spec.dataset.trigger"
    :controls="spec.dataset.edit ? controlOptions : undefined"
  />
  <div v-if="sampleSuggestion && !usingSuggestion" class="suggest">
    <i class="mdi mdi-lightning-bolt-outline" />
    <span>
      Preview faster on <b>{{ sampleSuggestion.model }}</b>: {{ sampleSuggestion.steps }} steps, CFG {{ sampleSuggestion.cfg }},
      {{ sampleSuggestion.sampler }} / {{ sampleSuggestion.scheduler }}, the settings ComfyUI's template uses with it.
    </span>
    <button v-if="sampleSuggestion.installed" class="chip-btn" @click="useSuggestion">Use</button>
    <template v-else-if="['queued', 'downloading'].includes(downloadFor(sampleSuggestion.folder, sampleSuggestion.model)?.status ?? '')">
      <span class="mono small">{{ Math.floor(((downloadFor(sampleSuggestion.folder, sampleSuggestion.model)!.done) / (downloadFor(sampleSuggestion.folder, sampleSuggestion.model)!.total || 1)) * 100) }}%</span>
    </template>
    <button v-else class="chip-btn" @click="engine.download(sampleSuggestion.family, sampleSuggestion.folder, sampleSuggestion.model)">
      <i class="mdi mdi-download" /> {{ fmtBytes(sampleSuggestion.size) || 'Download' }}
    </button>
  </div>
  <div class="grid-2">
    <ZenField label="Every" stack tip="Steps between samples; 0 = off (only Sample now). The save interval puts samples, and a likeness score, on every checkpoint">
      <ZenNumber v-model="spec.samples.every" :min="0" :step="50" :precision="0" />
    </ZenField>
    <ZenField label="Baseline before training" stack tip="Render once with the untouched model"><div class="sw-row"><ZenSwitch v-model="spec.samples.baseline" /><span class="sw-state">{{ spec.samples.baseline ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField label="Sample with another model" stack tip="E.g. train on raw, preview on turbo (same family)">
      <ZenCombo
        v-model="sampleModel"
        :items="[{ value: '', label: 'The model being trained' }, ...items(spec.model.kind === 'split' ? choices.unet : choices.ckpt)]"
        :item-height="30"
      />
    </ZenField>
    <ZenField v-if="spec.samples.model_name" label="LoRA strength" stack>
      <ZenNumber v-model="spec.samples.lora_strength" :min="0" :max="2" :step="0.05" :precision="2" />
    </ZenField>
  </div>
  <button class="more" @click="advanced.samples = !advanced.samples">
    <i :class="advanced.samples ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Sampler settings
  </button>
  <div v-if="advanced.samples" class="grid-2">
    <ZenField label="Size" stack>
      <div class="inline">
        <ZenNumber v-model="spec.samples.width" :min="256" :step="64" :precision="0" />
        <span class="muted">×</span>
        <ZenNumber v-model="spec.samples.height" :min="256" :step="64" :precision="0" />
      </div>
    </ZenField>
    <ZenField v-if="isVideoModel" label="Frames / fps" stack tip="Video samples: frames per sample (snapped to a length the VAE decodes; 1 = a still) and the frame rate they're saved at">
      <div class="inline">
        <ZenNumber :model-value="spec.samples.frames ?? 1" :min="1" :max="361" :step="17" :precision="0" @update:model-value="spec.samples.frames = $event" />
        <ZenNumber :model-value="spec.samples.fps ?? 24" :min="1" :max="60" :step="1" :precision="0" @update:model-value="spec.samples.fps = $event" />
      </div>
    </ZenField>
    <ZenField label="Turbo LoRA" stack tip="Rides under every sample (never training or the saved LoRA) so previews take a few steps: MiniMax H3's minimax_h3_turbo_v4_step600_ema at 0.75, 6 steps, CFG 1">
      <div class="inline">
        <ZenCombo :model-value="spec.samples.turbo_lora ?? ''" :items="[{ value: '', label: 'None' }, ...items(choices.loras)]" :item-height="30" @update:model-value="spec.samples.turbo_lora = String($event) || null" />
        <ZenNumber v-if="spec.samples.turbo_lora" :model-value="spec.samples.turbo_strength ?? 0.75" :min="0" :max="1.5" :step="0.05" :precision="2" @update:model-value="spec.samples.turbo_strength = $event" />
      </div>
    </ZenField>
    <ZenField label="Steps / CFG" stack>
      <div class="inline">
        <ZenNumber v-model="spec.samples.steps" :min="1" :step="1" :precision="0" />
        <ZenNumber v-model="spec.samples.cfg" :min="0" :step="0.5" :precision="1" />
      </div>
    </ZenField>
    <ZenField label="Sampler" stack><ZenCombo v-model="spec.samples.sampler" :items="items(choices.sampler ?? ['euler'])" /></ZenField>
    <ZenField label="Scheduler" stack><ZenCombo v-model="spec.samples.scheduler" :items="items(choices.scheduler ?? ['normal'])" /></ZenField>
    <ZenField label="Seed" stack><ZenNumber v-model="spec.samples.seed" :min="0" :step="1" :precision="0" /></ZenField>
    <ZenField label="Skip-layer guidance" stack tip="Negative pass: skip layers in the unconditional pass (free). Extra pass: one more forward per step">
      <div class="seg">
        <button v-for="m in (['off', 'uncond', 'standard'] as const)" :key="m" :class="{ on: spec.samples.slg_mode === m }" @click="spec.samples.slg_mode = m">
          {{ { off: 'Off', uncond: 'Negative pass', standard: 'Extra pass' }[m] }}
        </button>
      </div>
    </ZenField>
    <ZenField v-if="spec.samples.slg_mode !== 'off'" label="Skipped layers" stack tip="Block indices, from 0">
      <ZenInput v-model="slgLayersText" placeholder="22" />
    </ZenField>
    <template v-if="spec.samples.slg_mode === 'standard'">
      <ZenField label="SLG scale" stack><ZenNumber v-model="spec.samples.slg_scale" :min="0" :step="0.5" :precision="1" /></ZenField>
      <ZenField label="SLG range" stack tip="Fraction of the steps it runs over">
        <div class="inline">
          <ZenNumber v-model="spec.samples.slg_start" :min="0" :max="1" :step="0.01" :precision="2" />
          <ZenNumber v-model="spec.samples.slg_end" :min="0" :max="1" :step="0.01" :precision="2" />
        </div>
      </ZenField>
    </template>
    <ZenField label="Negative" stack class="wide-f"><ZenInput v-model="spec.samples.negative" /></ZenField>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.sample-head {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.sample-head {
  min-height: 32px;
}
.sample-head > :first-child {
  width: 240px;
}
.set-pick {
  position: relative;
}
.set-dot {
  position: absolute;
  top: 50%;
  right: 30px;
  width: 7px;
  height: 7px;
  margin-top: -3.5px;
  border-radius: var(--r-sm);
  background: var(--zen-warn);
  pointer-events: none;
}
.suggest {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: var(--r-lg);
  font-size: 12px;
  line-height: 1.45;
  background: color-mix(in srgb, var(--zen-accent) 9%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 30%, transparent);
}
.suggest > i {
  font-size: 18px;
  color: var(--zen-accent);
}
.suggest > span:not(.muted) {
  flex: 1;
}
</style>
