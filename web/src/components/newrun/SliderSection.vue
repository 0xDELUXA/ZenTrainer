<script setup lang="ts">
// Slider: a LoRA whose strength moves pictures along one attribute (engine harness/slider.py). Only
// here once added from Model's "Add" menu, under the dataset whose picture sets it pairs. From image pairs (the dataset's pictures one end, a
// control set of the same names the other) or from two prompts.
import { ZenCombo, ZenField, ZenInput, ZenNumber } from '@nynxz/zenkit-ui'
import { computed, watchEffect } from 'vue'
import { useRunFormContext } from './form'

const { spec, resuming, controlCounts } = useRunFormContext()

const pairSets = computed(() =>
  Object.entries(controlCounts.value).map(([name, n]) => ({ value: name, label: `${name} · ${n} picture${n === 1 ? '' : 's'}`, keywords: name })),
)
const pairs = computed(() => controlCounts.value[spec.slider.pair_set] ?? 0)
// a dataset with one picture set beside it: that's the other end
watchEffect(() => {
  const sets = Object.keys(controlCounts.value)
  if (spec.slider.kind === 'pairs' && !spec.slider.pair_set && sets.length === 1) spec.slider.pair_set = sets[0]!
})
const ownEnd = computed({
  get: () => spec.slider.dataset_end,
  set: (v: 'positive' | 'negative') => (spec.slider.dataset_end = v),
})
// one anchor per line
const anchors = computed({
  get: () => spec.slider.anchors.join('\n'),
  set: (v: string) => (spec.slider.anchors = v.split('\n').map((x) => x.trim()).filter(Boolean)),
})
function remove() {
  spec.slider.kind = ''
}
</script>

<template>
<section v-if="spec.slider.kind" class="card sec slider">
  <div class="head-row">
    <h2><i class="mdi mdi-tune-vertical-variant" /> Slider</h2>
    <div class="seg" role="group" aria-label="Learns from">
      <button :class="{ on: spec.slider.kind === 'pairs' }" :disabled="resuming" @click="spec.slider.kind = 'pairs'">Image pairs</button>
      <button :class="{ on: spec.slider.kind === 'text' }" :disabled="resuming" @click="spec.slider.kind = 'text'">Prompts</button>
    </div>
    <span class="grow" />
    <button v-if="!resuming" class="remove small" title="Train a normal LoRA instead" @click="remove"><i class="mdi mdi-close" /> Remove</button>
  </div>
  <p class="muted small lede">
    The LoRA's strength becomes a slider: +1 one way, −1 the other, 0 the model as it is. Samples show each prompt at −1 … +1.
  </p>

  <div v-if="spec.slider.kind === 'pairs'" class="grid-2">
    <ZenField label="The other end" stack tip="A set of pictures beside the dataset (its _controls/<set>/ folder) with the same file names: each the same picture as its partner, differing only in what the slider moves">
      <ZenCombo v-if="pairSets.length" v-model="spec.slider.pair_set" :items="pairSets" :sorted="false" placeholder="Pick its set…" />
      <span v-else class="muted small">This dataset has no picture sets beside it yet.</span>
    </ZenField>
    <ZenField label="The dataset's own pictures are" stack tip="Which end of the slider your dataset's pictures are; their partners are the other">
      <div class="seg">
        <button :class="{ on: ownEnd === 'positive' }" @click="ownEnd = 'positive'">+1</button>
        <button :class="{ on: ownEnd === 'negative' }" @click="ownEnd = 'negative'">−1</button>
      </div>
    </ZenField>
    <p class="muted small wide-f note">
      <template v-if="spec.slider.pair_set && pairs">{{ pairs }} pair{{ pairs === 1 ? '' : 's' }}: pictures without a partner are left out. </template>
      Put each picture's partner in <span class="mono">{{ spec.dataset.folder || 'the dataset' }}/_controls/&lt;set&gt;/</span> under the same file name: the same shot,
      differing only in what the slider moves (smiling and not, warm and cool light). Captions are the pair's. A step costs about two normal ones.
    </p>
  </div>

  <div v-else class="grid-2">
    <ZenField label="Positive (+1)" stack tip="What raising the slider adds, written as a whole prompt: “a photo of a person, old, wrinkled skin, grey hair”">
      <ZenInput v-model="spec.slider.positive" placeholder="a photo of a person, old, wrinkled skin" />
    </ZenField>
    <ZenField label="Negative (−1)" stack tip="What lowering it adds: “a photo of a person, young, smooth skin”">
      <ZenInput v-model="spec.slider.negative" placeholder="a photo of a person, young, smooth skin" />
    </ZenField>
    <ZenField label="Neutral" stack tip="The subject without either end (“a photo of a person”): what the LoRA trains on. Empty works too">
      <ZenInput v-model="spec.slider.neutral" placeholder="a photo of a person" />
    </ZenField>
    <ZenField label="Strength of the push" stack tip="guidance: how far past the base model's own difference between the two prompts each end goes. 3 is usual; lower is subtler and drifts less">
      <ZenNumber v-model="spec.slider.guidance" :min="0.5" :max="20" :step="0.5" :precision="1" />
    </ZenField>
    <ZenField label="Keep these the same (anchors)" stack class="wide-f" tip="Prompts the slider should leave alone, one per line (“a landscape”, “a car”): at either end their pictures are held to the base model's. Each adds a forward pass to a step">
      <textarea v-model="anchors" class="anchors" rows="2" placeholder="a photo of a landscape" />
    </ZenField>
    <p class="muted small wide-f note">From the model's own pictures of the two prompts: the dataset's pictures are only where the noise starts (their captions aren't used). A step costs about three normal ones.</p>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.head-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.head-row h2 {
  margin: 0;
}
.grow {
  flex: 1;
}
.lede {
  margin: 6px 0 12px;
}
.remove {
  background: none;
  border: 1px solid var(--zen-border);
  border-radius: var(--zen-radius);
  color: var(--zen-muted);
  padding: 3px 8px;
  cursor: pointer;
}
.remove:hover {
  color: var(--zen-danger);
  border-color: var(--zen-danger);
}
.anchors {
  width: 100%;
  resize: vertical;
  font: inherit;
  color: var(--zen-text);
  background: var(--zen-field-bg);
  border: 1px solid var(--zen-border);
  border-radius: var(--zen-radius);
  padding: 6px 10px;
}
.note {
  margin: 0;
}
</style>
