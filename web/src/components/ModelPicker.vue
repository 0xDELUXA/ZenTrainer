<script setup lang="ts">
// Pick a model the way ComfyUI loads it: one checkpoint file, or a diffusion model +
// text encoder + VAE. Choices come from the server's own loaders (object_info).
import { ZenCombo } from '@nynxz/zenkit-ui'
import { onMounted, reactive } from 'vue'
import { inputChoices } from '@/api/comfy'

export interface ModelValue {
  kind: 'checkpoint' | 'split'
  ckpt_name?: string | null
  unet_name?: string | null
  unet_weight_dtype?: string
  clip_name?: string | null
  clip_type?: string | null
  vae_name?: string | null
}

const model = defineModel<ModelValue>({ required: true })
const choices = reactive<Record<string, string[]>>({})
const items = (v?: string[]) => (v ?? []).map((x) => ({ value: x, label: x }))

onMounted(async () => {
  const want: [string, string, string][] = [
    ['ckpt', 'CheckpointLoaderSimple', 'ckpt_name'],
    ['unet', 'UNETLoader', 'unet_name'],
    ['dtype', 'UNETLoader', 'weight_dtype'],
    ['clip', 'CLIPLoader', 'clip_name'],
    ['type', 'CLIPLoader', 'type'],
    ['vae', 'VAELoader', 'vae_name'],
  ]
  await Promise.all(want.map(async ([k, n, i]) => (choices[k] = await inputChoices(n, i))))
})
const set = (key: keyof ModelValue, v: string | number) => (model.value = { ...model.value, [key]: String(v) })
</script>

<template>
  <div class="mp">
    <div class="seg">
      <button :class="{ on: model.kind === 'checkpoint' }" @click="model = { ...model, kind: 'checkpoint' }">Checkpoint</button>
      <button :class="{ on: model.kind === 'split' }" @click="model = { ...model, kind: 'split' }">Separate parts</button>
    </div>
    <template v-if="model.kind === 'checkpoint'">
      <ZenCombo :model-value="model.ckpt_name ?? ''" :items="items(choices.ckpt)" placeholder="Search checkpoints…" :item-height="30" @update:model-value="set('ckpt_name', $event)" />
    </template>
    <template v-else>
      <ZenCombo :model-value="model.unet_name ?? ''" :items="items(choices.unet)" placeholder="Diffusion model…" :item-height="30" @update:model-value="set('unet_name', $event)" />
      <div class="two">
        <ZenCombo :model-value="model.clip_name ?? ''" :items="items(choices.clip)" placeholder="Text encoder…" :item-height="30" @update:model-value="set('clip_name', $event)" />
        <ZenCombo :model-value="model.clip_type ?? ''" :items="items(choices.type)" placeholder="Family…" @update:model-value="set('clip_type', $event)" />
      </div>
      <div class="two">
        <ZenCombo :model-value="model.vae_name ?? ''" :items="items(choices.vae)" placeholder="VAE…" :item-height="30" @update:model-value="set('vae_name', $event)" />
        <ZenCombo :model-value="model.unet_weight_dtype ?? 'default'" :items="items(choices.dtype ?? ['default'])" :searchable="false" @update:model-value="set('unet_weight_dtype', $event)" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.mp {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.two {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 6px;
}
.seg {
  display: inline-flex;
  align-self: flex-start;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.seg button.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
</style>
