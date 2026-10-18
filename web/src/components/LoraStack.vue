<script setup lang="ts">
// A list of LoRAs with strengths, after ComfyUI-NynxzNodes' LoraStack widget: ZenKit's
// LoraPicker per row (search, folders, thumbnails, bookmarks), a strength, an on/off dot
// (off rows stay in the list but aren't applied), and ZenKit's LoraBrowser to pick from the
// whole library.
import { LoraBrowser, LoraPicker, ZenDot, ZenNumber } from '@nynxz/zenkit-ui'
import { ref } from 'vue'

export interface LoraRow {
  name: string
  strength: number
  on?: boolean
}
const rows = defineModel<LoraRow[]>({ default: () => [] })

const browse = ref(false)
const target = ref(-1) // the row the browser picks for; -1 = a new row
const set = (i: number, patch: Partial<LoraRow>) => (rows.value = rows.value.map((r, k) => (k === i ? { ...r, ...patch } : r)))
const remove = (i: number) => (rows.value = rows.value.filter((_, k) => k !== i))
const add = (name = '') => (rows.value = [...rows.value, { name, strength: 1, on: true }])
function openBrowse(i: number) {
  target.value = i
  browse.value = true
}
function picked(name: string) {
  if (target.value < 0) add(name)
  else set(target.value, { name })
}
</script>

<template>
  <div class="stack">
    <div v-for="(r, i) in rows" :key="i" class="row" :class="{ off: r.on === false }">
      <ZenDot :model-value="r.on !== false" title="Use this LoRA" @update:model-value="set(i, { on: $event })" />
      <LoraPicker class="pick" :model-value="r.name" placeholder="Choose a LoRA…" @update:model-value="set(i, { name: $event })" @browse="openBrowse(i)" />
      <ZenNumber class="str" :model-value="r.strength" :min="-2" :max="2" :step="0.05" :precision="2" @update:model-value="set(i, { strength: $event })" />
      <button type="button" class="icon-btn" title="Remove" @click="remove(i)"><i class="mdi mdi-close" /></button>
    </div>
    <div class="foot">
      <button type="button" class="chip-btn" @click="add()"><i class="mdi mdi-plus" /> Add LoRA</button>
      <button type="button" class="icon-btn" title="Browse the library" @click="openBrowse(-1)"><i class="mdi mdi-view-grid-outline" /></button>
    </div>
    <LoraBrowser v-model:open="browse" :selected="rows[target]?.name" @pick="picked" />
  </div>
</template>

<style scoped>
.stack {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
}
.row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) 88px 30px;
  gap: 6px;
  align-items: center;
}
.row.off .pick,
.row.off .str {
  opacity: 0.5;
}
.foot {
  display: flex;
  gap: 6px;
  align-items: center;
}
.chip-btn {
  height: 30px;
  padding: 0 10px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.chip-btn:hover {
  color: var(--zen-text);
}
.icon-btn {
  width: 30px;
  height: 30px;
  display: inline-grid;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.icon-btn:hover {
  color: var(--zen-text);
}
</style>
