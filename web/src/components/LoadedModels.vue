<script setup lang="ts">
// The models ComfyUI holds right now and how much of each sits in VRAM: one cell per
// transformer block (blocks.N, double_blocks.N, ...), filled by its share on the GPU. With
// offloading you see blocks stream in and out. Polls while shown.
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { engine, type LoadedModel } from '@/api/engine'
import { gb } from '@/system'

const models = ref<LoadedModel[]>([])
const error = ref<string | null>(null)
let timer: ReturnType<typeof setTimeout> | undefined
async function poll() {
  try {
    models.value = await engine.loadedModels()
    error.value = null
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  }
  timer = setTimeout(poll, 1500)
}
onMounted(poll)
onUnmounted(() => clearTimeout(timer))

const BLOCK = /^(.*\.)?([a-z_]*blocks?)\.(\d+)$/
const rows = computed(() =>
  models.value
    .filter((m) => m.bytes > 0)
    .map((m) => {
      // the block lists, each in order; everything else is "other"
      const lists = new Map<string, { i: number; bytes: number; vram: number; name: string }[]>()
      let other = { bytes: 0, vram: 0 }
      for (const g of m.groups ?? []) {
        const hit = g.name.match(BLOCK)
        if (hit) {
          const key = hit[2]!
          lists.set(key, [...(lists.get(key) ?? []), { i: Number(hit[3]), bytes: g.bytes, vram: g.vram, name: g.name }])
        } else other = { bytes: other.bytes + g.bytes, vram: other.vram + g.vram }
      }
      return {
        ...m,
        lists: [...lists.entries()].map(([k, v]) => ({ name: k, cells: v.sort((a, b) => a.i - b.i) })).filter((l) => l.cells.length > 1),
        other,
      }
    }),
)
const pct = (v: number, b: number) => (b ? Math.round((v / b) * 100) : 0)
</script>

<template>
  <section class="lm">
    <div class="head">
      <span class="eyebrow">Loaded models</span>
      <span class="muted small">filled cells are in VRAM</span>
    </div>
    <div v-if="error" class="muted small">{{ error }}</div>
    <div v-else-if="!rows.length" class="muted small">Nothing loaded.</div>
    <div v-for="m in rows" :key="m.name" class="model">
      <div class="line">
        <b>{{ m.name }}</b>
        <span v-if="m.dynamic" class="tag" title="Streams weights into VRAM as it needs them (dynamic VRAM)">streaming</span>
        <span class="spacer" />
        <span class="mono small">{{ gb(m.vram) }} / {{ gb(m.bytes) }} GB · {{ pct(m.vram, m.bytes) }}%</span>
      </div>
      <div v-for="l in m.lists" :key="l.name" class="strip" :title="l.name">
        <span
          v-for="c in l.cells"
          :key="c.i"
          class="cell"
          :style="{ '--f': c.bytes ? c.vram / c.bytes : 0 }"
          :title="`${c.name}: ${pct(c.vram, c.bytes)}% in VRAM (${gb(c.bytes)} GB)`"
        />
      </div>
      <div v-if="!m.lists.length" class="bar"><span :style="{ width: `${pct(m.vram, m.bytes)}%` }" /></div>
    </div>
  </section>
</template>

<style scoped>
.lm {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.head,
.line {
  display: flex;
  align-items: center;
  gap: 8px;
}
.spacer {
  flex: 1;
}
.small {
  font-size: 11px;
}
.model {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.line b {
  font-size: 12px;
}
.tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: var(--r-sm);
  color: var(--zen-muted);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.strip {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(3px, 1fr);
  gap: 2px;
  height: 14px;
}
.cell {
  border-radius: 2px;
  background: linear-gradient(to top, var(--zen-accent) calc(var(--f) * 100%), color-mix(in srgb, var(--zen-text) 10%, transparent) 0);
}
.bar {
  height: 6px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-text) 10%, transparent);
  overflow: hidden;
}
.bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
</style>
