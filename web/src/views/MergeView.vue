<script setup lang="ts">
// Merge LoRAs into one file: a "soup" of a few short runs on the same subject (malcolmrey's
// recipe: AdamW, Automagic and Prodigy, 0.7 each), or a stack baked down. Exact merge, then an
// optional shrink to a small rank that keeps what the members agree on.
import { ZenButton, ZenCombo, ZenInput, ZenNumber, ZenSlider, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { inputChoices } from '@/api/comfy'
import { EngineError, engine } from '@/api/engine'
import { fmtBytes } from '@/downloads'

const route = useRoute()
const loras = ref<string[]>([])
const members = ref<{ name: string; strength: number }[]>([])
const outName = ref('')
const shrink = ref(true)
const maxRank = ref(16)
const busy = ref(false)
const error = ref<string | null>(null)
const result = ref<{ lora_name: string; mean_rank?: number; kept_energy?: number; bytes: number } | null>(null)

onMounted(async () => {
  loras.value = await inputChoices('LoraLoader', 'lora_name').catch(() => [])
  const add = route.query.add
  const names = (Array.isArray(add) ? add : add ? [add] : []).filter((x): x is string => typeof x === 'string')
  const each = names.length >= 3 ? 0.7 : names.length === 2 ? 0.95 : 1
  members.value = names.map((name) => ({ name, strength: each }))
  if (!members.value.length) members.value = [{ name: '', strength: 0.7 }, { name: '', strength: 0.7 }]
})
const items = computed(() => loras.value.map((n) => ({ value: n, label: n })))
const ready = computed(() => members.value.filter((m) => m.name).length >= 1 && outName.value.trim())
const total = computed(() => members.value.reduce((n, m) => n + (m.name ? m.strength : 0), 0))
const add = () => members.value.push({ name: '', strength: 0.7 })

async function run() {
  busy.value = true
  error.value = null
  result.value = null
  try {
    result.value = await engine.mergeLoras({
      members: members.value.filter((m) => m.name),
      out_name: outName.value.trim(),
      max_rank: shrink.value ? maxRank.value : 0,
    })
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="page merge">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/tools" class="crumb">Tools</RouterLink> Merge LoRAs</h1>
        <p class="lede">Bake several LoRAs into one, at the strengths you choose.</p>
      </div>
    </div>

    <section class="card col">
      <div class="eyebrow">LoRAs</div>
      <div v-for="(m, i) in members" :key="i" class="mrow">
        <ZenCombo v-model="m.name" :items="items" placeholder="Pick a LoRA…" :item-height="30" />
        <div class="str">
          <ZenSlider v-model="m.strength" :min="0" :max="1.5" :step="0.05" />
          <span class="mono small">{{ m.strength.toFixed(2) }}</span>
        </div>
        <button class="x" title="Remove" @click="members.splice(i, 1)"><i class="mdi mdi-close" /></button>
      </div>
      <div class="foot">
        <button class="add" @click="add"><i class="mdi mdi-plus" /> Add a LoRA</button>
        <RouterLink
          v-if="members.some((m) => m.name)"
          :to="{ path: '/apps/text-to-image', query: { loras: JSON.stringify(members.filter((m) => m.name)) } }"
          title="Generate with these LoRAs stacked at these strengths: the same result as the merged file, before any shrink"
        >
          <ZenButton size="sm" icon="mdi mdi-creation-outline">Test in Text to Image</ZenButton>
        </RouterLink>
        <span class="muted small">Total strength {{ total.toFixed(2) }}. A soup of 3 runs on one subject: 0.7 each; 2 runs: 0.95. Too strong (waxy, ignores prompts)? Lower them; no retraining.</span>
      </div>
    </section>

    <section class="card col">
      <div class="eyebrow">Result</div>
      <div class="grid-2">
        <label class="f"><span class="small muted">Name</span><ZenInput v-model="outName" placeholder="e.g. grandma-rose-soup" /></label>
        <div class="f">
          <span class="small muted">Shrink</span>
          <div class="sw">
            <ZenSwitch v-model="shrink" />
            <template v-if="shrink">
              <span class="small">to rank ≤</span>
              <ZenNumber v-model="maxRank" :min="1" :max="128" :step="4" :precision="0" />
            </template>
            <span v-else class="small muted">Keep every member's rank (bigger file, exactly the stack)</span>
          </div>
        </div>
      </div>
      <p v-if="shrink" class="muted small">Keeps, per layer, the directions holding 99.9% of the change: what the members agree on stays, scattered noise goes.</p>
      <div v-if="error" class="err small">{{ error }}</div>
      <div class="acts">
        <ZenButton variant="primary" icon="mdi mdi-call-merge" :disabled="!ready || busy" @click="run">{{ busy ? 'Merging…' : 'Merge' }}</ZenButton>
      </div>
      <div v-if="result" class="done">
        <i class="mdi mdi-check-circle-outline" />
        <span>
          Saved <b class="mono">{{ result.lora_name }}</b> · {{ fmtBytes(result.bytes) }}
          <template v-if="result.mean_rank"> · average rank {{ result.mean_rank.toFixed(1) }}, {{ ((result.kept_energy ?? 1) * 100).toFixed(2) }}% kept</template>
        </span>
        <RouterLink :to="`/apps/text-to-image?lora=${encodeURIComponent(result.lora_name)}`"><ZenButton size="sm" icon="mdi mdi-creation-outline">Try it</ZenButton></RouterLink>
      </div>
    </section>
  </div>
</template>

<style scoped>
.merge {
  max-width: 900px;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.mrow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 220px 28px;
  gap: 10px;
  align-items: center;
}
.str {
  display: flex;
  align-items: center;
  gap: 8px;
}
.str > :first-child {
  flex: 1;
}
.x {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  border-radius: var(--r-sm);
  height: 28px;
}
.x:hover {
  color: var(--zen-danger);
  background: var(--zen-surface-2);
}
.foot {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.add {
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 6px 12px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.add:hover {
  border-color: var(--zen-accent);
  color: var(--zen-text);
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.sw {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
}
.acts {
  display: flex;
}
.done {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.done i {
  color: var(--zen-ok);
  font-size: 18px;
}
.err {
  color: var(--zen-danger);
}
.small {
  font-size: 12px;
}
.muted {
  color: var(--zen-muted);
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
</style>
