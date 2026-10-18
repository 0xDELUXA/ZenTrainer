<script setup lang="ts">
// One manifest input, drawn by type. Sweepable inputs get a "Sweep" toggle: on, they
// take several values (one per line, or comma-separated numbers) and the run expands
// into a grid with one image per combination.
import { ZenCombo, ZenInput, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { inputChoices } from '@/api/comfy'
import type { AppInput } from '@/api/engine'
import ModelPicker, { type ModelValue } from './ModelPicker.vue'

const props = defineProps<{ input: AppInput; runs?: { name: string; loras: string[] }[] }>()
const value = defineModel<unknown>('value')
const sweep = defineModel<unknown[] | undefined>('sweep')

const options = ref<string[]>(props.input.options ?? [])
const loraFiles = ref<string[]>([])
onMounted(async () => {
  if (props.input.options_from) {
    const [node, field] = props.input.options_from.split('.')
    options.value = await inputChoices(node!, field!)
  }
  if (props.input.type === 'loras' || props.input.type === 'lora') loraFiles.value = await inputChoices('LoraLoader', 'lora_name')
})

const sweeping = computed({
  get: () => sweep.value !== undefined,
  set: (on: boolean) => {
    if (on) {
      const v = value.value
      sweep.value = Array.isArray(v) ? [...v] : v != null && v !== '' ? [v] : []
    } else {
      if (sweep.value?.length) value.value = sweep.value[0]
      sweep.value = undefined
    }
  },
})

// text forms of a sweep: one per line for text, comma-separated for numbers
const isNumeric = computed(() => props.input.type === 'number' || props.input.type === 'seed')
const sweepText = computed({
  get: () => (sweep.value ?? []).join(isNumeric.value ? ', ' : '\n'),
  set: (t: string) => {
    sweep.value = isNumeric.value
      ? t.split(/[,\s]+/).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n))
      : t.split('\n').map((x) => x.trim()).filter(Boolean)
  },
})

// loras: [{name, strength}]
const loraList = computed(() => (Array.isArray(value.value) ? (value.value as { name: string; strength: number }[]) : []))
const setLoras = (l: { name: string; strength: number }[]) => (value.value = l)

// lora (sweepable files) and quick-fill from a run's checkpoints
const pickedLoras = computed(() => (sweeping.value ? ((sweep.value ?? []) as string[]) : value.value ? [value.value as string] : []))
function toggleLora(name: string) {
  if (!sweeping.value) {
    value.value = name
    return
  }
  const cur = (sweep.value ?? []) as string[]
  sweep.value = cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name]
}
function fillFromRun(name: string | number) {
  const run = props.runs?.find((r) => r.name === name)
  if (!run) return
  sweeping.value = true
  sweep.value = [...run.loras]
}

const RES = [
  [1024, 1024],
  [832, 1216],
  [1216, 832],
  [896, 1152],
  [768, 768],
  [512, 512],
]
const res = computed(() => (Array.isArray(value.value) ? (value.value as number[]) : [1024, 1024]))
const short = (n: string) => n.split('/').at(-1)!.replace(/\.safetensors$/, '')
</script>

<template>
  <div class="ai">
    <div class="head">
      <label class="lbl">{{ input.label ?? input.id }}</label>
      <label v-if="input.sweep && input.type !== 'lora'" class="sw" title="Several values: one image per combination">
        <span>Sweep</span><ZenSwitch v-model="sweeping" />
      </label>
    </div>

    <ModelPicker v-if="input.type === 'model'" :model-value="(value as ModelValue) ?? { kind: 'checkpoint' }" @update:model-value="value = $event" />

    <template v-else-if="sweeping && input.type !== 'lora'">
      <ZenInput v-model="sweepText" :type="isNumeric ? 'text' : 'textarea'" :rows="4" :placeholder="isNumeric ? 'e.g. 0.8, 1.0, 1.2' : 'one per line'" />
    </template>

    <ZenInput v-else-if="input.type === 'prompt'" :model-value="(value as string) ?? ''" type="textarea" :rows="4" :placeholder="input.placeholder" @update:model-value="value = $event" />
    <ZenInput v-else-if="input.type === 'text'" :model-value="(value as string) ?? ''" :placeholder="input.placeholder" @update:model-value="value = $event" />
    <ZenNumber
      v-else-if="input.type === 'number'"
      :model-value="Number(value ?? input.default ?? 0)"
      :min="input.min"
      :max="input.max"
      :step="input.step ?? 1"
      :precision="(input.step ?? 1) < 1 ? 2 : 0"
      @update:model-value="value = $event"
    />
    <div v-else-if="input.type === 'seed'" class="row">
      <ZenNumber :model-value="Number(value ?? -1)" :min="-1" :step="1" :precision="0" @update:model-value="value = $event" />
      <button class="icon" title="Random each run (-1)" @click="value = -1"><i class="mdi mdi-dice-multiple-outline" /></button>
    </div>
    <ZenSwitch v-else-if="input.type === 'toggle'" :model-value="!!value" @update:model-value="value = $event" />
    <ZenCombo v-else-if="input.type === 'choice'" :model-value="(value as string) ?? ''" :items="options.map((o) => ({ value: o, label: o }))" @update:model-value="value = $event" />

    <div v-else-if="input.type === 'resolution'" class="res">
      <button v-for="r in RES" :key="r.join()" class="chip" :class="{ on: res[0] === r[0] && res[1] === r[1] }" @click="value = [...r]">
        {{ r[0] }}×{{ r[1] }}
      </button>
      <div class="row">
        <ZenNumber :model-value="res[0]!" :min="256" :step="64" :precision="0" @update:model-value="value = [$event, res[1]]" />
        <span class="muted">×</span>
        <ZenNumber :model-value="res[1]!" :min="256" :step="64" :precision="0" @update:model-value="value = [res[0], $event]" />
      </div>
    </div>

    <div v-else-if="input.type === 'loras'" class="loras">
      <div v-for="(lo, i) in loraList" :key="i" class="row">
        <ZenCombo :model-value="lo.name" :items="loraFiles.map((f) => ({ value: f, label: f }))" :item-height="30" placeholder="LoRA…" @update:model-value="setLoras(loraList.map((x, j) => (j === i ? { ...x, name: String($event) } : x)))" />
        <ZenNumber :model-value="lo.strength" :min="-2" :max="3" :step="0.05" :precision="2" @update:model-value="setLoras(loraList.map((x, j) => (j === i ? { ...x, strength: $event } : x)))" />
        <button class="icon" title="Remove" @click="setLoras(loraList.filter((_, j) => j !== i))"><i class="mdi mdi-close" /></button>
      </div>
      <button class="add" @click="setLoras([...loraList, { name: '', strength: 1 }])"><i class="mdi mdi-plus" /> Add LoRA</button>
    </div>

    <div v-else-if="input.type === 'lora'" class="lora-pick">
      <div class="row">
        <ZenCombo
          v-if="runs?.length"
          model-value=""
          :items="runs.map((r) => ({ value: r.name, label: `${r.name} · ${r.loras.length} checkpoints` }))"
          placeholder="All checkpoints of a run…"
          :sorted="false"
          @update:model-value="fillFromRun"
        />
        <label class="sw"><span>Several</span><ZenSwitch v-model="sweeping" /></label>
      </div>
      <ZenCombo model-value="" :items="loraFiles.map((f) => ({ value: f, label: f }))" :item-height="30" placeholder="Add a LoRA file…" @update:model-value="toggleLora(String($event))" />
      <div class="picked">
        <button v-for="n in pickedLoras" :key="n" class="chip on" :title="n" @click="toggleLora(n)">{{ short(n) }} <i class="mdi mdi-close" /></button>
      </div>
    </div>

    <p v-if="input.hint" class="hint">{{ input.hint }}</p>
  </div>
</template>

<style scoped>
.ai {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.lbl {
  font-size: 12px;
  color: var(--zen-muted);
}
.sw {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--zen-muted);
}
.row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.row > :first-child {
  flex: 1;
}
.icon {
  width: 28px;
  height: 28px;
  flex: none;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.icon:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.res {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.res .row {
  width: 100%;
  margin-top: 4px;
}
.chip {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: var(--r-md);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.chip.on {
  color: var(--zen-text);
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 14%, transparent);
}
.loras,
.lora-pick {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.picked {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.add {
  align-self: flex-start;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 80%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.hint {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted);
}
</style>
