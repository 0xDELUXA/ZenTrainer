<script setup lang="ts">
// Compact a LoRA or LoKr to the size you want. Every layer's change is split into directions,
// all of the file's are ranked together by how much of the change each carries per parameter,
// and the copy keeps the top of that ranking: rank goes where the change is. Weighed against
// the base model, a change beside a large weight counts for less (engine/zentrainer/loras/compact.py).
import { ZenButton, ZenCombo, ZenInput, ZenNumber, ZenSelect, ZenSlider, ZenToggleGroup } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { inputChoices } from '@/api/comfy'
import { EngineError, engine, type CompactAnalysis, type CompactRequest } from '@/api/engine'
import { fmtBytes } from '@/stores/downloads'

const route = useRoute()
const loras = ref<string[]>([])
const bases = ref<{ value: string; label: string }[]>([])
const name = ref(typeof route.query.add === 'string' ? route.query.add : '')
const base = ref('') // "folder:name", or none
const format = ref<string>('same')
const mode = ref<string>('keep')
const keep = ref(99)
const sizeMB = ref(50)
const maxRank = ref(0)
const dtype = ref<string>('bf16')
const outName = ref('')

const analysis = ref<CompactAnalysis | null>(null)
const measuring = ref(false)
const error = ref<string | null>(null)
const saving = ref(false)
const saveError = ref<string | null>(null)
const result = ref<{ lora_name: string; bytes: number; mean_rank: number; kept: number; layers_kept: number; layers: number } | null>(null)

const BYTES: Record<string, number> = { bf16: 2, fp16: 2, fp32: 4 }
const bytesPer = computed(() => BYTES[dtype.value] ?? 2)

onMounted(async () => {
  loras.value = await inputChoices('LoraLoader', 'lora_name').catch(() => [])
  const files = await engine.tensorFiles().catch(() => ({}) as Awaited<ReturnType<typeof engine.tensorFiles>>)
  bases.value = [
    ...(files.diffusion_models ?? []).map((n) => ({ value: `diffusion_models:${n}`, label: n })),
    ...(files.checkpoints ?? []).map((n) => ({ value: `checkpoints:${n}`, label: `${n} (checkpoint)` })),
  ]
})
const loraItems = computed(() => loras.value.map((n) => ({ value: n, label: n })))
const baseItems = computed(() => [{ value: '', label: 'None: measure the change alone' }, ...bases.value])

const request = (): CompactRequest => {
  const [folder, ...rest] = base.value.split(':')
  return {
    name: name.value,
    format: format.value === 'lora' ? 'lora' : 'same',
    base: base.value ? { folder: folder as 'diffusion_models' | 'checkpoints', name: rest.join(':') } : null,
    max_rank: maxRank.value || 0,
    target: mode.value === 'keep' ? { keep: keep.value / 100 } : { params: Math.round((sizeMB.value * 1e6) / bytesPer.value) },
  }
}

let timer: ReturnType<typeof setTimeout> | undefined
let seq = 0
watch(
  [name, base, format, mode, keep, sizeMB, maxRank, dtype],
  ([n], old) => {
    clearTimeout(timer)
    result.value = null
    if (!n) {
      analysis.value = null
      return
    }
    if (!old || n !== old[0]) outName.value = `${n.split('/').pop()!.replace(/\.safetensors$/i, '')}_compact`
    timer = setTimeout(async () => {
      const mine = ++seq
      measuring.value = true
      error.value = null
      try {
        const a = await engine.analyzeCompact(request())
        if (mine === seq) analysis.value = a
      } catch (e) {
        if (mine === seq) error.value = e instanceof EngineError ? e.message : String(e)
      } finally {
        if (mine === seq) measuring.value = false
      }
    }, 250)
  },
  { immediate: true },
)

const hasLokr = computed(() => analysis.value?.kind.some((k) => k !== 'lora') ?? false)
const planBytes = computed(() => (analysis.value ? analysis.value.plan.params * bytesPer.value : 0))

// ---- the size-against-share chart ----------------------------------------------------------------
const W = 560
const H = 180
const PAD = 28
const chart = computed(() => {
  const a = analysis.value
  if (!a || a.curve.length < 2) return null
  const xs = a.curve.map(([p]) => p * bytesPer.value)
  const xmax = Math.max(...xs, a.original.bytes) || 1
  const ymin = 0.8
  const x = (b: number) => PAD + (b / xmax) * (W - PAD - 8)
  const y = (s: number) => 8 + (1 - (Math.max(s, ymin) - ymin) / (1 - ymin)) * (H - PAD - 8)
  const path = a.curve.map(([p, s], i) => `${i ? 'L' : 'M'}${x(p * bytesPer.value).toFixed(1)},${y(s).toFixed(1)}`).join(' ')
  const ticks = [0.8, 0.9, 0.95, 0.99, 1].map((s) => ({ s, y: y(s) }))
  return { path, ticks, x, y, xmax, at: { x: x(planBytes.value), y: y(a.plan.kept) }, orig: x(a.original.bytes) }
})

function pick(ev: MouseEvent) {
  const c = chart.value
  if (!c) return
  const box = (ev.currentTarget as SVGElement).getBoundingClientRect()
  const px = ((ev.clientX - box.left) / box.width) * W
  const b = Math.max(0, ((px - PAD) / (W - PAD - 8)) * c.xmax)
  mode.value = 'size'
  sizeMB.value = Math.max(0.1, Math.round(b / 1e5) / 10)
}

// ---- ranks by layer --------------------------------------------------------------------------------
const bars = computed(() => {
  const a = analysis.value
  if (!a) return []
  return a.plan.ranks.map((k, i) => ({ k, full: a.full_ranks[i] ?? 1, name: a.names[i] ?? '' }))
})

async function save() {
  saving.value = true
  saveError.value = null
  result.value = null
  try {
    result.value = await engine.compactLora({ ...request(), out_name: outName.value.trim(), dtype: dtype.value as 'bf16' | 'fp16' | 'fp32' })
  } catch (e) {
    saveError.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="page compact">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/tools" class="crumb">Tools</RouterLink> Compact a LoRA</h1>
        <p class="lede">The smallest file that keeps what matters: rank goes to the layers that change most.</p>
      </div>
    </div>

    <section class="card col">
      <div class="grid-2">
        <label class="f"><span class="small muted">LoRA or LoKr</span><ZenCombo v-model="name" :items="loraItems" placeholder="Pick a LoRA…" :item-height="30" /></label>
        <label class="f">
          <span class="small muted">Base model (optional)</span>
          <ZenCombo v-model="base" :items="baseItems" placeholder="None: measure the change alone" :item-height="30" />
        </label>
      </div>
      <span v-if="base" class="small muted">Each layer's change is measured against that layer's own weight in the model: a change beside a large weight counts for less. Only the weights' sizes are read.</span>
      <span v-if="analysis?.base_matched != null" class="small" :class="analysis.base_matched ? 'muted' : 'err'">
        {{ analysis.base_matched }} of {{ analysis.names.length }} layers found in the base model{{ analysis.base_matched ? '' : ': is it the model this LoRA was trained on?' }}
      </span>
    </section>

    <div v-if="error" class="card err small">{{ error }}</div>

    <section v-if="name" class="card col">
      <div class="eyebrow">Target</div>
      <div class="row">
        <ZenToggleGroup v-model="mode" :options="[{ value: 'keep', label: 'Share kept' }, { value: 'size', label: 'File size' }]" />
        <template v-if="mode === 'keep'">
          <ZenSlider v-model="keep" :min="80" :max="100" :step="0.1" class="grow" />
          <span class="mono">{{ keep.toFixed(1) }}%</span>
        </template>
        <template v-else>
          <ZenNumber v-model="sizeMB" :min="0.1" :max="100000" :step="5" :precision="1" />
          <span class="small muted">MB</span>
        </template>
      </div>
      <div class="row">
        <label class="f"><span class="small muted">Rank cap per layer (0: none)</span><ZenNumber v-model="maxRank" :min="0" :max="4096" :step="4" :precision="0" /></label>
        <label class="f"><span class="small muted">Precision</span><ZenSelect v-model="dtype" :options="['bf16', 'fp16', 'fp32']" /></label>
        <label v-if="hasLokr" class="f">
          <span class="small muted">LoKr layers</span>
          <ZenToggleGroup v-model="format" :options="[{ value: 'same', label: 'Keep as LoKr' }, { value: 'lora', label: 'Make a LoRA' }]" />
        </label>
      </div>
      <span v-if="hasLokr && format === 'lora'" class="small muted">A LoKr's Kronecker structure is very efficient; as a LoRA the same share usually takes many times the size.</span>

      <div v-if="measuring && !analysis" class="small muted">Measuring every layer… (a large LoKr takes a little while; moving the target afterwards is instant)</div>
      <template v-if="analysis">
        <div class="stats">
          <div class="stat"><span class="small muted">Size</span><b>{{ fmtBytes(planBytes) }}</b><span class="small muted">was {{ fmtBytes(analysis.original.bytes) }}</span></div>
          <div class="stat"><span class="small muted">Share kept</span><b>{{ (analysis.plan.kept * 100).toFixed(2) }}%</b></div>
          <div class="stat"><span class="small muted">Layers kept</span><b>{{ analysis.plan.layers_kept }} / {{ analysis.plan.layers }}</b></div>
          <div class="stat"><span class="small muted">Rank</span><b>{{ analysis.plan.mean_rank.toFixed(1) }} avg</b><span class="small muted">up to {{ analysis.plan.max_rank }}</span></div>
          <span v-if="measuring" class="small muted spin"><i class="mdi mdi-loading mdi-spin" /></span>
        </div>

        <div v-if="chart" class="chartbox">
          <svg :viewBox="`0 0 ${W} ${H}`" class="chart" @click="pick">
            <g v-for="t in chart.ticks" :key="t.s">
              <line :x1="PAD" :x2="W - 8" :y1="t.y" :y2="t.y" class="grid" />
              <text :x="PAD - 4" :y="t.y + 3" class="tick" text-anchor="end">{{ Math.round(t.s * 1000) / 10 }}</text>
            </g>
            <line :x1="chart.orig" :x2="chart.orig" :y1="8" :y2="H - PAD" class="orig" />
            <path :d="chart.path" class="line" />
            <circle :cx="chart.at.x" :cy="chart.at.y" r="5" class="dot" />
            <text :x="PAD" :y="H - 8" class="tick">0</text>
            <text :x="W - 8" :y="H - 8" class="tick" text-anchor="end">{{ fmtBytes(chart.xmax) }}</text>
          </svg>
          <span class="small muted">Share of the change kept (%) against file size; the dashed line is the original. Click to pick a size.</span>
        </div>

        <div class="col">
          <span class="small muted">Rank by layer, in file order (pale: what the layer had)</span>
          <svg :viewBox="`0 0 ${Math.max(bars.length, 1) * 4} 60`" preserveAspectRatio="none" class="bars">
            <g v-for="(b, i) in bars" :key="b.name">
              <rect :x="i * 4" :y="0" width="3" height="60" class="full"><title>{{ b.name }}: {{ b.k }} of {{ b.full }}</title></rect>
              <rect :x="i * 4" :y="60 - (b.k / Math.max(1, b.full)) * 60" width="3" :height="(b.k / Math.max(1, b.full)) * 60" class="kept"><title>{{ b.name }}: {{ b.k }} of {{ b.full }}</title></rect>
            </g>
          </svg>
        </div>
        <span v-if="analysis.unsupported.length" class="small muted">{{ analysis.unsupported.length }} layers (LoHa or Tucker LoKr) are copied as they are.</span>
      </template>
    </section>

    <section v-if="analysis" class="card col">
      <div class="eyebrow">Save</div>
      <label class="f"><span class="small muted">Name</span><ZenInput v-model="outName" placeholder="e.g. my-lora_compact" /></label>
      <div v-if="saveError" class="err small">{{ saveError }}</div>
      <div class="row">
        <ZenButton variant="primary" icon="mdi mdi-arrow-collapse-all" :disabled="!outName.trim() || saving" @click="save">{{ saving ? 'Compacting…' : 'Compact' }}</ZenButton>
        <RouterLink :to="`/apps/text-to-image?lora=${encodeURIComponent(name)}`"><ZenButton size="sm" icon="mdi mdi-creation-outline">Try the original</ZenButton></RouterLink>
      </div>
      <div v-if="result" class="done">
        <i class="mdi mdi-check-circle-outline" />
        <span>Saved <b class="mono">{{ result.lora_name }}</b> · {{ fmtBytes(result.bytes) }} · average rank {{ result.mean_rank.toFixed(1) }}, {{ (result.kept * 100).toFixed(2) }}% kept</span>
        <RouterLink :to="`/apps/text-to-image?lora=${encodeURIComponent(result.lora_name)}`"><ZenButton size="sm" icon="mdi mdi-creation-outline">Try it</ZenButton></RouterLink>
        <RouterLink :to="{ path: '/tools/inspect', query: { folder: 'loras', name: result.lora_name } }"><ZenButton size="sm" icon="mdi mdi-file-tree-outline">Inspect it</ZenButton></RouterLink>
      </div>
    </section>
  </div>
</template>

<style scoped>
.compact {
  max-width: 900px;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.row {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  flex-wrap: wrap;
}
.row > .grow {
  flex: 1;
  min-width: 160px;
  align-self: center;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 12px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  min-width: 110px;
}
.chartbox {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.chart {
  width: 100%;
  height: auto;
  cursor: crosshair;
}
.grid {
  stroke: var(--zen-border);
  stroke-width: 1;
}
.orig {
  stroke: var(--zen-muted);
  stroke-dasharray: 4 3;
}
.line {
  fill: none;
  stroke: var(--zen-accent);
  stroke-width: 2;
}
.dot {
  fill: var(--zen-accent);
  stroke: var(--zen-surface);
  stroke-width: 2;
}
.tick {
  font-size: 10px;
  fill: var(--zen-muted);
}
.bars {
  width: 100%;
  height: 60px;
}
.bars .full {
  fill: var(--zen-surface-2);
}
.bars .kept {
  fill: var(--zen-accent);
}
.done {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  flex-wrap: wrap;
}
.done i {
  color: var(--zen-ok);
  font-size: 18px;
}
.err {
  color: var(--zen-danger);
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
</style>
