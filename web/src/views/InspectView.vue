<script setup lang="ts">
// Inspect a .safetensors file, as Hugging Face's file viewer does: its tensors as a tree of
// layers (shape, precision, parameters), its metadata, and totals; read from the header alone,
// so a whole model opens at once. Save a copy beside it: other metadata, another precision,
// layers left out (engine/zentrainer/models/tensors.py).
import { ZenButton, ZenCombo, ZenEmpty, ZenInput, ZenSelect } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EngineError, engine, type TensorFileInfo, type TensorFolder, type TensorRow } from '@/api/engine'
import { fmtBytes } from '@/stores/downloads'

const route = useRoute()
const router = useRouter()

const FOLDER_LABELS: Record<TensorFolder, string> = {
  loras: 'LoRAs',
  diffusion_models: 'Diffusion models',
  checkpoints: 'Checkpoints',
  vae: 'VAEs',
  text_encoders: 'Text encoders',
}
const files = ref<Partial<Record<TensorFolder, string[]>>>({})
const folder = ref<string>((route.query.folder as string) || 'loras')
const name = ref<string>((route.query.name as string) || '')
const info = ref<TensorFileInfo | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const folderOptions = computed(() =>
  (Object.keys(FOLDER_LABELS) as TensorFolder[]).filter((f) => files.value[f]).map((f) => ({ value: f, label: `${FOLDER_LABELS[f]} (${files.value[f]?.length ?? 0})` })),
)
const fileItems = computed(() => (files.value[folder.value as TensorFolder] ?? []).map((n) => ({ value: n, label: n })))

onMounted(async () => {
  files.value = await engine.tensorFiles().catch(() => ({}))
})

watch(
  [folder, name],
  async ([f, n], old) => {
    if (old && f !== old[0] && n === old[1]) {
      name.value = '' // a new folder: pick again
      return
    }
    router.replace({ query: n ? { folder: f, name: n } : { folder: f } })
    info.value = null
    error.value = null
    reset()
    if (!n) return
    loading.value = true
    try {
      info.value = await engine.inspectTensors(f as TensorFolder, n)
      startMeta()
      outName.value = `${n.split('/').pop()!.replace(/\.safetensors$/i, '')}_copy`
    } catch (e) {
      error.value = e instanceof EngineError ? e.message : String(e)
    } finally {
      loading.value = false
    }
  },
  { immediate: true },
)

// ---- numbers ----------------------------------------------------------------------------------
const fmtParams = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n)
const kindLine = computed(() => {
  const i = info.value
  if (!i) return ''
  if (i.kind === 'lora') {
    const r = i.ranks?.length ? ` · rank ${i.ranks.length > 1 ? `${i.ranks[0]}–${i.ranks[i.ranks.length - 1]}` : i.ranks[0]}` : ''
    return `${i.algorithm} · ${i.layers} layers${r}`
  }
  return i.quantized ? 'Model · quantized' : 'Model'
})

// ---- the tensor tree --------------------------------------------------------------------------
interface Node {
  key: string
  label: string
  depth: number
  children: Map<string, Node>
  tensor?: TensorRow
  count: number
  params: number
  dtypes: Set<string>
}

const tree = computed<Node | null>(() => {
  const i = info.value
  if (!i) return null
  const root: Node = { key: '', label: '', depth: -1, children: new Map(), count: 0, params: 0, dtypes: new Set() }
  for (const t of i.tensors) {
    const parts = t.name.split('.')
    let node = root
    const path: Node[] = [root]
    parts.forEach((p, d) => {
      const key = parts.slice(0, d + 1).join('.')
      let next = node.children.get(p)
      if (!next) {
        next = { key, label: p, depth: d, children: new Map(), count: 0, params: 0, dtypes: new Set() }
        node.children.set(p, next)
      }
      node = next
      path.push(node)
    })
    node.tensor = t
    for (const n of path) {
      n.count += 1
      n.params += t.params
      n.dtypes.add(t.dtype)
    }
  }
  return root
})

const open = ref(new Set<string>())
const query = ref('')

function reset() {
  open.value = new Set()
  dropped.value = new Set()
  query.value = ''
  metaEditing.value = false
  result.value = null
  saveError.value = null
}

watch(tree, (t) => {
  // open the levels that hold a single group, so the first screen shows something
  const s = new Set<string>()
  let n = t
  while (n && n.children.size === 1) {
    const only: Node = n.children.values().next().value!
    if (!only.tensor) s.add(only.key)
    n = only
  }
  open.value = s
})

const rows = computed<Node[]>(() => {
  const t = tree.value
  if (!t) return []
  const out: Node[] = []
  const walk = (n: Node) => {
    for (const c of n.children.values()) {
      out.push(c)
      if (c.children.size && open.value.has(c.key)) walk(c)
    }
  }
  walk(t)
  return out
})

const MATCH_CAP = 400
const matches = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q || !info.value) return null
  return info.value.tensors.filter((t) => t.name.toLowerCase().includes(q))
})

function toggle(n: Node) {
  if (!n.children.size) return
  const s = new Set(open.value)
  if (s.has(n.key)) s.delete(n.key)
  else s.add(n.key)
  open.value = s
}

function expandAll(on: boolean) {
  const s = new Set<string>()
  if (on && tree.value) {
    const walk = (n: Node) => n.children.forEach((c) => c.children.size && (s.add(c.key), walk(c)))
    walk(tree.value)
  }
  open.value = s
}

// ---- leaving tensors out of the copy ------------------------------------------------------------
const dropped = ref(new Set<string>())
const isOut = (key: string) => [...dropped.value].some((p) => key === p || key.startsWith(p + '.'))
function toggleOut(key: string) {
  const s = new Set(dropped.value)
  if (s.has(key)) s.delete(key)
  else if (!isOut(key)) {
    for (const p of s) if (p.startsWith(key + '.')) s.delete(p) // the group now covers them
    s.add(key)
  } else return // inside a group that's out: bring the group back instead
  dropped.value = s
}

// ---- metadata -----------------------------------------------------------------------------------
const meta = ref<{ key: string; value: string }[]>([])
const metaEditing = ref(false)
function startMeta() {
  meta.value = Object.entries(info.value?.metadata ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => ({ key, value }))
}
const metaObject = computed(() => Object.fromEntries(meta.value.filter((m) => m.key.trim()).map((m) => [m.key.trim(), m.value])))
const metaChanged = computed(() => {
  const a = info.value?.metadata ?? {}
  const b = metaObject.value
  return Object.keys(a).length !== Object.keys(b).length || Object.entries(a).some(([k, v]) => b[k] !== v)
})
/** a JSON value shown pretty-printed; anything else as it is */
function pretty(v: string): { json: boolean; text: string } {
  const s = v.trim()
  if (s.startsWith('{') || s.startsWith('[')) {
    try {
      return { json: true, text: JSON.stringify(JSON.parse(s), null, 2) }
    } catch {
      /* not JSON after all */
    }
  }
  return { json: false, text: v }
}

// ---- saving a copy ------------------------------------------------------------------------------
const outName = ref('')
const dtype = ref<string>('')
const DTYPE_OPTIONS = [
  { value: '', label: 'Keep each tensor’s precision' },
  { value: 'bf16', label: 'bf16' },
  { value: 'fp16', label: 'fp16' },
  { value: 'fp32', label: 'fp32' },
]
const plan = ref<{ tensors: number; bytes: number; converted: number; dropped: number } | null>(null)
const saving = ref(false)
const saveError = ref<string | null>(null)
const result = ref<{ folder: TensorFolder; name: string; bytes: number; tensors: number; converted: number; dropped: number } | null>(null)

const body = () => ({
  folder: folder.value as TensorFolder,
  name: name.value,
  out_name: outName.value.trim(),
  dtype: (dtype.value || null) as 'bf16' | 'fp16' | 'fp32' | null,
  drop: [...dropped.value],
  metadata: metaChanged.value ? metaObject.value : undefined,
})

let planTimer: ReturnType<typeof setTimeout> | undefined
watch([dtype, dropped, info], () => {
  clearTimeout(planTimer)
  plan.value = null
  if (!info.value || (!dtype.value && !dropped.value.size)) return
  planTimer = setTimeout(async () => {
    plan.value = await engine.planTensorCopy(body()).catch(() => null)
  }, 200)
})

const willChange = computed(() => !!dtype.value || dropped.value.size > 0 || metaChanged.value)

async function save() {
  saving.value = true
  saveError.value = null
  result.value = null
  try {
    result.value = await engine.copyTensors(body())
    files.value = await engine.tensorFiles().catch(() => files.value)
  } catch (e) {
    saveError.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="page inspect">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/tools" class="crumb">Tools</RouterLink> Inspect a model file</h1>
        <p class="lede">Every layer, shape, precision and metadata entry in a .safetensors file, and a copy saved the way you want it.</p>
      </div>
    </div>

    <section class="card pick">
      <label class="f"><span class="small muted">Folder</span><ZenSelect v-model="folder" :options="folderOptions" /></label>
      <label class="f grow"><span class="small muted">File</span><ZenCombo v-model="name" :items="fileItems" placeholder="Pick a .safetensors file…" :item-height="30" /></label>
    </section>

    <div v-if="error" class="card err small">{{ error }}</div>
    <div v-else-if="loading" class="card muted small">Reading the header…</div>
    <ZenEmpty v-else-if="!info" icon="mdi mdi-file-tree-outline" title="Pick a file">A LoRA, a diffusion model, a VAE or a text encoder: only its header is read, so even a large model opens at once.</ZenEmpty>

    <template v-if="info">
      <section class="stats">
        <div class="stat"><span class="small muted">Kind</span><b>{{ kindLine }}</b></div>
        <div class="stat"><span class="small muted">Size</span><b>{{ fmtBytes(info.size) }}</b></div>
        <div class="stat"><span class="small muted">Parameters</span><b>{{ fmtParams(info.params) }}</b></div>
        <div class="stat"><span class="small muted">Tensors</span><b>{{ info.tensors.length.toLocaleString() }}</b></div>
        <div class="stat wide">
          <span class="small muted">Precision</span>
          <div class="chips">
            <span v-for="(d, k) in info.by_dtype" :key="k" class="chip mono" :title="`${d.tensors} tensors · ${fmtParams(d.params)} parameters`">{{ k }} <span class="muted">{{ fmtBytes(d.bytes) }}</span></span>
          </div>
        </div>
      </section>

      <div class="cols">
        <section class="card col tensors">
          <div class="bar">
            <div class="eyebrow">Tensors</div>
            <ZenInput v-model="query" placeholder="Filter by name…" class="q" />
            <template v-if="!matches">
              <button class="link small" @click="expandAll(true)">Expand all</button>
              <button class="link small" @click="expandAll(false)">Collapse</button>
            </template>
          </div>
          <div class="list">
            <template v-if="matches">
              <div v-for="t in matches.slice(0, MATCH_CAP)" :key="t.name" class="row" :class="{ out: isOut(t.name) }">
                <button class="tick" :title="isOut(t.name) ? 'Keep in the copy' : 'Leave out of the copy'" @click="toggleOut(t.name)">
                  <i :class="isOut(t.name) ? 'mdi mdi-checkbox-blank-outline' : 'mdi mdi-checkbox-marked-outline'" />
                </button>
                <span class="nm mono">{{ t.name }}</span>
                <span class="shape mono">[{{ t.shape.join(', ') }}]</span>
                <span class="dt mono">{{ t.dtype }}</span>
              </div>
              <div v-if="matches.length > MATCH_CAP" class="small muted pad">and {{ (matches.length - MATCH_CAP).toLocaleString() }} more: narrow the filter</div>
              <div v-if="!matches.length" class="small muted pad">No tensor names contain that.</div>
            </template>
            <template v-else>
              <div v-for="n in rows" :key="n.key" class="row" :class="{ out: isOut(n.key), group: n.children.size }" :style="{ paddingLeft: `${n.depth * 16 + 4}px` }">
                <button class="tick" :title="isOut(n.key) ? 'Keep in the copy' : 'Leave out of the copy'" @click="toggleOut(n.key)">
                  <i :class="isOut(n.key) ? 'mdi mdi-checkbox-blank-outline' : 'mdi mdi-checkbox-marked-outline'" />
                </button>
                <button v-if="n.children.size" class="nm mono caret" @click="toggle(n)">
                  <i :class="open.has(n.key) ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" />{{ n.label }}
                  <span class="muted">{{ n.count.toLocaleString() }} · {{ fmtParams(n.params) }}</span>
                </button>
                <span v-else class="nm mono leaf">{{ n.label }}</span>
                <span class="shape mono">{{ n.tensor ? `[${n.tensor.shape.join(', ')}]` : '' }}</span>
                <span class="dt mono">{{ n.tensor ? n.tensor.dtype : [...n.dtypes].join(' ') }}</span>
              </div>
            </template>
          </div>
        </section>

        <div class="side">
          <section class="card col">
            <div class="bar">
              <div class="eyebrow">Metadata</div>
              <span class="small muted">{{ meta.length }} entries</span>
              <button class="link small" @click="metaEditing = !metaEditing">{{ metaEditing ? 'Done' : 'Edit' }}</button>
              <button v-if="metaChanged" class="link small" @click="startMeta()">Undo changes</button>
            </div>
            <div v-if="!meta.length && !metaEditing" class="small muted">No metadata.</div>
            <div class="meta">
              <div v-for="(m, i) in meta" :key="i" class="mrow">
                <template v-if="metaEditing">
                  <ZenInput v-model="m.key" class="mono" />
                  <textarea v-model="m.value" class="val mono" rows="2" />
                  <button class="x" title="Remove" @click="meta.splice(i, 1)"><i class="mdi mdi-close" /></button>
                </template>
                <template v-else>
                  <span class="k mono">{{ m.key }}</span>
                  <pre class="v mono" :class="{ json: pretty(m.value).json }">{{ pretty(m.value).text }}</pre>
                </template>
              </div>
            </div>
            <button v-if="metaEditing" class="add" @click="meta.push({ key: '', value: '' })"><i class="mdi mdi-plus" /> Add an entry</button>
          </section>

          <section class="card col">
            <div class="eyebrow">Save a copy</div>
            <label class="f"><span class="small muted">Name (saved beside the original)</span><ZenInput v-model="outName" placeholder="e.g. my-lora_bf16" /></label>
            <label class="f"><span class="small muted">Precision</span><ZenSelect v-model="dtype" :options="DTYPE_OPTIONS" /></label>
            <ul class="small changes">
              <li v-if="dtype">Plain float weights become {{ dtype }}<template v-if="info.quantized">; quantized weights and their scales stay as they are</template></li>
              <li v-if="dropped.size">Leaving out {{ plan ? plan.dropped.toLocaleString() : '…' }} tensors ({{ dropped.size }} {{ dropped.size === 1 ? 'pick' : 'picks' }} in the tree)</li>
              <li v-if="metaChanged">Metadata as edited</li>
              <li v-if="!willChange" class="muted">An exact copy. Untick layers, change the precision or edit the metadata to make it different.</li>
              <li v-if="plan">About {{ fmtBytes(plan.bytes) }}, was {{ fmtBytes(info.size) }}</li>
            </ul>
            <div v-if="saveError" class="err small">{{ saveError }}</div>
            <div class="acts">
              <ZenButton variant="primary" icon="mdi mdi-content-save-outline" :disabled="!outName.trim() || saving" @click="save">{{ saving ? 'Saving…' : 'Save copy' }}</ZenButton>
            </div>
            <div v-if="result" class="done">
              <i class="mdi mdi-check-circle-outline" />
              <span>Saved <b class="mono">{{ result.name }}</b> · {{ fmtBytes(result.bytes) }}</span>
              <ZenButton size="sm" icon="mdi mdi-file-tree-outline" @click="folder = result.folder; name = result.name">Inspect it</ZenButton>
            </div>
          </section>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.inspect {
  max-width: 1400px;
}
.pick {
  display: flex;
  gap: 12px;
  align-items: end;
  flex-wrap: wrap;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 200px;
}
.grow {
  flex: 1;
}
.stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 14px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface);
  min-width: 120px;
}
.stat.wide {
  flex: 1;
}
.chips {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.chip {
  font-size: 12px;
  padding: 1px 8px;
  border-radius: 99px;
  background: var(--zen-surface-2);
}
.cols {
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(320px, 2fr);
  gap: 16px;
  align-items: start;
}
@media (max-width: 1000px) {
  .cols {
    grid-template-columns: minmax(0, 1fr);
  }
}
.side {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}
.bar {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.q {
  flex: 1;
  min-width: 160px;
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
  padding: 0;
}
.link:hover {
  color: var(--zen-text);
}
.list {
  max-height: 70vh;
  overflow: auto;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
}
.row {
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr) auto 64px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
  min-height: 26px;
  padding-right: 8px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.row:hover {
  background: var(--zen-surface-2);
}
.row.out .nm,
.row.out .shape,
.row.out .dt {
  opacity: 0.4;
  text-decoration: line-through;
}
.tick {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 0;
  font-size: 15px;
}
.tick:hover {
  color: var(--zen-accent);
}
.nm {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: left;
}
.caret {
  border: none;
  background: none;
  color: var(--zen-text);
  cursor: pointer;
  font-size: 12px;
  padding: 0;
  display: flex;
  gap: 6px;
  align-items: center;
}
.leaf {
  padding-left: 18px;
}
.shape {
  color: var(--zen-muted);
  white-space: nowrap;
}
.dt {
  color: var(--zen-muted);
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pad {
  padding: 8px;
}
.meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 50vh;
  overflow: auto;
}
.mrow {
  display: grid;
  grid-template-columns: minmax(120px, 1fr) minmax(0, 2fr) auto;
  gap: 8px;
  align-items: start;
  font-size: 12px;
}
.k {
  word-break: break-all;
  color: var(--zen-muted);
}
.v {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 160px;
  overflow: auto;
  font-size: 12px;
}
.v.json {
  background: var(--zen-surface-2);
  border-radius: var(--r-sm);
  padding: 4px 6px;
}
.val {
  font-size: 12px;
  resize: vertical;
  background: var(--zen-surface-2);
  color: var(--zen-text);
  border: 1px solid var(--zen-border);
  border-radius: var(--r-sm);
  padding: 4px 6px;
  min-width: 0;
}
.x {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  height: 28px;
}
.x:hover {
  color: var(--zen-danger);
}
.add {
  align-self: flex-start;
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
.changes {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.acts {
  display: flex;
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
