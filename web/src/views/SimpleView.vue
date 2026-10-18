<script setup lang="ts">
// Simple mode: pick a recipe card, add photos, name it, train. Everything else (settings, captions,
// model files) comes from the recipe; what's missing is downloaded and the photos are captioned
// before the run starts, each as a visible step.
import { ZenButton, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { EngineError, defaultSpec, engine, thumbUrl, type DatasetInfo, type DatasetItem } from '@/api/engine'
import { captionJobs, startCaptionSync } from '@/captions'
import { downloadFor, finished, fmtBytes, startDownloadSync } from '@/downloads'
import { RECIPES, buildSpec, eta, plan, recipe as findRecipe, stepsFor, triggerFrom, type Plan, type Recipe } from '@/recipes'
import { settings } from '@/settings'

const props = defineProps<{ id?: string }>()
const router = useRouter()
const r = computed<Recipe | null>(() => (props.id ? findRecipe(props.id) : null))

// ---- the gallery ---------------------------------------------------------------------------
const kinds = [
  { id: 'all', label: 'All' },
  { id: 'character', label: 'People & characters' },
  { id: 'style', label: 'Styles' },
] as const
const kind = ref<(typeof kinds)[number]['id']>('all')
const shown = computed(() => RECIPES.filter((x) => kind.value === 'all' || x.kind === kind.value))

// ---- the photos ----------------------------------------------------------------------------
const name = ref('')
const trigger = ref('')
const triggerEdited = ref(false)
watch(name, (n) => {
  if (!triggerEdited.value) trigger.value = triggerFrom(n)
})
const datasets = ref<DatasetInfo[]>([])
const source = ref<'new' | 'existing'>('new')
const existing = ref('')
const dsName = computed(() => (source.value === 'existing' ? existing.value : name.value ? `simple_${triggerFrom(name.value)}` : ''))
const items = ref<DatasetItem[]>([])
const uploading = ref(false)
const error = ref<string | null>(null)

async function loadItems() {
  if (!dsName.value) return (items.value = [])
  try {
    items.value = (await engine.dataset(dsName.value)).items
  } catch {
    items.value = []
  }
}
watch(dsName, loadItems)

async function addFiles(files: File[]) {
  const pics = files.filter((f) => /^image\//.test(f.type))
  if (!pics.length) return
  if (!name.value.trim()) {
    error.value = 'Give it a name first: the photos are kept in a folder called after it.'
    return
  }
  uploading.value = true
  error.value = null
  try {
    if (!datasets.value.some((d) => d.folder === dsName.value)) {
      await engine.createDataset(dsName.value).catch(() => {})
      datasets.value = await engine.datasets()
    }
    const res = await engine.upload(dsName.value, pics)
    if (res.errors.length) error.value = res.errors.join('; ')
    await loadItems()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    uploading.value = false
  }
}
const over = ref(false)
function onDrop(e: DragEvent) {
  over.value = false
  addFiles([...(e.dataTransfer?.files ?? [])])
}
const picker = ref<HTMLInputElement | null>(null)
async function removeItem(it: DatasetItem) {
  await engine.deleteItem(dsName.value, it.path)
  await loadItems()
}

const count = computed(() => items.value.length)
const enough = computed(() => !!r.value && count.value >= r.value.photos.min)

// ---- the plan: files, downloads, time --------------------------------------------------------
const p = ref<Plan | null>(null)
watch(
  r,
  async (x) => {
    p.value = null
    if (x) p.value = await plan(x).catch((e) => ({ problems: [String(e)] }) as Plan)
  },
  { immediate: true },
)
const downloadBytes = computed(() => (p.value?.downloads ?? []).reduce((n, d) => n + (d.size ?? 0), 0) + (p.value?.adapter?.size ?? 0))
const steps = computed(() => (r.value ? stepsFor(r.value, Math.max(count.value, r.value.photos.min)) : 0))
const estimate = computed(() => {
  if (!r.value) return ''
  const uncaptioned = items.value.filter((i) => !i.caption).length
  const s = steps.value * r.value.secPerStep + 120 + uncaptioned * 6 + (steps.value / (r.value.spec.samples?.steps ? 200 : 300)) * 20
  return eta(s)
})

// ---- training: downloads, then captions, then the run ---------------------------------------
type Phase = 'idle' | 'downloading' | 'captioning' | 'starting'
const phase = ref<Phase>('idle')
const capJob = ref<string | null>(null)
const pendingDownloads = computed(() =>
  (p.value?.downloads ?? []).map((d) => ({ ...d, job: downloadFor(d.folder, d.name) })),
)
const capProgress = computed(() => {
  const j = capJob.value ? captionJobs[capJob.value] : undefined
  return j ? { done: j.done, total: j.total || count.value, status: j.status, loading: j.message?.startsWith('loading') } : null
})

function waitFor(check: () => boolean | string, every = 1000): Promise<void> {
  return new Promise((resolve, reject) => {
    const tick = () => {
      const v = check()
      if (typeof v === 'string') return reject(new Error(v))
      if (v) return resolve()
      setTimeout(tick, every)
    }
    tick()
  })
}

async function train() {
  const x = r.value
  if (!x || !p.value) return
  error.value = null
  try {
    // 1. models
    if (p.value.downloads.length || p.value.adapter) {
      phase.value = 'downloading'
      for (const d of p.value.downloads) await engine.download(x.family, d.folder, d.name)
      if (p.value.adapter) await engine.downloadAdapter(p.value.adapter.id)
      await waitFor(() => {
        const jobs = p.value!.downloads.map((d) => downloadFor(d.folder, d.name))
        const bad = jobs.find((j) => j && (j.status === 'error' || j.status === 'cancelled'))
        if (bad) return `${bad.name}: ${bad.error ?? bad.status}`
        return jobs.every((j) => j?.status === 'done')
      })
      p.value = await plan(x) // now installed: the names ComfyUI lists them by
    }
    // 2. captions for photos without one
    await loadItems()
    if (items.value.some((i) => !i.caption)) {
      phase.value = 'captioning'
      const info = await engine.captionInfo(dsName.value)
      const model =
        info.models.find((m) => /qwen3vl_4b/i.test(m)) ?? info.models.find((m) => /qwen3.?vl/i.test(m)) ?? info.models[0]
      if (!model) throw new Error('no vision-language model to caption with: install Qwen3-VL 4B (Models page)')
      const { job } = await engine.startCaption(dsName.value, {
        clip_name: model,
        clip_type: 'auto',
        set: 'default',
        mode: 'missing',
        prompt: x.caption.prompt,
        subject: x.caption.subject ? trigger.value : undefined,
      } as Parameters<typeof engine.startCaption>[1])
      capJob.value = job.job_id
      await waitFor(() => {
        const j = captionJobs[job.job_id]
        if (j?.status === 'error') return `captioning failed: ${j.message}`
        if (j?.status === 'stopped') return 'captioning was stopped'
        return j?.status === 'done'
      })
    }
    // 3. the run
    phase.value = 'starting'
    const base = defaultSpec()
    base.output.keep_states = settings.keepStates
    const spec = buildSpec(x, p.value, base, { name: name.value.trim(), trigger: trigger.value, dataset: dsName.value, photos: count.value })
    const run = await engine.start(spec)
    router.push(`/simple/run/${run.run_id}`)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : (e as Error).message ?? String(e)
    phase.value = 'idle'
  }
}

onMounted(async () => {
  startDownloadSync()
  startCaptionSync()
  datasets.value = await engine.datasets().catch(() => [])
})
watch(finished, () => r.value && plan(r.value).then((x) => phase.value === 'idle' && (p.value = x)))
let leaving = false
onUnmounted(() => (leaving = true))
void leaving
</script>

<template>
  <div class="page simple">
    <!-- the gallery -->
    <template v-if="!r">
      <div class="page-head">
        <div class="titles">
          <h1>Train a LoRA</h1>
          <p class="lede">Pick what you want to teach. Everything is set up for you.</p>
        </div>
        <RouterLink to="/runs/new"><ZenButton variant="ghost" icon="mdi mdi-tune-variant">All settings</ZenButton></RouterLink>
      </div>
      <div class="kinds">
        <button v-for="k in kinds" :key="k.id" :class="{ on: kind === k.id }" @click="kind = k.id">{{ k.label }}</button>
      </div>
      <div class="cards">
        <RouterLink v-for="x in shown" :key="x.id" :to="`/simple/${x.id}`" class="rcard">
          <div class="cover" :style="{ '--a': x.cover.from, '--b': x.cover.to }">
            <i :class="x.cover.icon" />
            <span class="badge">{{ x.kind === 'character' ? 'Character' : x.kind === 'style' ? 'Style' : 'Video' }}</span>
          </div>
          <div class="rbody">
            <b>{{ x.title }}</b>
            <p>{{ x.blurb }}</p>
            <span class="muted small">{{ x.photos.ideal }} · {{ eta(stepsFor(x, 25) * x.secPerStep + 180) }}</span>
          </div>
        </RouterLink>
      </div>
    </template>

    <!-- one recipe -->
    <template v-else>
      <div class="page-head">
        <div class="titles">
          <h1><RouterLink to="/simple" class="crumb">Train</RouterLink> {{ r.title }}</h1>
          <p class="lede">{{ r.blurb }}</p>
        </div>
      </div>

      <div class="flow">
        <section class="card step">
          <div class="num">1</div>
          <div class="sbody">
            <h2>What's it called?</h2>
            <ZenInput v-model="name" :placeholder="r.kind === 'style' ? 'e.g. Watercolour postcards' : 'e.g. Grandma Rose'" />
            <div class="muted small trig">
              Use the word
              <input v-model="trigger" class="trig-in mono" spellcheck="false" @input="triggerEdited = true" />
              in a prompt to bring it in.
            </div>
          </div>
        </section>

        <section class="card step">
          <div class="num">2</div>
          <div class="sbody">
            <h2>{{ r.kind === 'style' ? 'Add images in the style' : 'Add photos' }}</h2>
            <div class="seg small">
              <button :class="{ on: source === 'new' }" @click="source = 'new'">New photos</button>
              <button :class="{ on: source === 'existing' }" :disabled="!datasets.length" @click="source = 'existing'">A dataset I have</button>
            </div>
            <select v-if="source === 'existing'" v-model="existing" class="sel">
              <option value="" disabled>Pick a dataset…</option>
              <option v-for="d in datasets" :key="d.folder" :value="d.folder">{{ d.folder }} ({{ d.images }})</option>
            </select>
            <div
              v-else
              class="drop"
              :class="{ over }"
              @dragover.prevent="over = true"
              @dragleave="over = false"
              @drop.prevent="onDrop"
              @click="picker?.click()"
            >
              <i :class="uploading ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-image-plus-outline'" />
              <span>{{ uploading ? 'Adding…' : 'Drop photos here, or click to choose' }}</span>
              <input ref="picker" type="file" accept="image/*" multiple hidden @change="addFiles([...(($event.target as HTMLInputElement).files ?? [])])" />
            </div>
            <div class="tips">
              <span class="muted small"><b>{{ r.photos.ideal }}</b> works best:</span>
              <span v-for="t in r.photos.tips" :key="t" class="tip small"><i class="mdi mdi-check" />{{ t }}</span>
            </div>
            <div v-if="items.length" class="thumbs">
              <div v-for="it in items" :key="it.path" class="th">
                <img :src="thumbUrl(dsName, it.path, 192)" loading="lazy" alt="" />
                <button v-if="source === 'new'" class="x" title="Remove" @click="removeItem(it)"><i class="mdi mdi-close" /></button>
              </div>
            </div>
            <div class="count small" :class="{ warn: count && !enough }">
              {{ count }} {{ count === 1 ? 'photo' : 'photos' }}<template v-if="count && !enough"> · add at least {{ r.photos.min - count }} more</template>
            </div>
          </div>
        </section>

        <section class="card step">
          <div class="num">3</div>
          <div class="sbody">
            <h2>Train it</h2>
            <ul class="checks small">
              <li v-if="p && !p.downloads.length && !p.adapter"><i class="mdi mdi-check-circle-outline ok" /> Models are installed</li>
              <li v-else-if="p"><i class="mdi mdi-download-outline" /> Downloads {{ fmtBytes(downloadBytes) }} of models first</li>
              <li v-else><i class="mdi mdi-loading mdi-spin" /> Checking models…</li>
              <li><i class="mdi mdi-text-box-outline" /> Describes each photo for you</li>
              <li v-if="estimate"><i class="mdi mdi-timer-outline" /> Takes {{ estimate }}</li>
              <li v-for="q in p?.problems ?? []" :key="q" class="err"><i class="mdi mdi-alert-circle-outline" /> {{ q }}</li>
            </ul>

            <div v-if="phase !== 'idle'" class="doing">
              <template v-if="phase === 'downloading'">
                <div v-for="d in pendingDownloads" :key="d.name" class="prow">
                  <span class="ellipsis small">Downloading {{ d.name }}</span>
                  <div class="pbar"><span :style="{ width: `${d.job?.total ? (d.job.done / d.job.total) * 100 : 0}%` }" /></div>
                </div>
              </template>
              <div v-else-if="phase === 'captioning'" class="prow">
                <span class="small">{{ capProgress?.loading ? 'Getting the describer ready…' : `Describing photos ${capProgress?.done ?? 0} of ${capProgress?.total ?? count}` }}</span>
                <div class="pbar"><span :style="{ width: `${capProgress?.total ? (capProgress.done / capProgress.total) * 100 : 0}%` }" /></div>
              </div>
              <div v-else class="prow small"><i class="mdi mdi-loading mdi-spin" /> Starting…</div>
            </div>

            <div v-if="error" class="err small">{{ error }}</div>
            <ZenButton
              variant="primary"
              icon="mdi mdi-play"
              :disabled="phase !== 'idle' || !name.trim() || !enough || !p || !!p.problems.length"
              @click="train"
            >
              Train {{ name.trim() || 'it' }}
            </ZenButton>
          </div>
        </section>
      </div>
    </template>
  </div>
</template>

<style scoped>
.kinds,
.seg {
  display: flex;
  gap: 4px;
}
.kinds button,
.seg button {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.kinds button.on,
.seg button.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.seg button:disabled {
  opacity: 0.4;
  cursor: default;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}
.rcard {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-lg, 12px);
  overflow: hidden;
  background: var(--zen-surface);
  transition:
    transform var(--zen-dur) ease,
    border-color var(--zen-dur-fast) ease;
}
.rcard:hover {
  border-color: var(--zen-accent);
  transform: translateY(-2px);
}
.cover {
  position: relative;
  aspect-ratio: 16 / 9;
  display: grid;
  place-items: center;
  background:
    radial-gradient(circle at 30% 20%, color-mix(in srgb, var(--a) 90%, white 10%), transparent 60%),
    linear-gradient(135deg, var(--a), var(--b));
  color: rgba(255, 255, 255, 0.92);
  font-size: 48px;
}
.cover i {
  transition: transform var(--zen-dur) ease;
}
.rcard:hover .cover i {
  transform: scale(1.12) rotate(-4deg);
}
.badge {
  position: absolute;
  top: 10px;
  left: 10px;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.35);
  color: #fff;
  backdrop-filter: blur(6px);
}
.rbody {
  padding: 12px 14px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.rbody p {
  margin: 0;
  font-size: 13px;
  color: var(--zen-muted);
  line-height: 1.45;
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.flow {
  display: flex;
  flex-direction: column;
  gap: 14px;
  max-width: 820px;
}
.step {
  display: grid;
  grid-template-columns: 36px 1fr;
  gap: 14px;
  padding: 18px;
}
.num {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--zen-accent) 18%, transparent);
  color: var(--zen-accent);
  font-weight: 700;
}
.sbody {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}
.sbody h2 {
  margin: 4px 0 0;
  font-size: 16px;
}
.trig {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.trig-in {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
  border-radius: var(--r-sm);
  padding: 2px 6px;
  font-size: 12px;
  width: 16ch;
}
.drop {
  border: 2px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  border-radius: var(--r-lg, 12px);
  padding: 28px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--zen-muted);
  cursor: pointer;
  transition:
    border-color var(--zen-dur-fast),
    background var(--zen-dur-fast);
}
.drop i {
  font-size: 28px;
}
.drop:hover,
.drop.over {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 6%, transparent);
}
.sel {
  padding: 8px;
  border-radius: var(--r-md);
  border: 1px solid var(--zen-border);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
  font: inherit;
}
.tips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
}
.tip i {
  color: var(--zen-ok);
  margin-right: 4px;
}
.thumbs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
  gap: 6px;
}
.th {
  position: relative;
  aspect-ratio: 1;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.th img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.th .x {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: none;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.th:hover .x {
  opacity: 1;
}
.count.warn,
.warn {
  color: var(--zen-warn);
}
.checks {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.checks i {
  margin-right: 6px;
  color: var(--zen-muted);
}
.checks i.ok {
  color: var(--zen-ok);
}
.doing {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prow {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.pbar {
  height: 6px;
  border-radius: 3px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.pbar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
  transition: width var(--zen-dur);
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
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.step :deep(.zen-button) {
  align-self: flex-start;
}
</style>
