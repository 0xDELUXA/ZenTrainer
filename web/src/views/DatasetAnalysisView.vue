<script setup lang="ts">
// A dataset's analysis (engine datasets/analysis.py): what to fix before training. The pictures worth
// a look by issue, how the trainer's buckets crop them at a resolution, which caption phrases fill
// most of the set, the spread of brightness, contrast, sharpness and size, and near-duplicate pairs.
import { ZenButton, ZenEmpty, ZenToggleGroup } from '@nynxz/zenkit-ui'
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EngineError, engine, thumbUrl, type DatasetAnalysis } from '@/api/engine'

const props = defineProps<{ name: string }>()
const route = useRoute()
const router = useRouter()

const RESOLUTIONS = ['512', '768', '1024', '1280', '1536']
const resolution = ref<string>(RESOLUTIONS.includes(route.query.resolution as string) ? (route.query.resolution as string) : '1024')
const set = computed(() => (route.query.set as string) || 'default')
const data = ref<DatasetAnalysis | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

watch(
  [() => props.name, set, resolution],
  async () => {
    router.replace({ query: { ...route.query, resolution: resolution.value } })
    loading.value = true
    error.value = null
    try {
      data.value = await engine.datasetAnalysis(props.name, set.value, Number(resolution.value))
    } catch (e) {
      error.value = e instanceof EngineError ? e.message : String(e)
    } finally {
      loading.value = false
    }
  },
  { immediate: true },
)

const thumb = (path: string, size = 256) => thumbUrl(props.name, path, size)
const pct = (x: number) => `${Math.round(x * 100)}%`

// ---- pictures to check, by issue ---------------------------------------------------------------
// what each flag means, in a line (the engine's stats.image_flags)
const FLAG_INFO: Record<string, string> = {
  small: 'Shorter side under 768 px: it gets upscaled.',
  blurry: 'Much softer than the rest of the set.',
  dark: 'Very dark overall.',
  'blown out': 'Very bright overall: detail lost in the highlights.',
  flat: 'Very little contrast.',
  bars: 'Flat bars along the edges (letterboxing or a frame).',
  greyscale: 'No colour, in a set that has colour.',
  transparent: 'See-through parts: training drops transparency, so whatever colour sits under it shows.',
  compressed: 'Heavily compressed JPEG: its blocky artefacts can be learned.',
  'near-duplicate': 'A re-saved, resized, lightly edited or mirrored copy of another picture: it counts twice.',
}
const flagFilter = ref<string | null>(null)
const flagged = computed(() =>
  (data.value?.pictures ?? []).filter((p) => p.flags.length && (!flagFilter.value || p.flags.includes(flagFilter.value))),
)
const flagChips = computed(() => Object.entries(data.value?.flag_counts ?? {}).sort((a, b) => b[1] - a[1]))

// ---- buckets -------------------------------------------------------------------------------------
const maxBucket = computed(() => Math.max(1, ...(data.value?.buckets.rows.map((r) => r.count) ?? [1])))
const openBucket = ref<string | null>(null)
const bucketKey = (size: [number, number]) => size.join('x')
type Placed = { bucket: [number, number]; kept: number; size: [number, number] }
/** where the crop cuts a picture, as insets (in %) for the box over its thumbnail */
function cut(c: Placed) {
  const ar = c.size[0] / c.size[1]
  const br = c.bucket[0] / c.bucket[1]
  const lost = (1 - c.kept) * 50
  return ar > br ? { left: `${lost}%`, right: `${lost}%`, top: '0', bottom: '0' } : { top: `${lost}%`, bottom: `${lost}%`, left: '0', right: '0' }
}
const placed = (path: string) => data.value?.buckets.placed[path]

// ---- distributions ---------------------------------------------------------------------------------
// each chart over a fixed range, with the median marked and the ranges a picture gets flagged in shaded
// (the engine's stats.image_flags thresholds)
interface Hist {
  key: string
  label: string
  note: string
  bins: number[]
  lo: number
  hi: number
  median: number | null
  warn: [number, number][]
  fmt: (x: number) => string
}
const BINS = 30
function histogram(values: number[], lo: number, hi: number): number[] {
  const bins = new Array(BINS).fill(0)
  for (const v of values) bins[Math.min(BINS - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * BINS)))]++
  return bins
}
const median = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]! : null)
const hists = computed<Hist[]>(() => {
  const ps = data.value?.pictures ?? []
  const vals = (f: (p: (typeof ps)[number]) => number | null | undefined) => ps.map(f).filter((v): v is number => v != null)
  const make = (key: string, label: string, note: string, v: number[], lo: number, hi: number, warn: [number, number][], fmt: (x: number) => string): Hist => ({
    key, label, note, lo, hi, warn, fmt, bins: histogram(v, lo, hi), median: median(v),
  })
  const sides = vals((p) => (p.size ? Math.min(...p.size) : null))
  const maxSide = Math.max(2048, Math.ceil(Math.max(0, ...sides) / 512) * 512)
  const ratio = (x: number) => (Math.abs(x) < 0.05 ? '1:1' : x > 0 ? `${(2 ** x).toFixed(1)}:1` : `1:${(2 ** -x).toFixed(1)}`)
  return [
    make('brightness', 'Brightness', 'Dark to bright. Shaded: flagged dark or blown out.', vals((p) => p.brightness), 0, 1, [[0, 0.15], [0.85, 1]], pct),
    make('contrast', 'Contrast', 'Flat to punchy. Shaded: flagged flat.', vals((p) => p.contrast), 0, 0.4, [[0, 0.06]], (x) => x.toFixed(2)),
    make('sharpness', 'Sharpness', 'Against the set’s median (1×). Shaded: flagged blurry.',
      vals((p) => (p.sharpness ? Math.log2(p.sharpness) : null)), -4, 4, [[-4, -2]], (x) => `${2 ** x < 1 ? +(2 ** x).toFixed(2) : Math.round(2 ** x)}×`),
    make('side', 'Shorter side', 'Pixels. Shaded: under 768, upscaled to train.', sides, 0, maxSide, [[0, 768]], (x) => `${Math.round(x)}`),
    make('aspect', 'Aspect ratio', 'Tall on the left, wide on the right.', vals((p) => p.aspect), -2, 2, [], ratio),
  ]
})
const histMax = (h: Hist) => Math.max(1, ...h.bins)
const at = (h: Hist, x: number) => `${((Math.min(h.hi, Math.max(h.lo, x)) - h.lo) / (h.hi - h.lo)) * 100}%`
</script>

<template>
  <div class="page full analysis">
    <div class="page-head">
      <div class="titles">
        <h1>
          <RouterLink to="/datasets" class="crumb">Datasets</RouterLink>
          <RouterLink :to="`/datasets/${encodeURIComponent(name)}`" class="crumb">{{ name }}</RouterLink>
          Analysis
        </h1>
        <p v-if="data" class="lede">
          {{ data.pictures.length }} pictures · {{ data.pictures.filter((p) => p.flags.length).length }} worth a look ·
          {{ data.captions.captions }} captioned<template v-if="data.set !== 'default'"> · caption set “{{ data.set }}”</template>
          <i v-if="loading" class="mdi mdi-loading mdi-spin" />
        </p>
      </div>
      <RouterLink :to="`/datasets/${encodeURIComponent(name)}`"><ZenButton size="sm" icon="mdi mdi-arrow-left">Back to the dataset</ZenButton></RouterLink>
    </div>

    <div v-if="error" class="card err small"><i class="mdi mdi-alert-circle-outline" /> {{ error }}</div>
    <div v-else-if="!data" class="card muted small"><i class="mdi mdi-loading mdi-spin" /> Looking at every picture… the first time takes a moment; after that only changed ones are read again.</div>

    <template v-if="data">
      <!-- pictures to check -->
      <section class="card col">
        <div class="head">
          <h2>Pictures to check</h2>
          <span class="muted small">Click an issue to show only those.</span>
        </div>
        <div v-if="flagChips.length" class="chips">
          <button class="chip" :class="{ on: !flagFilter }" @click="flagFilter = null">All <span class="muted">{{ data.pictures.filter((p) => p.flags.length).length }}</span></button>
          <button v-for="[f, n] in flagChips" :key="f" class="chip" :class="{ on: flagFilter === f }" :title="FLAG_INFO[f]" @click="flagFilter = flagFilter === f ? null : f">
            {{ f }} <span class="muted">{{ n }}</span>
          </button>
        </div>
        <p v-if="flagFilter && FLAG_INFO[flagFilter]" class="muted small">{{ FLAG_INFO[flagFilter] }}</p>
        <ZenEmpty v-if="!flagged.length" icon="mdi mdi-check-circle-outline" title="Nothing to fix">No picture has an image issue.</ZenEmpty>
        <div v-else class="grid">
          <figure v-for="p in flagged.slice(0, 120)" :key="p.path" class="tile">
            <img :src="thumb(p.path)" loading="lazy" :alt="p.path" />
            <figcaption>
              <span class="ellipsis small mono" :title="p.path">{{ p.path }}</span>
              <span class="tags"><span v-for="f in p.flags" :key="f" class="tag" :title="FLAG_INFO[f]">{{ f }}</span></span>
            </figcaption>
          </figure>
        </div>
        <p v-if="flagged.length > 120" class="muted small">and {{ flagged.length - 120 }} more</p>
      </section>

      <!-- buckets -->
      <section class="card col">
        <div class="head">
          <h2>Buckets</h2>
          <ZenToggleGroup v-model="resolution" :options="RESOLUTIONS.map((r) => ({ value: r, label: r }))" size="sm" />
        </div>
        <p class="muted small">
          Each picture trains at the bucket closest to its shape, about {{ data.buckets.resolution }}² pixels, cropped to fit.
          Half the pictures keep at least {{ pct(data.buckets.kept_median) }} of themselves. Open a bucket to see each crop.
        </p>
        <div class="bars">
          <template v-for="r in data.buckets.rows" :key="bucketKey(r.size)">
            <button class="bar-row clickable" :class="{ open: openBucket === bucketKey(r.size) }" @click="openBucket = openBucket === bucketKey(r.size) ? null : bucketKey(r.size)">
              <span class="mono small label"><i :class="openBucket === bucketKey(r.size) ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" /> {{ r.size[0] }}×{{ r.size[1] }}</span>
              <span class="track"><span class="fill" :style="{ width: pct(r.count / maxBucket) }" /></span>
              <span class="mono small count">{{ r.count }}</span>
            </button>
            <div v-if="openBucket === bucketKey(r.size)" class="bucket-pics">
              <figure v-for="path in r.pictures.slice(0, 60)" :key="path" class="tile">
                <div class="crop">
                  <img :src="thumb(path)" loading="lazy" :alt="path" />
                  <span v-if="placed(path)" class="kept" :style="cut(placed(path)!)" />
                </div>
                <figcaption>
                  <span class="ellipsis small mono" :title="path">{{ path }}</span>
                  <span v-if="placed(path)" class="small" :class="{ warn: placed(path)!.kept < 0.85 }">keeps {{ pct(placed(path)!.kept) }}</span>
                </figcaption>
              </figure>
              <p v-if="r.pictures.length > 60" class="muted small">and {{ r.pictures.length - 60 }} more</p>
            </div>
          </template>
        </div>
        <p v-if="data.buckets.lonely.length" class="warn small">
          <i class="mdi mdi-account-alert-outline" /> {{ data.buckets.lonely.length }} {{ data.buckets.lonely.length === 1 ? 'picture is' : 'pictures are' }} alone in a bucket:
          with a batch size above 1 it can't share a batch.
        </p>
        <template v-if="data.buckets.cropped.length">
          <div class="eyebrow">Cropped the most</div>
          <div class="grid">
            <figure v-for="c in data.buckets.cropped.slice(0, 24)" :key="c.path" class="tile">
              <div class="crop">
                <img :src="thumb(c.path)" loading="lazy" :alt="c.path" />
                <span class="kept" :style="cut(c)" />
              </div>
              <figcaption>
                <span class="ellipsis small mono" :title="c.path">{{ c.path }}</span>
                <span class="small">keeps <b>{{ pct(c.kept) }}</b> · {{ c.bucket[0] }}×{{ c.bucket[1] }}</span>
              </figcaption>
            </figure>
          </div>
        </template>
      </section>

      <!-- captions -->
      <section class="card col">
        <div class="head">
          <h2>Captions</h2>
          <span v-if="data.captions.style" class="muted small">read as {{ data.captions.style === 'tags' ? 'tags' : 'sentences' }}<template v-if="data.captions.subject">, leaving out “{{ data.captions.subject }}”</template></span>
        </div>
        <ZenEmpty v-if="!data.captions.captions" icon="mdi mdi-text-box-remove-outline" title="No captions">Caption the set to see what it repeats.</ZenEmpty>
        <template v-else>
          <div v-if="data.captions.frequent.length" class="frequent">
            <i class="mdi mdi-alert-outline" />
            <div>
              <b>Most of the set has these:</b>&nbsp;<span v-for="(f, i) in data.captions.frequent" :key="f.text"><template v-if="i">, </template>“{{ f.text }}” ({{ pct(f.share) }})</span>.
              <div class="muted small">
                The LoRA will lean towards them. If one isn't part of what you're training, add pictures without it, or vary it.
              </div>
            </div>
          </div>
          <div class="bars">
            <div v-for="p in data.captions.phrases" :key="p.text" class="bar-row">
              <span class="small label ellipsis" :title="p.text">{{ p.text }}</span>
              <span class="track"><span class="fill" :class="{ hot: p.share > 0.5 }" :style="{ width: pct(p.share) }" /></span>
              <span class="mono small count">{{ pct(p.share) }}</span>
            </div>
          </div>
        </template>
      </section>

      <!-- distributions -->
      <section class="card col">
        <div class="head"><h2>Spread</h2></div>
        <div class="hists">
          <div v-for="h in hists" :key="h.key" class="hist">
            <div class="hist-head">
              <b>{{ h.label }}</b>
              <span v-if="h.median != null" class="mono small muted">median {{ h.fmt(h.median) }}</span>
            </div>
            <div class="plot">
              <span v-for="(w, i) in h.warn" :key="i" class="zone" :style="{ left: at(h, w[0]), width: `calc(${at(h, w[1])} - ${at(h, w[0])})` }" />
              <div class="hbars">
                <span v-for="(b, i) in h.bins" :key="i" class="hb" :style="{ height: b ? `${Math.max(4, (b / histMax(h)) * 100)}%` : '0' }" :title="`${b} pictures`" />
              </div>
              <span v-if="h.median != null" class="med" :style="{ left: at(h, h.median) }" />
            </div>
            <div class="axis small muted mono"><span>{{ h.fmt(h.lo) }}</span><span>{{ h.fmt((h.lo + h.hi) / 2) }}</span><span>{{ h.fmt(h.hi) }}</span></div>
            <p class="note small muted">{{ h.note }}</p>
          </div>
        </div>
      </section>

      <!-- near-duplicates -->
      <section class="card col">
        <div class="head">
          <h2>Near-duplicates</h2>
          <span class="muted small">Copies of the same picture (re-saved, resized, lightly edited or mirrored) count twice in training.</span>
        </div>
        <ZenEmpty v-if="!data.duplicates.length" icon="mdi mdi-content-duplicate" title="None">No two pictures are close copies.</ZenEmpty>
        <div v-else class="pairs">
          <div v-for="d in data.duplicates.slice(0, 40)" :key="d.a + d.b" class="pair">
            <img :src="thumb(d.a, 192)" loading="lazy" :alt="d.a" :title="d.a" />
            <img :src="thumb(d.b, 192)" loading="lazy" :alt="d.b" :title="d.b" />
            <span class="small"><b>{{ d.kind }}</b><span v-if="d.mirrored" class="muted"> · mirrored</span></span>
          </div>
        </div>
      </section>

    </template>
  </div>
</template>

<style scoped>
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
  flex-wrap: wrap;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}
.head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.head h2 {
  margin: 0;
  font-size: 15px;
  flex: 1;
}
.err {
  color: var(--zen-danger);
}
.warn {
  color: var(--zen-warn);
  margin: 0;
}

/* ---- chips and picture tiles -------------------------------------------------------------------- */
.chips {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.chip {
  border: 1px solid var(--zen-border);
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 99px;
  cursor: pointer;
}
.chip:hover {
  background: var(--zen-surface-2);
}
.chip.on {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 12%, transparent);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
}
.tile {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.tile img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  background: var(--zen-surface-2);
  border-radius: var(--r-sm);
  display: block;
}
.tile figcaption {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}
.tags {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.tag {
  font-size: 10.5px;
  padding: 0 6px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--zen-warn) 15%, transparent);
  color: var(--zen-warn);
}
/* the tile takes the picture's own shape, so the kept box lines up with it */
.crop {
  position: relative;
  border-radius: var(--r-sm);
  overflow: hidden;
}
.crop img {
  aspect-ratio: auto;
  height: auto;
  border-radius: 0;
}
/* the part the crop keeps: everything around it shaded */
.kept {
  position: absolute;
  outline: 2px solid var(--zen-accent);
  box-shadow: 0 0 0 999px color-mix(in srgb, black 55%, transparent);
  pointer-events: none;
}

/* ---- bars --------------------------------------------------------------------------------------- */
.bars {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.bar-row {
  display: grid;
  grid-template-columns: minmax(90px, 220px) minmax(0, 1fr) 48px;
  gap: 10px;
  align-items: center;
}
.label {
  min-width: 0;
}
.track {
  height: 8px;
  border-radius: 99px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  border-radius: 99px;
  background: color-mix(in srgb, var(--zen-text) 35%, transparent);
}
.fill.hot {
  background: var(--zen-warn);
}
.count {
  text-align: right;
  color: var(--zen-muted);
}
.frequent {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-warn) 40%, transparent);
  background: color-mix(in srgb, var(--zen-warn) 8%, transparent);
  font-size: 13px;
}
.frequent > i {
  color: var(--zen-warn);
  font-size: 18px;
}

/* ---- histograms: one shape each, so they line up --------------------------------------------- */
.hists {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 22px 28px;
}
.hist {
  display: grid;
  grid-template-rows: 20px 84px auto auto;
  gap: 6px;
  min-width: 0;
}
.hist-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  font-size: 13px;
  white-space: nowrap;
}
.plot {
  position: relative;
  border-bottom: 1px solid var(--zen-border);
}
.zone {
  position: absolute;
  top: 0;
  bottom: 0;
  background: color-mix(in srgb, var(--zen-warn) 10%, transparent);
  border-left: 1px dashed color-mix(in srgb, var(--zen-warn) 40%, transparent);
  border-right: 1px dashed color-mix(in srgb, var(--zen-warn) 40%, transparent);
}
.hbars {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: flex-end;
  gap: 2px;
}
.hb {
  flex: 1;
  background: color-mix(in srgb, var(--zen-text) 45%, transparent);
  border-radius: 2px 2px 0 0;
}
.med {
  position: absolute;
  top: -2px;
  bottom: 0;
  width: 2px;
  margin-left: -1px;
  background: var(--zen-accent);
}
.axis {
  display: flex;
  justify-content: space-between;
}
.note {
  margin: 0;
}

/* ---- a bucket's pictures, opened ---------------------------------------------------------------- */
button.bar-row {
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  padding: 3px 4px;
  border-radius: var(--r-sm);
  cursor: pointer;
  text-align: left;
}
button.bar-row:hover,
button.bar-row.open {
  background: var(--zen-surface-2);
}
.bucket-pics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 10px;
  align-items: start;
  padding: 10px 4px 14px;
}
.bucket-pics p {
  grid-column: 1 / -1;
}

/* ---- pairs -------------------------------------------------------------------------------------- */
.pairs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 12px;
}
.pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
}
.pair img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
}
.pair span {
  grid-column: 1 / -1;
}
@media (max-width: 600px) {
  .bar-row {
    grid-template-columns: minmax(70px, 120px) minmax(0, 1fr) 40px;
  }
}
</style>
