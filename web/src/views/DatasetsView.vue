<script setup lang="ts">
// Every dataset folder in ComfyUI's input directory: as cards (a fixed-shape mosaic, so every
// card is the same size) or as a table that sorts. The view is remembered per browser.
import { ZenButton, ZenEmpty, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { engine, thumbUrl, type DatasetInfo } from '@/api/engine'
import HealthPentagon from '@/components/HealthPentagon.vue'
import TemplateCard from '@/components/TemplateCard.vue'

const router = useRouter()
const datasets = ref<DatasetInfo[] | null>(null)
const error = ref<string | null>(null)
const creating = ref(false)
const newName = ref('')
const query = ref('')

const VIEW = 'zentrainer.datasets.view'
const read = (k: string, d: string) => {
  try {
    return localStorage.getItem(k) ?? d
  } catch {
    return d
  }
}
const view = ref<'cards' | 'table'>(read(VIEW, 'cards') === 'table' ? 'table' : 'cards')
watch(view, (v) => {
  try {
    localStorage.setItem(VIEW, v)
  } catch {
    // storage blocked
  }
})

onMounted(async () => {
  try {
    datasets.value = await engine.datasets()
  } catch (e) {
    error.value = String(e)
  }
})

async function create() {
  const name = newName.value.trim()
  if (!name) return
  try {
    await engine.createDataset(name)
    router.push(`/datasets/${encodeURIComponent(name)}`)
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  }
}

type Key = 'name' | 'items' | 'captioned' | 'size' | 'modified'
const sortKey = ref<Key>('modified')
const sortDir = ref(-1)
function sortBy(k: Key) {
  if (sortKey.value === k) sortDir.value = -sortDir.value
  else {
    sortKey.value = k
    sortDir.value = k === 'name' ? 1 : -1
  }
}
const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  const list = (datasets.value ?? []).filter((d) => !q || d.folder.toLowerCase().includes(q))
  const val = (d: DatasetInfo): string | number =>
    ({ name: d.folder.toLowerCase(), items: d.images, captioned: d.images ? d.captioned / d.images : 0, size: d.bytes ?? 0, modified: d.modified ?? 0 })[sortKey.value]
  return [...list].sort((a, b) => (val(a) < val(b) ? -1 : val(a) > val(b) ? 1 : 0) * sortDir.value)
})

const link = (d: DatasetInfo) => `/datasets/${encodeURIComponent(d.folder)}`
const thumb = (d: DatasetInfo, t: DatasetInfo['thumbs'][number], size = 256) =>
  thumbUrl(d.folder, t.path ?? t.subfolder.split('/').slice(1).concat(t.filename).join('/'), size)
const pics = (d: DatasetInfo) => d.thumbs.slice(0, 4)
const counts = (d: DatasetInfo) => {
  const v = d.videos ?? 0
  const im = d.images - v
  return [im ? `${im} image${im === 1 ? '' : 's'}` : '', v ? `${v} video${v === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ')
}
const tagsFor = (d: DatasetInfo) => {
  const v = d.videos ?? 0
  const im = d.images - v
  return [im ? `${im} image${im === 1 ? '' : 's'}` : '', v ? `${v} video${v === 1 ? '' : 's'}` : '', d.masked ? `${d.masked} masked` : ''].filter(Boolean)
}
const fmtBytes = (b?: number) => (b == null ? '' : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 1e6))} MB`)
const ago = (t?: number) => {
  if (!t) return ''
  const s = Date.now() / 1000 - t
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  if (s < 86400 * 30) return `${Math.round(s / 86400)}d ago`
  return new Date(t * 1000).toLocaleDateString()
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Datasets</h1>
        <p v-if="datasets?.length" class="lede">{{ datasets.length }} in ComfyUI's input folder</p>
      </div>
      <form v-if="creating" class="create" @submit.prevent="create">
        <ZenInput v-model="newName" placeholder="dataset name" />
        <ZenButton variant="primary" :disabled="!newName.trim()" @click="create">Create</ZenButton>
        <ZenButton variant="ghost" @click="creating = false">Cancel</ZenButton>
      </form>
      <ZenButton v-else variant="primary" icon="mdi mdi-plus" @click="creating = true">New dataset</ZenButton>
    </div>
    <div v-if="error" class="card muted">Can't reach the engine: {{ error }}</div>
    <ZenEmpty v-else-if="datasets && !datasets.length" title="No datasets yet" icon="mdi mdi-image-multiple-outline">
      Put a folder of images and captions in ComfyUI's input directory.
    </ZenEmpty>
    <template v-else-if="datasets">
      <div class="bar">
        <div class="seg">
          <button :class="{ on: view === 'cards' }" title="Cards" @click="view = 'cards'"><i class="mdi mdi-view-grid-outline" /></button>
          <button :class="{ on: view === 'table' }" title="Table" @click="view = 'table'"><i class="mdi mdi-view-list-outline" /></button>
        </div>
        <div class="search"><ZenInput v-model="query" placeholder="Search datasets" size="sm" /></div>
      </div>

      <!-- cards: one shape for every card -->
      <div v-if="view === 'cards'" class="grid">
        <TemplateCard
          v-for="d in shown"
          :key="d.folder"
          :to="link(d)"
          :title="d.folder"
          :tags="tagsFor(d)"
          :meta="[ago(d.modified), fmtBytes(d.bytes)].filter(Boolean)"
        >
          <template #thumb>
            <div class="mosaic" :class="`n${pics(d).length}`">
              <img v-for="t in pics(d)" :key="t.filename" :src="thumb(d, t)" loading="lazy" alt="" />
              <span v-if="!pics(d).length" class="none"><i class="mdi mdi-image-off-outline" /></span>
            </div>
          </template>
          <div class="capline" :title="`${d.captioned} of ${d.images} have a default caption`">
            <HealthPentagon v-if="d.health" :scores="d.health" :size="30" class="card-pent" />
            <div class="meter"><span :class="{ part: d.captioned < d.images }" :style="{ width: `${d.images ? (d.captioned / d.images) * 100 : 0}%` }" /></div>
            <span class="small" :class="d.captioned < d.images ? 'warn' : 'muted'">{{ d.captioned }}/{{ d.images }} captioned</span>
          </div>
        </TemplateCard>
      </div>

      <!-- table -->
      <div v-else class="table card">
        <div class="tr th">
          <span />
          <button class="sortable" @click="sortBy('name')">Name <i v-if="sortKey === 'name'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
          <button class="sortable" @click="sortBy('items')">Items <i v-if="sortKey === 'items'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
          <button class="sortable" @click="sortBy('captioned')">Captioned <i v-if="sortKey === 'captioned'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
          <span class="hide-s">Caption sets</span>
          <button class="sortable hide-s" @click="sortBy('size')">Size <i v-if="sortKey === 'size'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
          <button class="sortable hide-s" @click="sortBy('modified')">Changed <i v-if="sortKey === 'modified'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
          <span />
        </div>
        <div v-for="d in shown" :key="d.folder" class="tr" @click="router.push(link(d))">
          <span class="strip">
            <img v-for="t in pics(d).slice(0, 3)" :key="t.filename" :src="thumb(d, t, 128)" loading="lazy" alt="" />
          </span>
          <RouterLink :to="link(d)" class="name ellipsis" :title="d.folder" @click.stop>{{ d.folder }}</RouterLink>
          <span class="small">{{ counts(d) }}<span v-if="d.masked" class="muted"> · {{ d.masked }} masked</span></span>
          <span class="capline">
            <span class="meter"><span :class="{ part: d.captioned < d.images }" :style="{ width: `${d.images ? (d.captioned / d.images) * 100 : 0}%` }" /></span>
            <span class="small mono" :class="d.captioned < d.images ? 'warn' : 'muted'">{{ d.captioned }}/{{ d.images }}</span>
          </span>
          <span class="sets hide-s">
            <span v-for="st in d.sets" :key="st" class="set">{{ st }}</span>
          </span>
          <span class="small mono muted hide-s">{{ fmtBytes(d.bytes) }}</span>
          <span class="small muted hide-s">{{ ago(d.modified) }}</span>
          <RouterLink :to="{ path: '/runs/new', query: { dataset: d.folder } }" @click.stop>
            <ZenButton size="sm" icon="mdi mdi-play">Train</ZenButton>
          </RouterLink>
        </div>
        <div v-if="!shown.length" class="muted small empty">No dataset matches “{{ query }}”.</div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.create {
  display: flex;
  gap: 8px;
  align-items: center;
}
.bar {
  display: flex;
  gap: 10px;
  align-items: center;
}
.search {
  width: 260px;
}
.seg {
  display: inline-flex;
  padding: 2px;
  gap: 2px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
}
.seg button {
  border: none;
  background: none;
  color: var(--zen-muted);
  width: 30px;
  height: 26px;
  border-radius: calc(var(--r-md) - 2px);
  cursor: pointer;
}
.seg button.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
}
/* a fixed 4:3 mosaic whatever the thumbnails' shapes or count */
.mosaic {
  display: grid;
  height: 100%;
  gap: 2px;
  background: var(--zen-surface-2);
  grid-template: 1fr 1fr / 1fr 1fr;
  overflow: hidden;
  transition: opacity var(--zen-dur-fast) ease;
}
.mosaic img {
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  object-fit: cover;
  display: block;
}
.mosaic.n0,
.mosaic.n1 {
  grid-template: 1fr / 1fr;
}
.mosaic.n2 {
  grid-template: 1fr / 1fr 1fr;
}
.mosaic.n3 img:first-child {
  grid-row: span 2;
}
.none {
  display: grid;
  place-items: center;
  color: var(--zen-muted);
  font-size: 28px;
}
.name {
  font-weight: 650;
  font-size: 14px;
  min-width: 0;
}
.capline {
  display: flex;
  align-items: center;
  gap: 8px;
}
.meter {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.meter span {
  display: block;
  height: 100%;
  background: var(--zen-ok, var(--zen-accent));
}
.meter span.part {
  background: var(--zen-warn);
}
.spacer {
  flex: 1;
}
.small {
  font-size: 11px;
}
.warn {
  color: var(--zen-warn);
}
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.table {
  padding: 0;
  overflow: hidden;
}
.tr {
  display: grid;
  grid-template-columns: 112px minmax(140px, 1.4fr) 150px 150px minmax(120px, 1fr) 70px 80px 84px;
  align-items: center;
  gap: 12px;
  padding: 8px 14px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  cursor: pointer;
}
.tr:not(.th):hover {
  background: color-mix(in srgb, var(--zen-text) 4%, transparent);
}
.tr.th {
  border-top: none;
  cursor: default;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--zen-muted);
}
.sortable {
  border: none;
  background: none;
  padding: 0;
  text-align: left;
  font: inherit;
  text-transform: inherit;
  letter-spacing: inherit;
  color: inherit;
  cursor: pointer;
}
.sortable:hover {
  color: var(--zen-text);
}
.strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2px;
  height: 36px;
  border-radius: var(--r-sm);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.strip img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.sets {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.set {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: var(--r-sm);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  color: var(--zen-muted);
}
.empty {
  padding: 16px;
}
@media (max-width: 900px) {
  .tr {
    grid-template-columns: 80px minmax(0, 1fr) 110px 84px;
  }
  .hide-s,
  .tr > :nth-child(3) {
    display: none;
  }
}
.card-pent {
  flex: none;
}
</style>
