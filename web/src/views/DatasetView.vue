<script setup lang="ts">
// One dataset. Grid (uniform cards, zoomable) or table (sortable, inline captions);
// click any image for the lightbox with a caption editor beside it. Drop files anywhere
// to upload: images, same-name .txt captions, and masks (<name>-masklabel.png).
import { ZenButton, ZenEmpty, ZenInput, ZenLightbox, ZenSlider } from '@nynxz/zenkit-ui'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { engine, thumbUrl, viewUrl, type CaptionJob, type DatasetItem } from '@/api/engine'
import { activeCaptions } from '@/captions'
import CaptionPanel from '@/components/CaptionPanel.vue'
import { vAutosize } from '@/autosize'

const props = defineProps<{ name: string }>()
const router = useRouter()
const route = useRoute()

// ---- persisted view prefs -------------------------------------------------------------
const PREFS = 'zentrainer.datasetView'
const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem(PREFS) ?? '{}')
  } catch {
    return {}
  }
})()
const view = ref<'grid' | 'list'>(saved.view ?? 'grid')
const zoom = ref<number>(saved.zoom ?? 200)
const fit = ref<'cover' | 'contain'>(saved.fit ?? 'cover')
const showMasks = ref<boolean>(saved.masks ?? false) // dim what each mask leaves out, on the grid
watch([view, zoom, fit, showMasks], () => {
  try {
    localStorage.setItem(PREFS, JSON.stringify({ view: view.value, zoom: zoom.value, fit: fit.value, masks: showMasks.value }))
  } catch {
    // storage blocked
  }
})

// ---- data -------------------------------------------------------------------------------
const items = ref<DatasetItem[]>([])
// caption sets: <img>.txt is "default", <img>.<set>.txt the others; everything below
// (view, edit, filters, bulk) works on the selected set
const sets = ref<string[]>(['default'])
const activeSet = ref('default')
const cap = (it: DatasetItem) => it.captions?.[activeSet.value] ?? ''
const k = (it: DatasetItem) => `${activeSet.value}:${it.path}`
function addSet() {
  const name = prompt('New caption set (e.g. tags, nl, short):')?.trim().toLowerCase()
  if (!name) return
  if (!/^[a-z0-9][a-z0-9_-]{0,23}$/.test(name)) return alert('Use lowercase letters, digits, - and _ (max 24).')
  if (!sets.value.includes(name)) sets.value = [...sets.value, name]
  activeSet.value = name
}
const loading = ref(true)
const error = ref<string | null>(null)
const notice = ref<string | null>(null)
const drafts = reactive<Record<string, string>>({})
const saving = reactive<Record<string, boolean>>({})
const bust = ref(Date.now())

async function load() {
  try {
    const r = await engine.dataset(props.name)
    items.value = r.items
    sets.value = [...new Set([...r.sets, ...sets.value])].sort((a, b) => (a === 'default' ? -1 : b === 'default' ? 1 : a.localeCompare(b)))
    error.value = null
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    loading.value = false
  }
}
onMounted(load)

// ---- filter / sort / select -------------------------------------------------------------------
type Filter = 'all' | 'uncaptioned' | 'masked' | 'unmasked'
const filter = ref<Filter>('all')
const query = ref('')
const sortKey = ref<'name' | 'size' | 'res' | 'caption'>('name')
const sortDir = ref<1 | -1>(1)
const selected = reactive(new Set<string>())
// Review on a caption job: only the images it wrote
const reviewing = ref<{ job: CaptionJob; paths: Set<string> } | null>(null)
function review(job: CaptionJob) {
  reviewing.value = { job, paths: new Set(job.written ?? []) }
  activeSet.value = job.set_name
  filter.value = 'all'
}
// ?review=<caption job id>: opened from that job's page
onMounted(async () => {
  const id = route.query.review
  if (typeof id === 'string' && id) review(await engine.captionJob(id))
})

const counts = computed(() => ({
  uncaptioned: items.value.filter((i) => !cap(i)).length,
  masked: items.value.filter((i) => i.mask).length,
  mismatched: items.value.filter((i) => i.mask_mismatch).length,
  videos: items.value.filter((i) => i.kind === 'video').length,
}))
const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  const f = filter.value
  const only = reviewing.value?.paths
  const rows = items.value.filter(
    (it) =>
      (!only || only.has(it.path)) &&
      (f === 'all' ||
        (f === 'uncaptioned' && !cap(it)) ||
        (f === 'masked' && it.mask) ||
        (f === 'unmasked' && !it.mask)) &&
      (!q || cap(it).toLowerCase().includes(q) || it.path.toLowerCase().includes(q)),
  )
  const key = (it: DatasetItem): string | number =>
    sortKey.value === 'size'
      ? it.bytes
      : sortKey.value === 'res'
        ? (it.width ?? 0) * (it.height ?? 0)
        : sortKey.value === 'caption'
          ? cap(it).length
          : it.path.toLowerCase()
  return [...rows].sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * sortDir.value)
})
function sortBy(k: typeof sortKey.value) {
  if (sortKey.value === k) sortDir.value = sortDir.value === 1 ? -1 : 1
  else {
    sortKey.value = k
    sortDir.value = 1
  }
}
let lastClicked: number | null = null
function toggleSelect(it: DatasetItem, i: number, e?: MouseEvent) {
  if (e?.shiftKey && lastClicked != null) {
    const [a, b] = [Math.min(lastClicked, i), Math.max(lastClicked, i)]
    for (const row of shown.value.slice(a, b + 1)) selected.add(row.path)
  } else if (selected.has(it.path)) selected.delete(it.path)
  else selected.add(it.path)
  lastClicked = i
}
const allShownSelected = computed(() => shown.value.length > 0 && shown.value.every((r) => selected.has(r.path)))
function onEsc(e: KeyboardEvent) {
  if (e.key !== 'Escape' || lightbox.value != null || !selected.size) return
  if ((e.target as HTMLElement).closest('input, textarea, [contenteditable]')) return
  selected.clear()
}
function toggleAll() {
  if (allShownSelected.value) for (const r of shown.value) selected.delete(r.path)
  else for (const r of shown.value) selected.add(r.path)
}

// ---- editing ------------------------------------------------------------------------------
function flash(msg: string) {
  notice.value = msg
  setTimeout(() => notice.value === msg && (notice.value = null), 2500)
}
async function save(it: DatasetItem) {
  const key = k(it)
  const set = activeSet.value
  const text = drafts[key]
  if (text == null || text.trim() === cap(it)) {
    delete drafts[key]
    return
  }
  saving[key] = true
  try {
    await engine.setCaption(props.name, it.path, text, set)
    it.captions = { ...it.captions, [set]: text.trim() }
    delete drafts[key]
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    saving[key] = false
  }
}
async function removeMany(paths: string[]) {
  if (!paths.length || !confirm(`Delete ${paths.length} image${paths.length === 1 ? '' : 's'} with their captions and masks?`)) return
  for (const p of paths) await engine.deleteItem(props.name, p)
  for (const p of paths) selected.delete(p)
  items.value = items.value.filter((x) => !paths.includes(x.path))
  if (lightbox.value != null && lightbox.value >= shown.value.length) lightbox.value = shown.value.length ? shown.value.length - 1 : null
  flash(`${paths.length} deleted`)
}

const bulkOpen = ref(false)
const bulkText = ref('')
const findText = ref('')
async function bulk(op: 'prefix' | 'suffix' | 'replace') {
  if (!bulkText.value && op !== 'replace') return
  const { changed } = await engine.bulkCaptions(props.name, op, bulkText.value, findText.value, activeSet.value)
  flash(`${changed} caption${changed === 1 ? '' : 's'} updated`)
  await load()
}

// ---- auto-caption: CaptionPanel runs it as a job -----------------------------------------
// open while this dataset has a caption job going, so coming back to the page shows it
const capRunning = computed(() => activeCaptions.value.some((j) => j.dataset === props.name))
const capOpen = ref(capRunning.value)
watch(capRunning, (v) => v && (capOpen.value = true))
function onCaptioned(set: string) {
  if (!sets.value.includes(set)) sets.value = [...sets.value, set]
  load()
}

// the image's captions in the other sets, to compare (and copy one across)
const otherSets = (it: DatasetItem) => Object.entries(it.captions ?? {}).filter(([st, t]) => st !== activeSet.value && t)
async function useCaption(it: DatasetItem, text: string) {
  drafts[k(it)] = text
  await save(it)
}

// ---- upload -------------------------------------------------------------------------------
const dragging = ref(false)
const uploading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
async function upload(files: File[]) {
  if (!files.length) return
  uploading.value = true
  try {
    const r = await engine.upload(props.name, files)
    flash(`${r.saved.length} file${r.saved.length === 1 ? '' : 's'} added` + (r.errors.length ? `, ${r.errors.length} skipped` : ''))
    if (r.errors.length) error.value = r.errors.join('\n')
    bust.value = Date.now()
    await load()
  } finally {
    uploading.value = false
  }
}
let depth = 0
const onEnter = (e: DragEvent) => {
  if (e.dataTransfer?.types.includes('Files')) {
    depth++
    dragging.value = true
  }
}
const onLeave = () => {
  if (--depth <= 0) {
    depth = 0
    dragging.value = false
  }
}
const onOver = (e: DragEvent) => e.preventDefault()
const onDrop = (e: DragEvent) => {
  e.preventDefault()
  depth = 0
  dragging.value = false
  upload([...(e.dataTransfer?.files ?? [])])
}
onMounted(() => {
  addEventListener('dragenter', onEnter)
  addEventListener('dragleave', onLeave)
  addEventListener('dragover', onOver)
  addEventListener('drop', onDrop)
  addEventListener('keydown', onEsc)
})
onUnmounted(() => {
  removeEventListener('dragenter', onEnter)
  removeEventListener('dragleave', onLeave)
  removeEventListener('dragover', onOver)
  removeEventListener('drop', onDrop)
  removeEventListener('keydown', onEsc)
})

// ---- lightbox -----------------------------------------------------------------------------
const lightbox = ref<number | null>(null)
const lbItems = computed<LightboxItem[]>(() =>
  shown.value.map((it) => ({
    src: `${viewUrl({ ...it, type: 'input' })}&v=${bust.value}`,
    thumb: thumb(it, 256),
    label: it.path,
    meta: it.width ? `${it.width}×${it.height} · ${kb(it.bytes)}` : kb(it.bytes),
    overlay: it.mask ? thumb({ path: it.mask } as DatasetItem, 1024) : undefined,
    kind: it.kind === 'video' ? ('video' as const) : undefined,
  })),
)

const thumb = (it: Pick<DatasetItem, 'path'>, size = 384) => `${thumbUrl(props.name, it.path, size)}&v=${bust.value}`
const secs = (d?: number | null) => (d == null ? '' : d < 60 ? `${d.toFixed(1)}s` : `${Math.floor(d / 60)}:${String(Math.round(d % 60)).padStart(2, '0')}`)
const pct = (v?: number) => (v == null ? '' : `${Math.round(v * 100)}%`)
const kb = (b: number) => (b > 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(1)} MB` : `${Math.round(b / 1024)} KB`)
const thumbSize = computed(() => (zoom.value > 260 ? 768 : 384))
</script>

<template>
  <div class="page wide">
    <div class="page-head">
      <div class="titles">
        <RouterLink to="/datasets" class="crumb">Datasets</RouterLink>
        <h1>{{ name }}</h1>
        <p class="lede">
          {{ items.length - counts.videos }} image{{ items.length - counts.videos === 1 ? '' : 's' }}<template v-if="counts.videos"> · {{ counts.videos }} video{{ counts.videos === 1 ? '' : 's' }}</template>
          <template v-if="counts.uncaptioned"> · <span class="warn">{{ counts.uncaptioned }} uncaptioned</span></template>
          <template v-if="counts.masked"> · {{ counts.masked }} masked</template>
          <template v-if="counts.mismatched"> · <span class="warn" title="A mask with a different shape from its image is stretched over it and won't line up">{{ counts.mismatched }} mask{{ counts.mismatched === 1 ? '' : 's' }} don't fit</span></template>
        </p>
      </div>
      <ZenButton icon="mdi mdi-upload" :disabled="uploading" @click="fileInput?.click()">Add files</ZenButton>
      <ZenButton variant="primary" icon="mdi mdi-play" @click="router.push({ path: '/runs/new', query: { dataset: name } })"
        >Train on this</ZenButton
      >
      <input ref="fileInput" type="file" multiple accept="image/*,video/*,.txt" hidden @change="upload([...(($event.target as HTMLInputElement).files ?? [])])" />
    </div>

    <div v-if="error" class="card err mono">{{ error }}</div>

    <!-- with a selection the same bar swaps to selection actions: same place, same height -->
    <div class="toolbar card" :class="{ selecting: selected.size }">
      <template v-if="selected.size">
        <button class="chip-btn" :title="allShownSelected ? 'Deselect the shown images' : 'Select every shown image'" @click="toggleAll">
          <i :class="allShownSelected ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-multiple-blank-outline'" /> {{ allShownSelected ? 'Deselect shown' : 'Select shown' }}
        </button>
        <b class="sel-count">{{ selected.size }} selected</b>
        <span class="spacer" />
        <button class="chip-btn danger" title="Delete the selected images with their captions and masks (asks first)" @click="removeMany([...selected])">
          <i class="mdi mdi-trash-can-outline" /> Delete
        </button>
        <button class="chip-btn" title="Clear the selection (Esc)" @click="selected.clear()"><i class="mdi mdi-close" /> Done</button>
      </template>
      <template v-else>
      <div class="seg" role="tablist" aria-label="View">
        <button :class="{ on: view === 'grid' }" title="Grid" @click="view = 'grid'"><i class="mdi mdi-view-grid-outline" /></button>
        <button :class="{ on: view === 'list' }" title="Table" @click="view = 'list'"><i class="mdi mdi-view-list-outline" /></button>
      </div>
      <template v-if="view === 'grid'">
        <label class="zoom" title="Card size">
          <i class="mdi mdi-image-size-select-small" />
          <ZenSlider v-model="zoom" :min="120" :max="420" :step="10" />
          <i class="mdi mdi-image-size-select-large" />
        </label>
        <button class="chip-btn" :title="fit === 'cover' ? 'Showing crops; click to show whole images' : 'Showing whole images; click to crop'" @click="fit = fit === 'cover' ? 'contain' : 'cover'">
          <i :class="fit === 'cover' ? 'mdi mdi-crop' : 'mdi mdi-fit-to-screen-outline'" /> {{ fit === 'cover' ? 'Crop' : 'Fit' }}
        </button>
        <button v-if="counts.masked" class="chip-btn" :class="{ on: showMasks }" title="Dim what each mask leaves out: the bright part trains at full weight" @click="showMasks = !showMasks">
          <i class="mdi mdi-layers-outline" /> Masks
        </button>
      </template>
      <div class="seg">
        <button :class="{ on: filter === 'all' }" @click="filter = 'all'">All</button>
        <button :class="{ on: filter === 'uncaptioned' }" @click="filter = 'uncaptioned'">Uncaptioned {{ counts.uncaptioned }}</button>
        <button :class="{ on: filter === 'masked' }" @click="filter = 'masked'">Masked {{ counts.masked }}</button>
        <button :class="{ on: filter === 'unmasked' }" @click="filter = 'unmasked'">Unmasked</button>
      </div>
      <div class="search"><ZenInput v-model="query" placeholder="Search captions or names" size="sm" /></div>
      <span class="spacer" />
      <button class="chip-btn" :class="{ on: capOpen }" @click="capOpen = !capOpen"><i class="mdi mdi-creation-outline" /> Auto-caption</button>
      <button class="chip-btn" :class="{ on: bulkOpen }" @click="bulkOpen = !bulkOpen"><i class="mdi mdi-text-box-edit-outline" /> Bulk</button>
      </template>
    </div>

    <div class="sets">
      <span class="eyebrow">Caption set</span>
      <button v-for="st in sets" :key="st" class="set-tab" :class="{ on: activeSet === st }" @click="activeSet = st">
        {{ st }}
        <span class="muted small">{{ items.filter((i) => i.captions?.[st]).length }}/{{ items.length }}</span>
      </button>
      <button class="set-tab add" title="Add a caption set" @click="addSet"><i class="mdi mdi-plus" /></button>
    </div>

    <CaptionPanel v-if="capOpen" :dataset="name" :items="items" :sets="sets" :selected="[...selected]" @changed="onCaptioned" @review="review" />

    <div v-if="reviewing" class="reviewing">
      <i class="mdi mdi-filter-outline" />
      <span class="small">Showing the {{ reviewing.paths.size }} image{{ reviewing.paths.size === 1 ? '' : 's' }} a caption job wrote into <b>“{{ reviewing.job.set_name }}”</b></span>
      <button class="chip-btn" @click="reviewing = null"><i class="mdi mdi-close" /> Show all</button>
    </div>

    <div v-if="bulkOpen" class="bulk card">
      <span class="muted small">On the “{{ activeSet }}” set:</span>
      <ZenInput v-model="findText" placeholder="find (for replace)" size="sm" />
      <ZenInput v-model="bulkText" placeholder="trigger word / text" size="sm" />
      <ZenButton size="sm" title="Put the text at the start of every caption" @click="bulk('prefix')">Prefix all</ZenButton>
      <ZenButton size="sm" title="Append the text to every caption" @click="bulk('suffix')">Suffix all</ZenButton>
      <ZenButton size="sm" :disabled="!findText" title="Replace 'find' with the text everywhere" @click="bulk('replace')">Replace</ZenButton>
    </div>


    <div v-if="notice" class="notice">{{ notice }}</div>

    <ZenEmpty v-if="!loading && !items.length" title="Empty dataset" icon="mdi mdi-image-plus-outline">
      Drop images, videos, same-name .txt captions and masks (<span class="mono">name-masklabel.png</span>) anywhere on this page.
    </ZenEmpty>

    <!-- grid -->
    <div v-else-if="view === 'grid'" class="grid" :style="{ '--card': `${zoom}px` }">
      <article
        v-for="(it, i) in shown"
        :key="it.path"
        class="card-item"
        :class="{ sel: selected.has(it.path) }"
      >
        <button class="thumb" :title="it.path" @click="lightbox = i">
          <img :src="thumb(it, thumbSize)" loading="lazy" :alt="cap(it)" :style="{ objectFit: fit }" />
          <img v-if="showMasks && it.mask" :src="thumb({ path: it.mask }, thumbSize)" loading="lazy" alt="" class="mask-ov" :style="{ objectFit: fit }" />
          <span class="badges">
            <span v-if="it.mask" class="badge" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'This mask is a different shape from its image: it won\'t line up' : `Masked: ${pct(it.mask_coverage)} trains at full weight`">
              <i :class="it.mask_mismatch ? 'mdi mdi-alert-outline' : 'mdi mdi-layers-outline'" /> {{ pct(it.mask_coverage) }}
            </span>
            <span v-if="it.kind === 'video'" class="badge" :title="`Video: ${it.frames ?? '?'} frames at ${it.fps ?? '?'} fps`"><i class="mdi mdi-play" /> {{ secs(it.duration) }}</span>
            <span v-if="it.repeats > 1" class="badge" :title="`Repeats ${it.repeats}×`">×{{ it.repeats }}</span>
            <span v-if="!cap(it)" class="badge warn" title="No caption"><i class="mdi mdi-text-box-remove-outline" /></span>
          </span>
        </button>
        <label class="check" :class="{ on: selected.has(it.path) }" @click.prevent="toggleSelect(it, i, $event)">
          <i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
        </label>
        <button class="card-body" :title="cap(it) || 'No caption'" @click="lightbox = i">
          <span class="cap-text" :class="{ empty: !cap(it) }">{{ cap(it) || 'No caption' }}</span>
          <span class="card-meta mono">{{ it.path.split('/').at(-1) }}<span v-if="it.width"> · {{ it.width }}×{{ it.height }}</span></span>
        </button>
      </article>
    </div>

    <!-- table -->
    <div v-else class="table card">
      <div class="tr th">
        <label class="c-check" @click.prevent="toggleAll"><i :class="allShownSelected ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></label>
        <span class="c-thumb" />
        <button class="c-name sortable" @click="sortBy('name')">Name <i v-if="sortKey === 'name'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
        <button class="c-res sortable" @click="sortBy('res')">Size <i v-if="sortKey === 'res'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
        <button class="c-bytes sortable" @click="sortBy('size')">File <i v-if="sortKey === 'size'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
        <span class="c-mask">Mask</span>
        <button class="c-cap sortable" @click="sortBy('caption')">Caption <i v-if="sortKey === 'caption'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
        <span class="c-act" />
      </div>
      <div v-for="(it, i) in shown" :key="it.path" class="tr" :class="{ sel: selected.has(it.path), dirty: drafts[k(it)] != null }">
        <label class="c-check" @click.prevent="toggleSelect(it, i, $event)"><i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></label>
        <button class="c-thumb" @click="lightbox = i"><img :src="thumb(it, 128)" loading="lazy" alt="" /></button>
        <span class="c-name mono ellipsis" :title="it.path">{{ it.path }}</span>
        <span class="c-res mono muted">{{ it.width ? `${it.width}×${it.height}` : '–' }}</span>
        <span class="c-bytes mono muted">{{ kb(it.bytes) }}</span>
        <span class="c-mask small" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'Different shape from its image: it won\'t line up' : ''">{{ it.mask ? pct(it.mask_coverage) : '' }}</span>
        <textarea
          v-autosize
          class="c-cap cap-edit"
          :class="{ empty: !cap(it) && drafts[k(it)] == null }"
          :value="drafts[k(it)] ?? cap(it)"
          placeholder="No caption: add one"
          rows="1"
          @input="drafts[k(it)] = ($event.target as HTMLTextAreaElement).value"
          @blur="save(it)"
          @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
        />
        <span class="c-act">
          <span v-if="saving[k(it)]" class="muted small">saving…</span>
        </span>
      </div>
    </div>

    <ZenLightbox v-if="lightbox != null && lbItems.length" v-model:index="lightbox" :items="lbItems" :searchable="false" @close="lightbox = null">
      <template #side="{ index }">
        <div v-if="shown[index]" class="side">
          <div class="eyebrow">Caption</div>
          <textarea
            v-autosize
            class="side-cap"
            :value="drafts[k(shown[index])] ?? cap(shown[index])"
            placeholder="No caption: add one"
            rows="3"
            @input="drafts[k(shown[index])] = ($event.target as HTMLTextAreaElement).value"
            @blur="save(shown[index])"
            @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
          />
          <div class="muted small">
            {{ saving[k(shown[index])] ? 'saving…' : drafts[k(shown[index])] != null ? 'unsaved: click away or Ctrl+Enter' : 'saved' }}
          </div>
          <div v-for="[st, text] in otherSets(shown[index])" :key="st" class="other-set">
            <div class="other-head">
              <span class="eyebrow">{{ st }}</span>
              <span class="spacer" />
              <button class="mini" :title="`Copy this into “${activeSet}”`" @click="useCaption(shown[index], text)">Use this</button>
            </div>
            <p>{{ text }}</p>
          </div>
          <div v-if="shown[index].mask" class="mask-info">
            <img :src="thumb({ path: shown[index].mask! }, 256)" alt="" />
            <div class="small">
              <b>Mask</b> · {{ pct(shown[index].mask_coverage) }} trains at full weight
              <div v-if="shown[index].mask_mismatch" class="warn">A different shape from the image: it's stretched over it and won't line up.</div>
              <div class="muted">M toggles it over the picture</div>
            </div>
          </div>
          <dl>
            <dt>File</dt><dd class="mono">{{ shown[index].path }}</dd>
            <dt>Size</dt><dd class="mono">{{ shown[index].width }}×{{ shown[index].height }} · {{ kb(shown[index].bytes) }}</dd>
            <template v-if="shown[index].kind === 'video'">
              <dt>Video</dt><dd class="mono">{{ secs(shown[index].duration) }} · {{ shown[index].fps }} fps · {{ shown[index].frames }} frames</dd>
            </template>
            <template v-if="!shown[index].mask && shown[index].kind !== 'video'"><dt>Mask</dt><dd>none: drop name-masklabel.png to add one</dd></template>
            <dt v-if="shown[index].repeats > 1">Repeats</dt><dd v-if="shown[index].repeats > 1">×{{ shown[index].repeats }}</dd>
          </dl>
        </div>
      </template>
    </ZenLightbox>

    <div v-if="dragging" class="dropzone"><i class="mdi mdi-tray-arrow-down" /> Drop to add to {{ name }}</div>
  </div>
</template>

<style scoped>
.wide {
  max-width: 1400px;
}
.warn {
  color: var(--zen-warn);
}
.err {
  border-color: var(--zen-danger);
  white-space: pre-wrap;
  font-size: 12px;
}
.small {
  font-size: 11px;
}
.spacer {
  flex: 1;
}
.toolbar.selecting {
  border-color: color-mix(in srgb, var(--zen-accent) 45%, transparent);
}
.sel-count {
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
.chip-btn.danger:hover {
  color: var(--zen-danger);
  border-color: var(--zen-danger);
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  min-height: 50px;
  box-sizing: border-box;
  padding: 8px 10px;
  position: sticky;
  top: 8px;
  z-index: 5;
  background: color-mix(in srgb, var(--zen-surface) 85%, transparent);
  backdrop-filter: blur(10px);
}
.seg {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button,
.chip-btn {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.seg button.on,
.chip-btn.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.chip-btn {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.chip-btn:hover,
.seg button:hover {
  color: var(--zen-text);
}
.zoom {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
}
.zoom .zen-slider {
  width: 110px;
}
.search {
  width: 220px;
}
.sets {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.set-tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.set-tab.on {
  border-color: var(--zen-accent);
  color: var(--zen-text);
  background: color-mix(in srgb, var(--zen-accent) 14%, transparent);
}
.set-tab.add {
  border-style: dashed;
}
.reviewing {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 40%, transparent);
  background: color-mix(in srgb, var(--zen-accent) 8%, transparent);
}
.reviewing > :nth-child(2) {
  flex: 1;
}
/* the mask over its picture: black (left out) darkens, white (trained) leaves it as is */
.thumb img.mask-ov {
  mix-blend-mode: multiply;
  opacity: 0.8;
  pointer-events: none;
}
.other-set {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.other-set p {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--zen-muted);
  max-height: 110px;
  overflow: auto;
  scrollbar-width: thin;
}
.other-head {
  display: flex;
  align-items: center;
}
.mini {
  height: 22px;
  padding: 0 8px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.mini:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.mask-info {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.mask-info img {
  width: 64px;
  height: 64px;
  flex: none;
  object-fit: contain;
  border-radius: var(--r-md);
  background: #000;
}
.bulk {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  padding: 10px 12px;
}
.bulk :deep(input) {
  width: 180px;
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  text-decoration: underline;
  cursor: pointer;
  font: inherit;
}
.notice {
  align-self: flex-start;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-ok) 18%, transparent);
}

/* grid: every card the same size; the slider sets it */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(var(--card), 1fr));
  gap: 12px;
  align-items: start;
}
.card-item {
  position: relative;
  display: flex;
  flex-direction: column;
  border-radius: var(--r-xl);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 60%, transparent);
  overflow: hidden;
  transition: border-color 0.15s ease;
}
.card-item:hover {
  border-color: var(--zen-muted);
}
.card-item.sel {
  border-color: var(--zen-accent);
  box-shadow: 0 0 0 1px var(--zen-accent);
}
/* every card the same size: a square thumbnail (image positioned inside, so its own
   aspect can't stretch the card) over a fixed-height caption block */
.thumb {
  position: relative;
  display: block;
  width: 100%;
  aspect-ratio: 1;
  padding: 0;
  border: none;
  background: var(--zen-surface-2);
  cursor: zoom-in;
  overflow: hidden;
}
.thumb img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.badges {
  position: absolute;
  right: 6px;
  bottom: 6px;
  display: flex;
  gap: 4px;
}
.badge {
  font-size: 11px;
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.6);
  color: #fff;
}
.badge.warn {
  color: #fbbf24;
}
.check {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: var(--r-md);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  font-size: 18px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s ease;
}
.card-item:hover .check,
.check.on,
.grid:has(.check.on) .check {
  opacity: 1;
}
.check.on {
  background: var(--zen-accent);
  color: var(--zen-accent-text);
}
.card-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  height: 74px;
  flex: none;
  padding: 9px 10px 8px;
  border: none;
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.cap-text {
  font-size: 12px;
  line-height: 1.4;
  height: 2.8em; /* exactly two lines; the clamp lives here, padding stays outside */
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.cap-text.empty {
  color: var(--zen-warn);
}
.card-meta {
  font-size: 10px;
  color: var(--zen-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* table */
.table {
  padding: 0;
  overflow: hidden;
}
.tr {
  display: grid;
  grid-template-columns: 32px 64px minmax(120px, 0.9fr) 84px 64px 44px minmax(280px, 4fr) 48px;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.tr.th {
  border-top: none;
  position: sticky;
  top: 0;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--zen-muted);
  background: color-mix(in srgb, var(--zen-surface) 90%, transparent);
}
.tr.sel {
  background: color-mix(in srgb, var(--zen-accent) 10%, transparent);
}
.tr.dirty .cap-edit {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
.sortable {
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  text-transform: inherit;
  letter-spacing: inherit;
  text-align: left;
  padding: 0;
  cursor: pointer;
}
.c-check {
  font-size: 18px;
  color: var(--zen-muted);
  cursor: pointer;
  display: grid;
  place-items: center;
}
.tr.sel .c-check {
  color: var(--zen-accent);
}
.c-thumb {
  width: 56px;
  height: 56px;
  padding: 0;
  border: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: zoom-in;
}
.c-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.c-name {
  font-size: 12px;
}
.c-res,
.c-bytes {
  font-size: 11px;
}
.c-mask {
  text-align: center;
  color: var(--zen-accent);
}
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cap-edit {
  display: block;
  width: 100%;
  resize: none;
  font: inherit;
  font-size: 13px;
  line-height: 1.5;
  max-height: calc(4 * 1.5em + 14px); /* 4 whole lines; focus opens it up */
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: color-mix(in srgb, var(--zen-text) 25%, transparent) transparent;
  padding: 6px 10px;
  border-radius: var(--r-md);
  border: 1px solid transparent;
  background: transparent;
  color: var(--zen-text);
}
.cap-edit.clamped:not(:focus) {
  mask-image: linear-gradient(to bottom, #000 calc(100% - 2.2em), transparent calc(100% - 4px));
}
.cap-edit:hover {
  border-color: color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.cap-edit:focus {
  max-height: calc(14 * 1.5em + 14px);
  outline: none;
  border-color: var(--zen-accent);
  background: var(--zen-field-bg, var(--zen-input));
}
.cap-edit.empty::placeholder {
  color: var(--zen-warn);
}
.icon {
  width: 30px;
  height: 30px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.icon:hover {
  color: var(--zen-danger);
  background: var(--zen-surface-2);
}

/* lightbox side panel */
.side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-start;
}
.side-cap {
  display: block;
  width: 100%;
  resize: none;
  font: inherit;
  font-size: 13px;
  line-height: 1.6;
  min-height: 5em;
  max-height: 55vh;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: color-mix(in srgb, var(--zen-text) 25%, transparent) transparent;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--zen-border);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
}
.side-cap:focus {
  outline: none;
  border-color: var(--zen-accent);
}
dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 12px;
  margin: 4px 0;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
  word-break: break-all;
}

.dropzone {
  position: fixed;
  inset: 16px;
  z-index: 60;
  display: grid;
  place-items: center;
  font-size: 18px;
  font-weight: 600;
  border: 2px dashed var(--zen-accent);
  border-radius: var(--r-xl);
  background: color-mix(in srgb, var(--zen-bg) 75%, transparent);
  backdrop-filter: blur(4px);
  pointer-events: none;
}
.dropzone i {
  font-size: 40px;
  color: var(--zen-accent);
}
@media (max-width: 900px) {
  .tr {
    grid-template-columns: 32px 56px 1fr 56px;
  }
  .c-res,
  .c-bytes,
  .c-mask {
    display: none;
  }
  .c-cap {
    grid-column: 2 / -1;
  }
}
</style>
