<script setup lang="ts">
// One dataset. Grid (uniform cards, zoomable) or table (sortable, inline captions);
// click any image for the lightbox with a caption editor beside it. Drop files anywhere
// to upload: images, same-name .txt captions, and masks (<name>-masklabel.png).
import { ZenButton, ZenCombo, ZenCompare, ZenContextMenu, ZenEmpty, ZenInput, ZenLightbox, ZenSlider, ZenSwitch, ZenTip } from '@nynxz/zenkit-ui'
import type { ContextMenuItem, LightboxItem } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { engine, thumbUrl, viewUrl, type CaptionJob, type DatasetFaces, type DatasetItem } from '@/api/engine'
import { reducedMotion } from '@/settings'
import { activeCaptions } from '@/captions'
import CaptionPanel from '@/components/CaptionPanel.vue'
import StickyBar from '@/components/StickyBar.vue'
import DatasetStats from '@/components/DatasetStats.vue'
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
const view = ref<'grid' | 'list' | 'pairs'>(saved.view ?? 'grid')
const zoom = ref<number>(saved.zoom ?? 200)
const fit = ref<'cover' | 'contain'>(saved.fit ?? 'cover')
const showMasks = ref<boolean>(saved.masks ?? false) // dim what each mask leaves out, on the grid
const sideOpen = ref<boolean>(saved.side ?? true) // the side panel: health, faces, auto-caption, bulk edit, display
// In a narrow window the panel would land under every picture, out of reach: there it's a drawer
// over the right edge instead, opened by the same button, closed on its own (not remembered)
const narrowQuery = window.matchMedia('(max-width: 1100px)')
const narrow = ref(narrowQuery.matches)
const drawerOpen = ref(false)
const onNarrow = (e: MediaQueryListEvent) => (narrow.value = e.matches)
narrowQuery.addEventListener('change', onNarrow)
onUnmounted(() => narrowQuery.removeEventListener('change', onNarrow))
const panelShown = computed(() => (narrow.value ? drawerOpen.value : sideOpen.value))
// The side panel stays pinned without a scrollbar of its own: under the sticky toolbar when it fits
// the window, or, when it's taller, scrolling with the page until its bottom reaches the window's
// bottom and staying there (a negative sticky top), so it never scrolls out of sight
const sideEl = ref<HTMLElement | null>(null)
const sideTop = ref(58)
const PIN = 66 // the sticky toolbar (10px down, 46px tall) + a 10px gap
function placeSide() {
  const h = sideEl.value?.offsetHeight ?? 0
  sideTop.value = Math.min(PIN, window.innerHeight - h - 12)
}
let sideObs: ResizeObserver | null = null
watch(sideEl, (el) => {
  sideObs?.disconnect()
  if (!el) return
  sideObs = new ResizeObserver(placeSide)
  sideObs.observe(el)
})
window.addEventListener('resize', placeSide)
onUnmounted(() => {
  sideObs?.disconnect()
  window.removeEventListener('resize', placeSide)
})

// the side panel changes the grid's columns: each card on screen takes its new place at once and
// grows or shrinks there from its old size, rather than snapping (no flying across the grid)
async function toggleSide() {
  if (narrow.value) {
    drawerOpen.value = !drawerOpen.value
    return
  }
  const reduce = reducedMotion()
  const vh = window.innerHeight
  const cards = reduce ? [] : [...document.querySelectorAll<HTMLElement>('.ds-main .card-item')]
  const before = new Map<HTMLElement, DOMRect>()
  for (const el of cards) {
    const r = el.getBoundingClientRect()
    if (r.bottom > -100 && r.top < vh + 100) before.set(el, r)
  }
  sideOpen.value = !sideOpen.value
  await nextTick()
  for (const [el, a] of before) {
    const b = el.getBoundingClientRect()
    if (!b.width || !b.height) continue
    const sx = a.width / b.width
    const sy = a.height / b.height
    if (Math.abs(sx - 1) < 0.01 && Math.abs(sy - 1) < 0.01) continue
    el.animate(
      [{ transform: `scale(${sx}, ${sy})` }, { transform: 'none' }],
      { duration: 240, easing: 'cubic-bezier(0.2, 0.7, 0.2, 1)' },
    )
  }
}
watch([view, zoom, fit, showMasks, sideOpen], () => {
  try {
    localStorage.setItem(PREFS, JSON.stringify({ view: view.value, zoom: zoom.value, fit: fit.value, masks: showMasks.value, side: sideOpen.value }))
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
// the toolbar's caption set and filter menus: every entry always there, so nothing shifts
const setItems = computed(() => [
  ...sets.value.map((st) => ({ value: st, label: `${st} · ${items.value.filter((i) => i.captions?.[st]).length}/${items.value.length}` })),
  { value: '__add__', label: '+ New caption set…' },
])
function pickSet(v: string) {
  if (v === '__add__') addSet()
  else activeSet.value = v
}
const filterItems = computed(() => [
  { value: 'all', label: `All · ${items.value.length}` },
  { value: 'uncaptioned', label: `Uncaptioned · ${counts.value.uncaptioned}` },
  { value: 'masked', label: `Masked · ${counts.value.masked}` },
  { value: 'unmasked', label: 'Unmasked' },
  { value: 'muted', label: `Muted in “${activeSet.value}” · ${counts.value.muted}` },
  { value: 'images', label: `Image issues · ${counts.value.imageIssues}` },
  { value: 'faces', label: `Face issues · ${faces.value?.items ? counts.value.faceIssues : '–'}` },
  ...(counts.value.pairs ? [{ value: 'pairs', label: `Edit pairs · ${counts.value.pairs}` }] : []),
])
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
    const facesOn = !!r.settings?.faces
    captionSubject.value = r.settings?.caption_subject ?? ''
    if (facesOn !== facesSetting.value) {
      facesSetting.value = facesOn
      if (facesOn && !faces.value) checkFaces() // cached per picture: quick after the first time
    }
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
type Filter = 'all' | 'uncaptioned' | 'masked' | 'unmasked' | 'pairs' | 'faces' | 'images' | 'muted'
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
  pairs: items.value.filter((i) => i.controls?.length).length,
  faceIssues: items.value.filter((i) => faceIssue(i)).length,
  imageIssues: items.value.filter((i) => imageFlags.value[i.path]?.length).length,
  muted: items.value.filter((i) => isMuted(i)).length,
}))

// ---- muting: kept, captions and all, but not trained on in the active caption set ----------------
const isMuted = (it: DatasetItem) => !!it.muted?.includes(activeSet.value)
async function setMuted(paths: string[], on: boolean) {
  if (!paths.length) return
  try {
    await engine.datasetMuted(props.name, activeSet.value, paths, on)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    return
  }
  const want = new Set(paths)
  for (const it of items.value) {
    if (!want.has(it.path)) continue
    const s = new Set(it.muted ?? [])
    if (on) s.add(activeSet.value)
    else s.delete(activeSet.value)
    it.muted = [...s]
  }
  statsPanel.value?.load()
}
// the stats panel's per-picture flags (dark, blurry, bars, small, near-duplicate...)
const imageFlags = ref<Record<string, string[]>>({})

// ---- faces (faces extra): a face per picture, how much it looks like the rest -----------------
const faces = ref<DatasetFaces | null>(null)
const facesBusy = ref(false)
const captionSubject = ref('') // this dataset's "caption_subject" setting: what auto-captions call the subject
const facesSetting = ref(false) // this dataset's "faces" setting: on = checked on open, in the stats and runs
async function toggleFaces() {
  const on = !facesSetting.value
  facesSetting.value = on
  try {
    await engine.datasetSettings(props.name, { faces: on })
  } catch {
    facesSetting.value = !on
    return
  }
  if (on) await checkFaces()
  else {
    faces.value = null
    if (filter.value === 'faces') filter.value = 'all'
    statsPanel.value?.load() // back to the balance score
  }
}
const statsPanel = ref<InstanceType<typeof DatasetStats> | null>(null)
async function checkFaces() {
  facesBusy.value = true
  try {
    faces.value = await engine.datasetFaces(props.name)
    statsPanel.value?.load() // the faces score joins the stats
  } catch (e) {
    faces.value = { available: false, hint: e instanceof Error ? e.message : String(e) }
  } finally {
    facesBusy.value = false
  }
}
const faceOf = (it: DatasetItem) => faces.value?.items?.[it.path]
/** why a picture's face needs a look: none found, more than one person, or not like the others */
function faceIssue(it: DatasetItem): string | null {
  const f = faceOf(it)
  if (!f) return null
  if (f.faces === 0) return f.masked ? `No face in the trained area (${f.masked} masked out)` : 'No face found'
  if (f.outlier) return `Doesn't look like the others (likeness ${f.likeness?.toFixed(2)})`
  if (f.faces > 1) return `${f.faces} faces: another person can bleed into the likeness. Mask the extra ones out`
  return null
}
/** a picture's face, described: its size, likeness, and any faces its mask leaves out */
const faceNote = (it: DatasetItem) => {
  const f = faceOf(it)!
  const out = f.masked ? ` · ${f.masked} masked out` : ''
  return `Face ${Math.round(f.share * 100)}% of the height; likeness to the others ${f.likeness?.toFixed(2)}${out}`
}
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
        (f === 'unmasked' && !it.mask) ||
        (f === 'pairs' && it.controls?.length) ||
        (f === 'faces' && faceIssue(it)) ||
        (f === 'images' && imageFlags.value[it.path]?.length) ||
        (f === 'muted' && isMuted(it))) &&
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
// a plain click opens the picture; Ctrl/Cmd+click toggles it in the selection, Shift+click a range
function onItemClick(it: DatasetItem, i: number, e: MouseEvent) {
  if (e.ctrlKey || e.metaKey || e.shiftKey) toggleSelect(it, i, e)
  else lightbox.value = i
}
// right-click: acts on the selection when the picture is in it, else on that picture alone
const menu = ref<InstanceType<typeof ZenContextMenu> | null>(null)
const menuItems = ref<ContextMenuItem[]>([])
function onItemMenu(it: DatasetItem, i: number, e: MouseEvent) {
  const many = selected.has(it.path) && selected.size > 1
  const paths = many ? [...selected] : [it.path]
  const targets = items.value.filter((x) => paths.includes(x.path))
  const anyMuted = targets.some(isMuted)
  const allMuted = targets.every(isMuted)
  const name = it.path.split('/').at(-1)!
  menuItems.value = [
    { heading: many ? `${paths.length} pictures` : name },
    ...(many ? [] : [{ label: 'Open', icon: 'mdi-open-in-app', run: () => (lightbox.value = i) }]),
    selected.has(it.path) && !many
      ? { label: 'Deselect', icon: 'mdi-checkbox-blank-outline', hint: 'Ctrl+click', run: () => selected.delete(it.path) }
      : many
        ? { label: 'Clear selection', icon: 'mdi-close', hint: 'Esc', run: () => selected.clear() }
        : { label: 'Select', icon: 'mdi-checkbox-marked-outline', hint: 'Ctrl+click', run: () => toggleSelect(it, i) },
    '-',
    ...(!allMuted ? [{ label: `Mute in “${activeSet.value}”`, icon: 'mdi-eye-off-outline', run: () => setMuted(paths, true) }] : []),
    ...(anyMuted ? [{ label: `Unmute in “${activeSet.value}”`, icon: 'mdi-eye-outline', run: () => setMuted(paths, false) }] : []),
    ...(many
      ? []
      : [
          { label: 'Copy caption', icon: 'mdi-content-copy', disabled: !cap(it), run: () => navigator.clipboard?.writeText(cap(it)) },
          ...(it.kind !== 'video'
            ? [{ label: 'Edit mask', icon: 'mdi-draw', run: () => router.push({ path: `/datasets/${encodeURIComponent(props.name)}/mask`, query: { path: it.path } }) }]
            : []),
        ]),
    '-',
    { label: many ? `Delete ${paths.length}…` : 'Delete…', icon: 'mdi-trash-can-outline', danger: true, run: () => removeMany(paths) },
  ]
  menu.value?.show(e)
}
const allShownSelected = computed(() => shown.value.length > 0 && shown.value.every((r) => selected.has(r.path)))
function onEsc(e: KeyboardEvent) {
  if (e.key === 'Escape' && narrow.value && drawerOpen.value && lightbox.value == null) {
    drawerOpen.value = false
    return
  }
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
// the captions as they were before the last bulk edit, so it can be undone
const bulkUndo = ref<{ set: string; before: Record<string, string> } | null>(null)
async function bulk(op: 'prefix' | 'suffix' | 'replace') {
  if (!bulkText.value && op !== 'replace') return
  const set = activeSet.value
  const before = Object.fromEntries(items.value.map((it) => [it.path, it.captions[set] ?? '']))
  const touched = Object.values(before).filter((c) => (op === 'replace' ? c.includes(findText.value) : true)).length
  const what = op === 'replace' ? `Replace “${findText.value}” with “${bulkText.value}”` : `${op === 'prefix' ? 'Put' : 'Append'} “${bulkText.value}” ${op === 'prefix' ? 'at the start of' : 'to'}`
  if (!touched || !confirm(`${what} ${op === 'replace' ? 'in' : ''} ${touched} caption${touched === 1 ? '' : 's'} in “${set}”? You can undo it from here.`)) return
  const { changed } = await engine.bulkCaptions(props.name, op, bulkText.value, findText.value, set)
  bulkUndo.value = changed ? { set, before } : null
  flash(`${changed} caption${changed === 1 ? '' : 's'} updated`)
  await load()
}
async function undoBulk() {
  const u = bulkUndo.value
  if (!u) return
  let n = 0
  for (const it of items.value) {
    const old = u.before[it.path]
    if (old != null && (it.captions[u.set] ?? '') !== old) {
      await engine.setCaption(props.name, it.path, old, u.set)
      n++
    }
  }
  bulkUndo.value = null
  flash(`${n} caption${n === 1 ? '' : 's'} put back`)
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

// ---- edit pairs -----------------------------------------------------------------------------
// the remembered view, except Pairs on a dataset without any (it shows the grid there)
const layout = computed(() => (view.value === 'pairs' && !counts.value.pairs ? 'grid' : view.value))
const isVideoPath = (p: string) => /\.(mp4|webm|mov|mkv)$/i.test(p)
// an input file sits beside its result: same folder, so /view takes the result's subfolder
const ctrlView = (it: DatasetItem, p: string) =>
  `${viewUrl({ filename: p.split('/').at(-1)!, subfolder: it.subfolder, type: 'input' })}&v=${bust.value}`
const aspect = (it: DatasetItem) => (it.width && it.height ? `${it.width} / ${it.height}` : '1')
const isSimplePair = (it: DatasetItem) => it.controls?.length === 1 && it.kind !== 'video' && !isVideoPath(it.controls[0]!)

// the lightbox's other caption sets: a few lines each until clicked open (no inner scrollbars)
const openSets = reactive(new Set<string>())
const toggleSet = (st: string) => (openSets.has(st) ? openSets.delete(st) : openSets.add(st))

// what holding Shift over an item shows: the file itself (a video plays), not its thumbnail
const full = (it: DatasetItem) => `${viewUrl({ ...it, type: 'input' })}&v=${bust.value}`

// ---- lightbox -----------------------------------------------------------------------------
const lightbox = ref<number | null>(null)
const lbItems = computed<LightboxItem[]>(() =>
  shown.value.map((it) => ({
    src: `${viewUrl({ ...it, type: 'input' })}&v=${bust.value}`,
    thumb: thumb(it, 256),
    label: it.path,
    meta: it.width ? `${it.width}×${it.height} · ${kb(it.bytes)}` : kb(it.bytes),
    overlay: it.mask ? thumb({ path: it.mask } as DatasetItem, 1024) : undefined,
    overlayMode: 'mask' as const, // the picture dims; what the mask trains shows at full strength
    kind: it.kind === 'video' ? ('video' as const) : undefined,
  })),
)

const thumb = (it: Pick<DatasetItem, 'path'>, size = 384) => `${thumbUrl(props.name, it.path, size)}&v=${bust.value}`
const secs = (d?: number | null) => (d == null ? '' : d < 60 ? `${d.toFixed(1)}s` : `${Math.floor(d / 60)}:${String(Math.round(d % 60)).padStart(2, '0')}`)
const pct = (v?: number) => (v == null ? '' : `${Math.round(v * 100)}%`)
// a card's file details, behind its ⓘ instead of a line under every caption
const fileInfo = (it: DatasetItem) =>
  [
    it.path,
    it.width ? `${it.width}×${it.height}` : '',
    it.bytes ? kb(it.bytes) : '',
    it.kind === 'video' ? `${it.frames ?? '?'} frames at ${it.fps ?? '?'} fps` : '',
  ]
    .filter(Boolean)
    .join(' · ')
const kb = (b: number) => (b > 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(1)} MB` : `${Math.round(b / 1024)} KB`)
const thumbSize = computed(() => (zoom.value > 260 ? 768 : 384))
</script>

<template>
  <div class="page full ds-page">
    <header class="ds-head">
      <RouterLink to="/datasets" class="crumb">Datasets</RouterLink>
      <h1>{{ name }}</h1>
      <span class="lede">
        {{ items.length - counts.videos }} image{{ items.length - counts.videos === 1 ? '' : 's' }}<template v-if="counts.videos"> · {{ counts.videos }} video{{ counts.videos === 1 ? '' : 's' }}</template>
        <template v-if="counts.uncaptioned"> · <span class="warn">{{ counts.uncaptioned }} uncaptioned</span></template>
        <template v-if="counts.mismatched"> · <span class="warn" title="A mask with a different shape from its image is stretched over it and won't line up">{{ counts.mismatched }} mask{{ counts.mismatched === 1 ? '' : 's' }} don't fit</span></template>
      </span>
      <span class="spacer" />
      <ZenButton size="sm" icon="mdi mdi-upload" :disabled="uploading" @click="fileInput?.click()">Add files</ZenButton>
      <ZenButton size="sm" variant="primary" icon="mdi mdi-play" @click="router.push({ path: '/runs/new', query: { dataset: name } })">Train on this</ZenButton>
      <input ref="fileInput" type="file" multiple accept="image/*,video/*,.txt" hidden @change="upload([...(($event.target as HTMLInputElement).files ?? [])])" />
    </header>

    <div v-if="error" class="card err mono">{{ error }}</div>

    <!-- one fixed-height row: nothing in it appears, disappears or wraps -->
    <StickyBar>
    <div class="ds-toolbar card">
      <div class="seg" role="tablist" aria-label="View">
        <button :class="{ on: layout === 'grid' }" title="Grid" @click="view = 'grid'"><i class="mdi mdi-view-grid-outline" /></button>
        <button :class="{ on: layout === 'list' }" title="Table" @click="view = 'list'"><i class="mdi mdi-view-list-outline" /></button>
        <button v-if="counts.pairs" :class="{ on: layout === 'pairs' }" title="Edit pairs: inputs → result, with the instruction" @click="view = 'pairs'"><i class="mdi mdi-swap-horizontal" /></button>
      </div>
      <label class="zoom" title="Card size" :class="{ off: layout === 'list' }">
        <i class="mdi mdi-image-size-select-small" />
        <ZenSlider v-model="zoom" :min="120" :max="420" :step="10" :disabled="layout === 'list'" />
      </label>
      <div class="tb-field" title="Caption set: the captions shown and edited, and which pictures are muted">
        <i class="mdi mdi-text-box-multiple-outline" />
        <ZenCombo :model-value="activeSet" :items="setItems" :sorted="false" :searchable="false" size="sm" @update:model-value="pickSet(String($event))" />
      </div>
      <div class="tb-field" title="Show only some of the pictures">
        <i class="mdi mdi-filter-variant" />
        <ZenCombo v-model="filter" :items="filterItems" :sorted="false" :searchable="false" size="sm" />
      </div>
      <div class="search"><ZenInput v-model="query" placeholder="Search captions or names" size="sm" /></div>
      <button class="icon-btn" :class="{ on: panelShown }" :title="panelShown ? 'Hide the side panel' : 'Show the side panel: health, faces, auto-caption, bulk edit, display'" @click="toggleSide">
        <i class="mdi mdi-dock-right" />
      </button>
    </div>
    </StickyBar>

    <div class="ds-body" :class="{ 'with-side': sideOpen && !narrow }">
      <main class="ds-main">
    <div v-if="reviewing" class="reviewing">
      <i class="mdi mdi-filter-outline" />
      <span class="small">Showing the {{ reviewing.paths.size }} image{{ reviewing.paths.size === 1 ? '' : 's' }} a caption job wrote into <b>“{{ reviewing.job.set_name }}”</b></span>
      <button class="chip-btn" @click="reviewing = null"><i class="mdi mdi-close" /> Show all</button>
    </div>

    <div v-if="notice" class="notice">{{ notice }}</div>

    <ZenEmpty v-if="!loading && !items.length" title="Empty dataset" icon="mdi mdi-image-plus-outline">
      Drop images, videos, same-name .txt captions and masks (<span class="mono">name-masklabel.png</span>) anywhere on this page.
    </ZenEmpty>

    <!-- grid -->
    <div v-else-if="layout === 'grid'" class="grid" :style="{ '--card': `${zoom}px` }">
      <article
        v-for="(it, i) in shown"
        :key="it.path"
        class="card-item"
        :class="{ sel: selected.has(it.path), muted: isMuted(it) }"
        @contextmenu="onItemMenu(it, i, $event)"
      >
        <button class="thumb" :title="it.path" :data-peek="full(it)" @click="onItemClick(it, i, $event)">
          <img :src="thumb(it, thumbSize)" loading="lazy" :alt="cap(it)" :class="{ 'mask-dim': showMasks && it.mask }" :style="{ objectFit: fit }" />
          <!-- the mask view: the picture dims, and only what the mask trains shows at full strength -->
          <img
            v-if="showMasks && it.mask"
            :src="thumb(it, thumbSize)"
            loading="lazy"
            alt=""
            class="mask-lit"
            :style="{ objectFit: fit, maskImage: `url(${thumb({ path: it.mask }, thumbSize)})`, WebkitMaskImage: `url(${thumb({ path: it.mask }, thumbSize)})`, maskSize: fit, WebkitMaskSize: fit }"
          />
          <!-- an edit pair: its input picture in the corner (the big one is the target) -->
          <img v-if="it.controls?.length" :src="thumb({ path: it.controls[0]! }, 256)" loading="lazy" alt="" class="control-inset" :title="`Input: ${it.controls.join(', ')}`" />
          <span class="badges">
            <span v-if="it.mask" class="badge" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'This mask is a different shape from its image: it won\'t line up' : `Masked: ${pct(it.mask_coverage)} trains at full weight`">
              <i :class="it.mask_mismatch ? 'mdi mdi-alert-outline' : 'mdi mdi-layers-outline'" /> {{ pct(it.mask_coverage) }}
            </span>
            <span v-if="it.kind === 'video'" class="badge" :title="`Video: ${it.frames ?? '?'} frames at ${it.fps ?? '?'} fps`"><i class="mdi mdi-play" /> {{ secs(it.duration) }}</span>
            <span v-if="it.repeats > 1" class="badge" :title="`Repeats ${it.repeats}×`">×{{ it.repeats }}</span>
            <span v-if="imageFlags[it.path]?.length" class="badge warn" :title="imageFlags[it.path]!.join(', ')"><i class="mdi mdi-image-filter-center-focus-weak" /> {{ imageFlags[it.path]!.length }}</span>
            <span v-if="faceOf(it)" class="badge" :class="{ warn: !!faceIssue(it) }" :title="faceIssue(it) ?? faceNote(it)">
              <i :class="faceOf(it)!.faces === 0 ? 'mdi mdi-face-man-outline' : 'mdi mdi-face-recognition'" />
              {{ faceOf(it)!.faces === 0 ? '0' : faceOf(it)!.likeness != null ? faceOf(it)!.likeness!.toFixed(2) : faceOf(it)!.faces }}
            </span>
            <span v-if="!cap(it)" class="badge warn" title="No caption"><i class="mdi mdi-text-box-remove-outline" /></span>
          </span>
        </button>
        <span class="info-btn" @click.stop><ZenTip :text="fileInfo(it)" label="File details" /></span>
        <button
          class="mute-btn"
          :class="{ on: isMuted(it) }"
          :title="isMuted(it) ? `Muted in “${activeSet}”: click to train on it again` : `Mute in “${activeSet}”: kept, but not trained on`"
          @click.stop="setMuted([it.path], !isMuted(it))"
        >
          <i :class="isMuted(it) ? 'mdi mdi-eye-off' : 'mdi mdi-eye-outline'" />
        </button>
        <label class="check" :class="{ on: selected.has(it.path) }" @click.prevent="toggleSelect(it, i, $event)">
          <i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
        </label>
        <button class="card-body" :title="cap(it) || 'No caption'" @click="onItemClick(it, i, $event)">
          <span class="cap-text" :class="{ empty: !cap(it) }">{{ cap(it) || 'No caption' }}</span>
        </button>
      </article>
    </div>

    <!-- edit pairs: every input, then the result, the instruction beside them -->
    <div v-else-if="layout === 'pairs'" class="pairs" :style="{ '--ph': `${zoom}px` }">
      <article v-for="(it, i) in shown" :key="it.path" class="pair card" :class="{ sel: selected.has(it.path), dirty: drafts[k(it)] != null }" @contextmenu="onItemMenu(it, i, $event)">
        <label class="check" :class="{ on: selected.has(it.path) }" @click.prevent="toggleSelect(it, i, $event)">
          <i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
        </label>
        <div class="pair-media">
          <!-- one picture in, one out, same shape: drag across to compare -->
          <div v-if="isSimplePair(it)" class="pane compare" :style="{ aspectRatio: aspect(it) }">
            <ZenCompare :before="thumb({ path: it.controls![0]! }, 768)" :after="thumb(it, 768)" before-label="Input" after-label="Result" track-closest=".pair" />
          </div>
          <template v-else>
            <button v-for="(c, n) in it.controls ?? []" :key="c" class="pane" :title="c" :data-peek="ctrlView(it, c)" @click="onItemClick(it, i, $event)">
              <video v-if="isVideoPath(c)" :src="ctrlView(it, c)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
              <img v-else :src="thumb({ path: c }, 512)" loading="lazy" alt="" />
              <span class="pane-tag">{{ (it.controls?.length ?? 0) > 1 ? `Input ${n + 1}` : 'Input' }}</span>
            </button>
            <i v-if="it.controls?.length" class="mdi mdi-arrow-right pair-arrow" />
            <button class="pane result" :title="it.path" :data-peek="full(it)" @click="onItemClick(it, i, $event)">
              <video v-if="it.kind === 'video'" :src="ctrlView(it, it.path)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
              <img v-else :src="thumb(it, 512)" loading="lazy" alt="" />
              <span class="pane-tag">{{ it.controls?.length ? 'Result' : 'No input' }}</span>
            </button>
          </template>
        </div>
        <div class="pair-text">
          <textarea
            v-autosize
            class="cap-edit"
            :class="{ empty: !cap(it) && drafts[k(it)] == null }"
            :value="drafts[k(it)] ?? cap(it)"
            placeholder="The instruction: what to change"
            rows="2"
            @input="drafts[k(it)] = ($event.target as HTMLTextAreaElement).value"
            @blur="save(it)"
            @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
          />
          <span class="card-meta mono">
            {{ it.path.split('/').at(-1) }}<span v-if="it.width"> · {{ it.width }}×{{ it.height }}</span>
            <span v-if="(it.controls?.length ?? 0) > 1"> · {{ it.controls!.length }} inputs</span>
            <span v-if="saving[k(it)]"> · saving…</span>
          </span>
        </div>
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
        <span class="c-face">Face</span>
        <button class="c-cap sortable" @click="sortBy('caption')">Caption <i v-if="sortKey === 'caption'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
        <span class="c-act" />
      </div>
      <div v-for="(it, i) in shown" :key="it.path" class="tr" :class="{ sel: selected.has(it.path), dirty: drafts[k(it)] != null, muted: isMuted(it) }" @contextmenu="onItemMenu(it, i, $event)">
        <label class="c-check" @click.prevent="toggleSelect(it, i, $event)"><i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></label>
        <button class="c-thumb" :data-peek="full(it)" @click="onItemClick(it, i, $event)"><img :src="thumb(it, 128)" loading="lazy" alt="" /></button>
        <span class="c-name mono ellipsis" :title="it.path">{{ it.path }}</span>
        <span class="c-res mono muted">{{ it.width ? `${it.width}×${it.height}` : '–' }}</span>
        <span class="c-bytes mono muted">{{ kb(it.bytes) }}</span>
        <span class="c-mask small" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'Different shape from its image: it won\'t line up' : ''">{{ it.mask ? pct(it.mask_coverage) : '' }}</span>
        <span
          class="c-face small mono"
          :class="{ warn: !!faceIssue(it) }"
          :title="faceOf(it) ? faceIssue(it) ?? faceNote(it) : 'Run Faces to check'"
        >
          <template v-if="faceOf(it)">
            <i v-if="faceIssue(it)" class="mdi mdi-alert-outline" />
            {{ faceOf(it)!.faces === 0 ? 'none' : faceOf(it)!.faces > 1 ? `${faceOf(it)!.faces} faces` : faceOf(it)!.likeness?.toFixed(2) }}
          </template>
        </span>
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

      </main>

      <div v-if="narrow && drawerOpen" class="drawer-scrim" @click="drawerOpen = false" />
      <aside v-if="panelShown" ref="sideEl" class="ds-side" :class="{ drawer: narrow }" :style="narrow ? undefined : { top: `${sideTop}px` }">
        <section class="side-sec">
          <header class="side-head"><span>Health</span><span class="muted small">“{{ activeSet }}”</span></header>
          <DatasetStats
            v-if="items.length"
            :key="`${name}:${activeSet}`"
            ref="statsPanel"
            :name="name"
            :set="activeSet"
            compact
            @loaded="(s) => (imageFlags = s.current.flags ?? {})"
          />
        </section>
        <section class="side-sec">
          <header class="side-head">
            <span>Faces</span>
            <ZenSwitch :model-value="facesSetting" :disabled="facesBusy" @update:model-value="toggleFaces" />
          </header>
          <p class="side-note">
            <template v-if="faces && !faces.available">{{ faces.hint }}</template>
            <template v-else-if="facesBusy">Checking faces…</template>
            <template v-else-if="facesSetting && faces?.items">
              {{ faces.with_face }}/{{ faces.total }} with a face{{ faces.one_identity === false ? ' · mixed people' : ` · alike ${faces.identity?.toFixed(2) ?? '–'}` }}{{ counts.faceIssues ? ` · ${counts.faceIssues} to look at` : '' }}
            </template>
            <template v-else>Off. On: faces checked on open, a Faces score, and likeness for this dataset's runs.</template>
          </p>
        </section>
        <section class="side-sec">
          <header class="side-head clickable" @click="capOpen = !capOpen">
            <span>Auto-caption</span><i :class="capOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
          </header>
          <CaptionPanel v-if="capOpen" :dataset="name" :items="items" :sets="sets" :selected="[...selected]" :subject="captionSubject" class="side-caption" @changed="onCaptioned" @review="review" />
        </section>
        <section class="side-sec">
          <header class="side-head clickable" @click="bulkOpen = !bulkOpen">
            <span>Bulk edit</span><i :class="bulkOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
          </header>
          <div v-if="bulkOpen" class="side-bulk">
            <span class="muted small">Every caption in “{{ activeSet }}”:</span>
            <ZenInput v-model="bulkText" placeholder="trigger word / text" size="sm" />
            <div class="side-row">
              <ZenButton size="sm" title="Put the text at the start of every caption" @click="bulk('prefix')">Prefix</ZenButton>
              <ZenButton size="sm" title="Append the text to every caption" @click="bulk('suffix')">Suffix</ZenButton>
            </div>
            <ZenInput v-model="findText" placeholder="find (to replace with the text)" size="sm" />
            <ZenButton size="sm" :disabled="!findText" title="Replace 'find' with the text everywhere" @click="bulk('replace')">Replace</ZenButton>
            <ZenButton v-if="bulkUndo" size="sm" icon="mdi mdi-undo" :title="`Put the captions in “${bulkUndo.set}” back as they were before the last bulk edit`" @click="undoBulk">Undo last edit</ZenButton>
          </div>
        </section>
        <section class="side-sec">
          <header class="side-head"><span>Display</span></header>
          <div class="side-row">
            <span class="small">Crop to the card</span><span class="spacer" />
            <ZenSwitch :model-value="fit === 'cover'" @update:model-value="fit = $event ? 'cover' : 'contain'" />
          </div>
          <div class="side-row" :title="counts.masked ? 'Shows what each mask trains: the rest dims' : 'No masks in this dataset'">
            <span class="small">Show masks</span><span class="spacer" />
            <ZenSwitch :model-value="showMasks" :disabled="!counts.masked" @update:model-value="showMasks = $event" />
          </div>
        </section>
      </aside>
    </div>

    <!-- selecting doesn't move anything: the actions float over the bottom of the page -->
    <Transition name="selbar">
      <div v-if="selected.size" class="sel-bar">
        <b>{{ selected.size }} selected</b>
        <button class="chip-btn" :title="allShownSelected ? 'Deselect the shown images' : 'Select every shown image'" @click="toggleAll">
          <i :class="allShownSelected ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-multiple-blank-outline'" /> {{ allShownSelected ? 'Deselect shown' : 'Select shown' }}
        </button>
        <span class="sel-sep" />
        <button class="chip-btn" :title="`Keep them, but don't train on them in “${activeSet}”`" @click="setMuted([...selected], true)"><i class="mdi mdi-eye-off-outline" /> Mute</button>
        <button class="chip-btn" :title="`Train on them again in “${activeSet}”`" @click="setMuted([...selected], false)"><i class="mdi mdi-eye-outline" /> Unmute</button>
        <button class="chip-btn danger" title="Delete the selected images with their captions and masks (asks first)" @click="removeMany([...selected])"><i class="mdi mdi-trash-can-outline" /> Delete</button>
        <span class="sel-sep" />
        <button class="chip-btn" title="Clear the selection (Esc)" @click="selected.clear()"><i class="mdi mdi-close" /></button>
      </div>
    </Transition>

    <ZenContextMenu ref="menu" :items="menuItems" />

    <ZenLightbox v-if="lightbox != null && lbItems.length" v-model:index="lightbox" :items="lbItems" :searchable="false" @close="lightbox = null">
      <template #side="{ index }">
        <div v-if="shown[index]" class="side">
          <template v-if="shown[index].controls?.length">
            <div class="eyebrow">{{ shown[index].controls!.length > 1 ? 'Inputs' : 'Input' }}</div>
            <div class="side-inputs">
              <template v-for="c in shown[index].controls" :key="c">
                <video v-if="isVideoPath(c)" :src="ctrlView(shown[index], c)" muted loop autoplay playsinline :title="c" />
                <img v-else :src="thumb({ path: c }, 384)" alt="" :title="c" />
              </template>
            </div>
          </template>
          <div class="eyebrow">{{ shown[index].controls?.length ? 'Instruction' : 'Caption' }}</div>
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
            <p :class="{ open: openSets.has(st) }" :title="openSets.has(st) ? '' : 'Show all'" @click="toggleSet(st)">{{ text }}</p>
          </div>
          <div v-if="shown[index].mask" class="mask-info">
            <img :src="thumb({ path: shown[index].mask! }, 256)" alt="" />
            <div class="small">
              <b>Mask</b> · {{ pct(shown[index].mask_coverage) }} trains at full weight
              <div v-if="shown[index].mask_mismatch" class="warn">A different shape from the image: it's stretched over it and won't line up.</div>
              <div class="muted">M toggles it over the picture</div>
            </div>
          </div>
          <RouterLink v-if="shown[index].kind !== 'video'" :to="{ path: `/datasets/${encodeURIComponent(name)}/mask`, query: { path: shown[index].path } }" class="edit-mask">
            <i class="mdi mdi-vector-selection" /> {{ shown[index].mask ? 'Edit mask' : 'Make a mask' }} <span class="muted">· SAM3, brush, feather</span>
          </RouterLink>
          <dl>
            <dt>File</dt><dd class="mono">{{ shown[index].path }}</dd>
            <dt>Size</dt><dd class="mono">{{ shown[index].width }}×{{ shown[index].height }} · {{ kb(shown[index].bytes) }}</dd>
            <template v-if="shown[index].kind === 'video'">
              <dt>Video</dt><dd class="mono">{{ secs(shown[index].duration) }} · {{ shown[index].fps }} fps · {{ shown[index].frames }} frames</dd>
            </template>
            <dt v-if="shown[index].repeats > 1">Repeats</dt><dd v-if="shown[index].repeats > 1">×{{ shown[index].repeats }}</dd>
          </dl>
        </div>
      </template>
    </ZenLightbox>

    <div v-if="dragging" class="dropzone"><i class="mdi mdi-tray-arrow-down" /> Drop to add to {{ name }}</div>
  </div>
</template>

<style scoped>
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
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
  cursor: pointer;
}
.other-set p.open {
  display: block;
  cursor: default;
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
.edit-mask {
  font-size: 12px;
  padding: 6px 10px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.edit-mask:hover {
  border-color: var(--zen-accent);
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

/* grid: the slider sets the smallest card; columns stretch to fill the row evenly */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(var(--card), 1fr));
  gap: 12px;
  align-items: start;
}
@keyframes side-in {
  from {
    opacity: 0;
    transform: translateX(16px);
  }
}
.card-item {
  transform-origin: top left;
  position: relative;
  display: flex;
  flex-direction: column;
  border-radius: var(--r-xl);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 60%, transparent);
  overflow: hidden;
  transition:
    border-color 0.15s ease,
    box-shadow 0.15s ease;
}
/* hover in the primary colour, softer than a selection's full accent ring */
.card-item:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
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
.pairs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(max(260px, calc(var(--ph) * 1.25)), 1fr));
  gap: 12px;
  align-items: start;
}
.pair {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
}
.pair.sel {
  border-color: var(--zen-accent);
}
.pair .check {
  top: 16px;
  left: 16px;
  z-index: 2;
}
.pair-media {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: var(--ph);
  overflow-x: auto;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.pane {
  position: relative;
  height: 100%;
  flex: none;
  padding: 0;
  border: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
}
.pane img,
.pane video {
  height: 100%;
  width: auto;
  max-width: 100%;
  object-fit: contain;
  display: block;
}
.pane.compare {
  max-width: 100%;
}
.pane.compare {
  cursor: ew-resize;
}
.pane.compare :deep(.zen-compare) {
  width: 100%;
  height: 100%;
}
.pane-tag {
  position: absolute;
  left: 6px;
  bottom: 6px;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  pointer-events: none;
}
.pair-arrow {
  color: var(--zen-muted);
  font-size: 18px;
  flex: none;
}
.pair-text {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  padding: 0 2px;
}
.side-inputs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
  gap: 6px;
  margin-bottom: 6px;
}
.side-inputs img,
.side-inputs video {
  width: 100%;
  border-radius: var(--r-sm);
  display: block;
}
.tr.muted .c-thumb img {
  opacity: 0.35;
}
.card-item.muted .thumb img:not(.control-inset):not(.mask-ov) {
  opacity: 0.35;
  filter: grayscale(0.6);
}
.info-btn {
  position: absolute;
  top: 8px;
  right: 40px;
  z-index: 2;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border-radius: var(--r-sm);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.info-btn :deep(*) {
  color: inherit;
}
.card-item:hover .info-btn,
.info-btn:focus-within {
  opacity: 1;
}
.mute-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 2;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.card-item:hover .mute-btn,
.mute-btn.on,
.mute-btn:focus-visible {
  opacity: 1;
}
.mute-btn.on {
  color: var(--zen-warn);
}
.stats-panel {
  margin-bottom: 12px;
}
.thumb .control-inset {
  position: absolute;
  left: 6px;
  top: 6px;
  width: 34%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-sm);
  border: 2px solid var(--zen-surface);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
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
  height: 52px;
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
  grid-template-columns: 32px 64px minmax(120px, 0.9fr) 84px 64px 44px 70px minmax(280px, 4fr) 48px;
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
/* longer than 4 lines: the end fades out, no scrollbar until it's clicked into (focus opens it up) */
.cap-edit.clamped:not(:focus) {
  overflow-y: hidden;
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

/* ---- page layout: a slim header, one fixed toolbar, the pictures and a side panel ---- */
.ds-page {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ds-head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
}
.ds-head .crumb {
  color: var(--zen-muted);
  font-size: 14px;
}
.crumb-sep {
  color: var(--zen-muted);
  font-size: 14px;
}
.ds-head h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}
.ds-head .lede {
  margin: 0 0 0 6px;
  font-size: 13px;
  color: var(--zen-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ds-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 46px;
  padding: 0 8px;
  flex-wrap: nowrap;
  overflow: hidden;
}
/* what sticks: the toolbar on a strip of the page's background from 10px above it down to its
   middle: nothing peeks over the top, and the strip's edge hides behind the toolbar, so below it
   pictures just pass under the bar (no line anywhere). The negative margin cancels the padding,
   so at rest the page lays out as if the strip weren't there */
/* everything gives way before anything is cut off: the side panel's button is always there */
.ds-toolbar .zoom {
  flex: 0 1 150px;
  min-width: 70px;
}
.ds-toolbar .zoom.off {
  opacity: 0.35;
}
.tb-field {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 0 1 190px;
  min-width: 110px;
  color: var(--zen-muted);
}
.tb-field :deep(.zen-combo) {
  flex: 1;
  min-width: 0;
}
.ds-toolbar .search {
  flex: 1 1 120px;
  min-width: 70px;
}
@media (max-width: 900px) {
  .ds-toolbar .zoom,
  .tb-field > .mdi {
    display: none;
  }
}
.icon-btn {
  flex: none;
  width: 32px;
  height: 30px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  font-size: 18px;
  cursor: pointer;
}
.icon-btn:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.icon-btn.on {
  color: var(--zen-accent);
}
.ds-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 14px;
  align-items: start;
}
.ds-body.with-side {
  grid-template-columns: minmax(0, 1fr) 320px;
}
.ds-side.drawer {
  position: fixed;
  top: 12px;
  right: 12px;
  bottom: 12px;
  z-index: 40;
  width: min(340px, calc(100vw - 24px));
  overflow-y: auto;
  scrollbar-width: thin;
  overscroll-behavior: contain;
  filter: drop-shadow(0 18px 40px rgba(0, 0, 0, 0.55));
}
.drawer-scrim {
  position: fixed;
  inset: 0;
  z-index: 39;
  background: rgba(0, 0, 0, 0.4);
}
.ds-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ds-side {
  /* pinned (top set by placeSide), never a scrollbar of its own */
  position: sticky;
  animation: side-in 0.24s cubic-bezier(0.2, 0.7, 0.2, 1);
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.side-sec {
  padding: 10px 12px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-lg, 12px);
  background: var(--zen-surface);
}
.side-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
}
.side-head.clickable {
  cursor: pointer;
}
.side-head.clickable i {
  color: var(--zen-muted);
}
.side-note {
  margin: 6px 0 0;
  font-size: 12px;
  line-height: 1.5;
  color: var(--zen-muted);
}
.side-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.side-bulk {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
}
.side-caption {
  margin-top: 8px;
  border: none !important;
  padding: 0 !important;
  background: none !important;
}
/* the selection's actions float over the page bottom: selecting moves nothing */
.sel-bar {
  position: fixed;
  left: 50%;
  bottom: 18px;
  transform: translateX(-50%);
  z-index: 50;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--zen-border);
  background: var(--zen-surface);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  font-size: 13px;
}
.sel-sep {
  width: 1px;
  height: 18px;
  background: var(--zen-border);
}
.selbar-enter-active,
.selbar-leave-active {
  transition: opacity var(--zen-dur-fast), transform var(--zen-dur-fast);
}
.selbar-enter-from,
.selbar-leave-to {
  opacity: 0;
  transform: translate(-50%, 8px);
}
/* the mask view */
.thumb img.mask-dim {
  filter: grayscale(0.85) brightness(0.32);
}
.thumb img.mask-lit {
  mask-mode: luminance;
  mask-repeat: no-repeat;
  mask-position: center;
  pointer-events: none;
}
</style>
