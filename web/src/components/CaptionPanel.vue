<script setup lang="ts">
// Auto-captioning for one dataset, as a job: pick the model, the style, which caption set it
// writes and which images; try it on a few first; watch it run, cancel it, and afterwards
// review what it wrote or undo it. It never rewrites captions in "default" without asking.
import { ZenButton, ZenCombo, ZenInput, ZenNumber } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { setSetting, settings } from '@/settings'
import { EVENTS, EngineError, engine, thumbUrl, type CaptionJob, type CaptionLora, type CaptionMode, type CaptionStyle, type DatasetItem, type ModelSenses } from '@/api/engine'

const props = defineProps<{ dataset: string; items: DatasetItem[]; sets: string[]; selected: string[] }>()
const emit = defineEmits<{ changed: [set: string]; review: [job: CaptionJob] }>()

const styles = ref<CaptionStyle[]>([])
const models = ref<string[]>([])
const hearing = ref<string[]>([]) // models that also hear a video's soundtrack (Gemma 4)
const hasVideo = computed(() => props.items.some((it) => it.kind === 'video'))
const senses = ref<Record<string, ModelSenses>>({})
// each model's senses as icons, coloured for this dataset: fine, short of what it needs (warn), or not needed (dim)
function senseIcons(name: string) {
  // an older engine sends only which models hear: those take whole clips, the rest stills
  const s = senses.value[name] ?? (models.value.includes(name)
    ? { image: true, video: hearing.value.includes(name) ? 'clip' : 'stills', audio: hearing.value.includes(name), frames: 4 }
    : null)
  if (!s) return []
  const vid = hasVideo.value
  return [
    { icon: 'mdi mdi-image-outline', tone: vid ? 'dim' : 'ok', tip: 'Sees pictures' },
    {
      icon: 'mdi mdi-movie-open-outline',
      tone: !vid ? 'dim' : s.video === 'clip' ? 'ok' : 'warn',
      tip: s.video === 'clip' ? 'Watches a video as a timestamped clip' : `Sees a video as ${s.frames} stills: motion and timing are guesses`,
    },
    {
      icon: s.audio ? 'mdi mdi-music-note' : 'mdi mdi-music-note-off',
      tone: !vid ? 'dim' : s.audio ? 'ok' : 'warn',
      tip: s.audio ? "Hears a video's soundtrack" : "Can't hear: captions won't describe the sound",
    },
  ]
}
// LoRAs on the captioning model only (abliterated, uncensored, prompt styles): never training
const teLoras = ref<string[]>([])
const loras = ref<{ name: string; strength: number }[]>([])
const loraLink = ref('')
const loraChoices = ref<{ file: string; size: number | null }[]>([])
const loraNote = ref('')
try {
  loras.value = JSON.parse(localStorage.getItem("zentrainer.caption.loras") ?? '[]')
} catch {
  // storage blocked
}
watch(
  loras,
  (v) => {
    try {
      localStorage.setItem("zentrainer.caption.loras", JSON.stringify(v))
    } catch {
      // storage blocked
    }
  },
  { deep: true },
)
// the catalog's LoRAs that fit the selected model
const catalog = ref<CaptionLora[]>([])
const suggested = computed(() => catalog.value.filter((e) => new RegExp(e.base, 'i').test(form.clip_name)))
const sizeOf = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(n / 1e6))} MB`)
const isOn = (e: CaptionLora) => loras.value.some((l) => l.name.split('/').at(-1) === e.file.split('/').at(-1))
async function useSuggested(e: CaptionLora) {
  if (isOn(e)) {
    loras.value = loras.value.filter((l) => l.name.split('/').at(-1) !== e.file.split('/').at(-1))
    return
  }
  if (e.installed) {
    loras.value.push({ name: e.installed, strength: 1 })
    return
  }
  loraNote.value = ''
  try {
    const r = await engine.addCaptionLora('', undefined, e.id)
    if (r.name) {
      if (!teLoras.value.includes(r.name)) teLoras.value = [...teLoras.value, r.name]
      loras.value.push({ name: r.name, strength: 1 })
      loraNote.value = `downloading ${e.label} (${sizeOf(e.size)}): ready when the download finishes`
    }
  } catch (err) {
    loraNote.value = err instanceof Error ? err.message : String(err)
  }
}
const addLora = () => loras.value.push({ name: teLoras.value.find((n) => !loras.value.some((l) => l.name === n)) ?? '', strength: 1 })
async function fetchLora(file?: string) {
  loraNote.value = ''
  try {
    const r = await engine.addCaptionLora(loraLink.value, file)
    if (r.choices) {
      loraChoices.value = r.choices
      return
    }
    loraChoices.value = []
    loraLink.value = ''
    if (r.name) {
      if (!teLoras.value.includes(r.name)) teLoras.value = [...teLoras.value, r.name]
      loras.value.push({ name: r.name, strength: 1 })
      loraNote.value = `downloading ${r.name.split('/').at(-1)}: it's ready when the download finishes`
    }
  } catch (e) {
    loraNote.value = e instanceof Error ? e.message : String(e)
  }
}
const jobs = ref<CaptionJob[]>([])
const error = ref<string | null>(null)
const PREFS = 'zentrainer.caption'
const form = reactive({ clip_name: '', style: 'nl', prompt: '', set: 'nl', mode: 'missing' as CaptionMode, subject: '' })
try {
  Object.assign(form, JSON.parse(localStorage.getItem(PREFS) ?? '{}'), { mode: 'missing' })
} catch {
  // storage blocked
}
watch(form, () => {
  try {
    const { mode: _m, ...keep } = form
    localStorage.setItem(PREFS, JSON.stringify(keep))
  } catch {
    // storage blocked
  }
})
watch(
  () => form.style,
  (st, old) => {
    // the set follows a built-in style while it was still following it
    if (st !== 'custom' && !st.startsWith('u:') && (form.set === old || !form.set)) form.set = st
  },
)

onMounted(async () => {
  const info = await engine.captionInfo(props.dataset)
  styles.value = info.styles
  models.value = info.models // loadable only, vision-language first
  hearing.value = info.hearing ?? []
  senses.value = info.senses ?? {}
  teLoras.value = info.loras ?? []
  catalog.value = info.lora_catalog ?? []
  jobs.value = info.jobs.filter((j) => !j.preview)
  preview.value = info.jobs.find((j) => j.preview) ?? null // a try-on still running, or just done
  if (!models.value.includes(form.clip_name))
    form.clip_name = models.value.find((n) => /qwen3vl_4b/i.test(n)) ?? models.value.find((n) => /qwen3vl/i.test(n)) ?? models.value[0] ?? ''
})

// ---- what a run would do -------------------------------------------------------------------
const has = (it: DatasetItem) => !!it.captions?.[form.set]
const targets = computed(() => {
  if (form.mode === 'selected') {
    const want = new Set(props.selected)
    return props.items.filter((it) => want.has(it.path))
  }
  return form.mode === 'all' ? props.items : props.items.filter((it) => !has(it))
})
const replaces = computed(() => targets.value.filter(has).length)
const counts = computed(() => ({
  missing: props.items.filter((it) => !has(it)).length,
  all: props.items.length,
  selected: props.selected.length,
}))
watch(
  () => props.selected.length,
  (n) => {
    if (n && form.mode !== 'selected') form.mode = 'selected'
    else if (!n && form.mode === 'selected') form.mode = 'missing'
  },
)
const summary = computed(() => {
  const n = targets.value.length
  if (!n) return form.mode === 'missing' ? `Every image already has a “${form.set}” caption` : 'No images to caption'
  const fresh = n - replaces.value
  const parts = []
  if (fresh) parts.push(`writes ${fresh} new`)
  if (replaces.value) parts.push(`replaces ${replaces.value} existing`)
  return `${n} image${n === 1 ? '' : 's'}: ${parts.join(', ')} in “${form.set}”`
})
// ---- your saved styles: kept in your ComfyUI user settings -----------------------------------
const mine = computed(() => settings.captionStyles)
const own = computed(() => mine.value.find((x) => `u:${x.id}` === form.style) ?? null)
function pick(id: string) {
  form.style = id
  const m = mine.value.find((x) => `u:${x.id}` === id)
  if (m) form.prompt = m.prompt
}
const editable = computed(() => form.style === 'custom' || !!own.value)
const dirty = computed(() => !!own.value && own.value.prompt !== form.prompt)
function saveStyle() {
  if (own.value) {
    setSetting('captionStyles', mine.value.map((x) => (x.id === own.value!.id ? { ...x, prompt: form.prompt } : x)))
    return
  }
  const label = prompt('Name this caption style:')?.trim()
  if (!label) return
  const id = `${label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'style'}-${Date.now().toString(36).slice(-4)}`
  setSetting('captionStyles', [...mine.value, { id, label, prompt: form.prompt }])
  form.style = `u:${id}`
}
function removeStyle() {
  if (!own.value || !confirm(`Delete the caption style “${own.value.label}”?`)) return
  setSetting('captionStyles', mine.value.filter((x) => x.id !== own.value!.id))
  form.style = 'custom'
}

const validSet = computed(() => form.set === 'default' || /^[a-z0-9][a-z0-9_-]{0,23}$/.test(form.set))

// ---- jobs ------------------------------------------------------------------------------------
const active = computed(() => jobs.value.find((j) => !j.preview && ['queued', 'running'].includes(j.status)) ?? null)
const history = computed(() => jobs.value.filter((j) => !j.preview && !['queued', 'running'].includes(j.status)).slice(0, 6))
const preview = ref<CaptionJob | null>(null)
const busy = ref(false)

function body(extra: Record<string, unknown> = {}) {
  return {
    clip_name: form.clip_name,
    clip_type: 'auto', // the engine loads it as the plain chat model
    set: form.set,
    mode: form.mode,
    paths: form.mode === 'selected' ? props.selected : undefined,
    subject: form.subject || undefined,
    loras: loras.value.filter((l) => l.name),
    ...(editable.value ? { prompt: form.prompt } : { style: form.style }),
    ...extra,
  }
}
async function start(previewN = 0) {
  if (!previewN && form.set === 'default' && replaces.value) {
    const ok = confirm(
      `Replace ${replaces.value} existing caption${replaces.value === 1 ? '' : 's'} in the default set?\n\n` +
        'These are the captions training uses. The job keeps what they said, so Undo puts them back.',
    )
    if (!ok) return
  }
  busy.value = true
  error.value = null
  try {
    const { job } = await engine.startCaption(props.dataset, body({ preview: previewN, replace_default: !previewN && form.set === 'default' }))
    if (previewN) preview.value = job
    else {
      jobs.value = [job, ...jobs.value]
      preview.value = null
    }
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
async function act(j: CaptionJob, action: 'cancel' | 'undo') {
  if (action === 'undo' && !confirm(`Undo this job? ${j.written?.length ?? j.done} caption(s) in “${j.set_name}” go back to how they were (any you edited since stay).`))
    return
  try {
    const r = await engine.captionAction(j.job_id, action)
    upsert(r.job)
    if (action === 'undo') emit('changed', j.set_name)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
async function review(j: CaptionJob) {
  emit('review', j.written ? j : await engine.captionJob(j.job_id))
}
function upsert(j: CaptionJob) {
  if (j.preview) {
    if (preview.value?.job_id === j.job_id) preview.value = { ...preview.value, ...j }
    else if (!preview.value && ['queued', 'running'].includes(j.status)) preview.value = j
    return
  }
  const i = jobs.value.findIndex((x) => x.job_id === j.job_id)
  jobs.value = i < 0 ? [j, ...jobs.value] : jobs.value.map((x, n) => (n === i ? { ...x, ...j } : x))
}
let off = () => {}
onMounted(() => {
  off = onComfyMessage((m) => {
    if (m.type !== EVENTS.caption || m.data.dataset !== props.dataset) return
    const j = m.data as unknown as CaptionJob
    const before = jobs.value.find((x) => x.job_id === j.job_id)?.done ?? 0
    upsert(j)
    if (!j.preview && (j.done !== before || ['done', 'stopped', 'error'].includes(j.status))) emit('changed', j.set_name)
  })
})
onUnmounted(() => off())

const ago = (t?: number) => {
  if (!t) return ''
  const s = Date.now() / 1000 - t
  return s < 60 ? 'just now' : s < 3600 ? `${Math.round(s / 60)}m ago` : s < 86400 ? `${Math.round(s / 3600)}h ago` : new Date(t * 1000).toLocaleDateString()
}
const thumb = (path: string) => thumbUrl(props.dataset, path, 256)
</script>

<template>
  <div class="cp card">
    <div class="row">
      <label class="f model"><span class="eyebrow">Model</span>
        <ZenCombo
          v-model="form.clip_name"
          :items="models.map((n) => ({ value: n, label: n }))"
          :sorted="false"
          :item-height="30"
          placeholder="A vision-language model (Qwen-VL, Qwen3.5, Gemma 4)"
          size="sm"
        >
          <template #option="{ item }">
            <span class="mopt">
              <span class="senses"><i v-for="s in senseIcons(String(item.value))" :key="s.icon" :class="[s.icon, s.tone]" :title="s.tip" /></span>
              <span class="mname">{{ item.label }}</span>
            </span>
          </template>
          <template #selected="{ item }">
            <span v-if="item" class="mopt">
              <span class="senses"><i v-for="s in senseIcons(String(item.value))" :key="s.icon" :class="[s.icon, s.tone]" :title="s.tip" /></span>
              <span class="mname">{{ item.label }}</span>
            </span>
            <span v-else class="muted">A vision-language model</span>
          </template>
        </ZenCombo>
        <button
          v-if="hasVideo && hearing.length && !hearing.includes(form.clip_name)"
          class="hear-tip"
          :title="`${hearing[0]} hears each clip's soundtrack too: captions can describe the music and how the edit follows the beat`"
          @click="form.clip_name = hearing[0]!"
        >
          <i class="mdi mdi-music-note" /> Caption with sound
        </button>
        <div class="loras">
          <div v-for="(l, i) in loras" :key="i" class="lora">
            <ZenCombo v-model="l.name" :items="teLoras.map((n) => ({ value: n, label: n }))" :item-height="30" placeholder="A text-encoder LoRA" size="sm" />
            <ZenNumber v-model="l.strength" :min="-2" :max="2" :step="0.05" :precision="2" size="sm" />
            <button class="x" title="Remove" @click="loras.splice(i, 1)"><i class="mdi mdi-close" /></button>
          </div>
          <div class="lora-add">
            <button class="hear-tip" title="Patch the captioning model with a text-encoder LoRA (captioning only, never training)" @click="addLora"><i class="mdi mdi-plus" /> LoRA</button>
            <ZenInput v-model="loraLink" placeholder="or a Hugging Face link" size="sm" @keydown.enter="fetchLora()" />
            <ZenButton v-if="loraLink" size="sm" icon="mdi mdi-download-outline" @click="fetchLora()">Get</ZenButton>
          </div>
          <div v-if="suggested.length" class="suggested">
            <span class="muted small">For this model:</span>
            <button v-for="e in suggested" :key="e.id" class="chip-s" :class="{ on: isOn(e) }" :title="`${e.note}. ${e.repo}`" @click="useSuggested(e)">
              <i :class="isOn(e) ? 'mdi mdi-check' : e.installed ? 'mdi mdi-plus' : 'mdi mdi-download-outline'" />
              {{ e.label }}<span v-if="!e.installed" class="muted"> · {{ sizeOf(e.size) }}</span>
            </button>
          </div>
          <div v-if="loraChoices.length" class="choices">
            <button v-for="c in loraChoices" :key="c.file" class="hear-tip" @click="fetchLora(c.file)">{{ c.file }}</button>
          </div>
          <div v-if="loraNote" class="muted small">{{ loraNote }}</div>
        </div>
      </label>
      <div class="f"><span class="eyebrow">Style</span>
        <div class="seg">
          <button v-for="st in styles" :key="st.id" :class="{ on: form.style === st.id }" :title="st.prompt" @click="pick(st.id)">{{ st.label }}</button>
          <button v-for="st in mine" :key="st.id" :class="{ on: form.style === `u:${st.id}` }" :title="st.prompt" @click="pick(`u:${st.id}`)"><i class="mdi mdi-account-outline" /> {{ st.label }}</button>
          <button :class="{ on: form.style === 'custom' }" title="Write your own instruction" @click="pick('custom')"><i class="mdi mdi-plus" /> Custom</button>
        </div>
      </div>
    </div>
    <div v-if="editable" class="own">
      <ZenInput v-model="form.prompt" type="textarea" :rows="3" placeholder="What should the model write? It sees the image, e.g. “Describe the outfit and pose in one sentence.”" />
      <div class="own-acts">
        <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="!form.prompt.trim() || (!!own && !dirty)" :title="own ? 'Save your changes to this style' : 'Keep this instruction as a style of your own'" @click="saveStyle">
          {{ own ? (dirty ? 'Save changes' : 'Saved') : 'Save as style' }}
        </ZenButton>
        <button v-if="own" class="mini" title="Delete this style" @click="removeStyle"><i class="mdi mdi-trash-can-outline" /></button>
        <span class="muted small">Saved styles follow you to any browser on this ComfyUI.</span>
      </div>
    </div>

    <div class="row">
      <label class="f set"><span class="eyebrow">Into set</span>
        <ZenInput v-model="form.set" size="sm" :class="{ bad: !validSet }" list="cp-sets" />
        <datalist id="cp-sets"><option v-for="s in sets" :key="s" :value="s" /></datalist>
      </label>
      <div class="f"><span class="eyebrow">Images</span>
        <div class="seg">
          <button :class="{ on: form.mode === 'missing' }" title="Only images with no caption in this set" @click="form.mode = 'missing'">Missing {{ counts.missing }}</button>
          <button :class="{ on: form.mode === 'all' }" title="Every image (existing captions in this set are replaced)" @click="form.mode = 'all'">All {{ counts.all }}</button>
          <button :class="{ on: form.mode === 'selected' }" :disabled="!counts.selected" title="The images you've selected" @click="form.mode = 'selected'">Selected {{ counts.selected }}</button>
        </div>
      </div>
      <label class="f subject"><span class="eyebrow">Call the subject</span>
        <ZenInput v-model="form.subject" size="sm" placeholder="optional, e.g. a trigger word" />
      </label>
    </div>

    <div class="foot">
      <span class="small" :class="form.set === 'default' && replaces ? 'warn' : 'muted'">
        <i v-if="form.set === 'default' && replaces" class="mdi mdi-alert-outline" />
        {{ summary }}<template v-if="replaces"> · undoable</template>
      </span>
      <span class="spacer" />
      <ZenButton size="sm" icon="mdi mdi-eye-outline" :disabled="busy || !form.clip_name || !targets.length || !!active" title="Caption 3 of these without saving anything" @click="start(3)">Try on 3</ZenButton>
      <ZenButton size="sm" variant="primary" icon="mdi mdi-creation-outline" :disabled="busy || !form.clip_name || !validSet || !targets.length || !!active" @click="start()">
        Caption {{ targets.length || '' }}
      </ZenButton>
    </div>
    <div v-if="error" class="err small mono">{{ error }}</div>

    <!-- try-on results: nothing saved -->
    <div v-if="preview" class="preview">
      <div class="line">
        <b class="small">Preview</b>
        <span class="muted small">{{ preview.status === 'done' ? 'not saved: run it to write them' : preview.status === 'running' ? `${preview.done}/${preview.total || 3}${preview.message ? ` · ${preview.message}` : ''}` : preview.status === 'queued' && preview.message ? preview.message : preview.status }}</span>
        <span class="spacer" />
        <button class="x" title="Close the preview" @click="preview = null"><i class="mdi mdi-close" /></button>
      </div>
      <div class="pv-grid">
        <div v-for="r in preview.results ?? []" :key="r.path" class="pv">
          <img :src="thumb(r.path)" alt="" />
          <p class="small">{{ r.text }}</p>
        </div>
        <div v-for="n in Math.max(0, (preview.total || 3) - (preview.results?.length ?? 0))" :key="`w${n}`" class="pv waiting">
          <div class="ph"><i class="mdi mdi-creation-outline" /></div>
          <p class="small muted">{{ preview.status === 'queued' ? 'waiting for the GPU' : 'writing…' }}</p>
        </div>
      </div>
      <div v-if="preview.message && preview.status !== 'done'" class="err small">{{ preview.message }}</div>
    </div>

    <!-- the running job -->
    <div v-if="active" class="job live">
      <div class="line">
        <b class="small">{{ active.status === 'queued' ? 'Queued' : `Captioning ${active.done}/${active.total || '…'}` }}</b>
        <span class="muted small">into “{{ active.set_name }}”<template v-if="active.message"> · {{ active.message }}</template></span>
        <span class="spacer" />
        <ZenButton size="sm" icon="mdi mdi-stop" title="Stop at the next image; what's written stays (and can be undone)" @click="act(active, 'cancel')">Cancel</ZenButton>
      </div>
      <div class="bar"><div :style="{ width: `${active.total ? (active.done / active.total) * 100 : 0}%` }" /></div>
      <div v-if="active.text" class="small latest"><span class="mono muted">{{ active.file }}</span> {{ active.text }}</div>
    </div>

    <!-- what earlier jobs did -->
    <div v-if="history.length" class="history">
      <div class="eyebrow">Recent jobs</div>
      <div v-for="j in history" :key="j.job_id" class="job-row">
        <i :class="j.status === 'done' ? 'mdi mdi-check-circle-outline ok' : j.status === 'undone' ? 'mdi mdi-undo-variant muted' : 'mdi mdi-alert-circle-outline warn'" />
        <span class="small"><b>{{ j.set_name }}</b> · {{ j.done }} written<template v-if="j.replaced"> ({{ j.replaced }} replaced)</template></span>
        <span class="muted small ellipsis" :title="j.message">{{ j.status === 'done' ? ago(j.finished ?? j.created) : j.message || j.status }}</span>
        <span class="spacer" />
        <button v-if="j.done && j.status !== 'undone'" class="mini" title="Show only the images this job captioned" @click="review(j)">Review</button>
        <button v-if="j.done && j.status !== 'undone'" class="mini" title="Put back what this job changed" @click="act(j, 'undo')">Undo</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cp {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: flex-end;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.model {
  flex: 2 1 260px;
}
.type {
  flex: 0 1 140px;
}
.set {
  flex: 0 1 160px;
}
.subject {
  flex: 1 1 200px;
}
.bad :deep(input) {
  outline: 1px solid var(--zen-danger);
}
.seg {
  display: inline-flex;
  height: 30px;
  box-sizing: border-box;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 0 10px;
  border-radius: var(--r-md);
  cursor: pointer;
  white-space: nowrap;
}
.seg button.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.seg button:disabled {
  opacity: 0.4;
  cursor: default;
}
.foot,
.line,
.job-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.spacer {
  flex: 1;
}
.small {
  font-size: 12px;
}
.warn {
  color: var(--zen-warn);
}
.ok {
  color: var(--zen-ok);
}
.err {
  color: var(--zen-danger);
  white-space: pre-wrap;
}
.preview,
.job,
.history {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.own {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.own-acts {
  display: flex;
  align-items: center;
  gap: 8px;
}
.pv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
}
.pv {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.pv img,
.pv .ph {
  width: 72px;
  height: 72px;
  flex: none;
  object-fit: cover;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  display: grid;
  place-items: center;
  color: var(--zen-muted);
}
.pv p {
  margin: 0;
  line-height: 1.45;
  max-height: 120px;
  overflow: auto;
  scrollbar-width: thin;
}
.x,
.mini {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  border-radius: var(--r-md);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.x {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
}
.mini {
  height: 24px;
  padding: 0 8px;
}
.x:hover,
.mini:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.bar {
  height: 4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: var(--zen-accent);
  transition: width var(--zen-dur, 0.3s) ease;
}
.latest {
  color: var(--zen-text);
  line-height: 1.45;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.ellipsis {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.loras {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 6px;
}
.lora {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 84px 22px;
  gap: 4px;
  align-items: center;
}
.lora-add {
  display: flex;
  gap: 6px;
  align-items: center;
}
.lora-add > :nth-child(2) {
  flex: 1;
  min-width: 0;
}
.suggested {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}
.chip-s {
  font: inherit;
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 85%, transparent);
  background: none;
  color: var(--zen-text);
  cursor: pointer;
}
.chip-s:hover,
.chip-s.on {
  border-color: var(--zen-accent);
}
.chip-s.on i {
  color: var(--zen-accent);
}
.choices {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.x {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 0;
}
.x:hover {
  color: var(--zen-text);
}
.mopt {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  width: 100%;
}
.mname {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.senses {
  display: inline-flex;
  gap: 3px;
  flex: none;
  width: 52px; /* a fixed column: names line up after it */
  font-size: 14px;
}
.senses .ok {
  color: var(--zen-accent);
}
.senses .warn {
  color: var(--zen-warn);
}
.senses .dim {
  color: var(--zen-muted);
  opacity: 0.55;
}
.hear-tip {
  margin-top: 4px;
  align-self: flex-start;
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  font-size: 11px;
  color: var(--zen-accent);
  cursor: pointer;
}
.hear-tip:hover {
  text-decoration: underline;
}
</style>
