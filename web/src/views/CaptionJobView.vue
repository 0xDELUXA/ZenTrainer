<script setup lang="ts">
// One caption job: what it was asked to do, its progress, every caption it wrote, and its log.
// Live while it runs (the websocket's caption events trigger a re-read).
import { ZenButton } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { EngineError, engine, thumbUrl, type CaptionJob } from '@/api/engine'
import StatusPill from '@/components/StatusPill.vue'
import { captionJobs, startCaptionSync } from '@/stores/captions'

const props = defineProps<{ id: string }>()
const job = ref<CaptionJob | null>(null)
const error = ref<string | null>(null)

async function load() {
  try {
    job.value = await engine.captionJob(props.id)
    error.value = null
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
onMounted(() => Promise.all([load(), startCaptionSync()]))
// re-read on its events, at most every 800 ms
let pending: ReturnType<typeof setTimeout> | undefined
watch(
  () => {
    const j = captionJobs[props.id]
    return j ? `${j.status}:${j.done}:${j.message}` : ''
  },
  (v) => {
    if (!v || pending) return
    pending = setTimeout(() => {
      pending = undefined
      load()
    }, 800)
  },
)

const active = computed(() => !!job.value && ['queued', 'running'].includes(job.value.status))
const total = computed(() => job.value?.total || job.value?.preview || 0)
const pct = computed(() => (total.value ? (job.value!.done / total.value) * 100 : 0))
const title = computed(() => (job.value ? `${job.value.preview ? 'Try captions' : 'Caption'} ${job.value.dataset}` : 'Caption job'))
const written = computed(() => {
  const j = job.value
  if (!j) return []
  if (j.preview) return (j.results ?? []).map((r) => ({ path: r.path, text: r.text }))
  return Object.entries(j.captions ?? {}).map(([path, text]) => ({ path, text }))
})
const facts = computed(() => {
  const j = job.value
  if (!j) return []
  return [
    ['Dataset', j.dataset],
    ['Into set', j.set_name],
    ['Images', j.preview ? `try-on of ${j.preview}` : j.mode ?? 'missing'],
    ['Model', j.clip_name],
    ['Replaced', j.replaced ? `${j.replaced} existing (undo puts them back)` : 'none'],
    ...(j.subject ? [['Subject', j.subject]] : []),
  ] as [string, string][]
})
const when = (t?: number | null) => (t ? new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' }) : '')
const clock = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
const took = computed(() => {
  const j = job.value
  if (!j?.created || !j.finished) return ''
  const s = Math.round(j.finished - j.created)
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
})

async function act(action: 'cancel' | 'undo') {
  if (!job.value) return
  if (action === 'undo' && !confirm(`Undo this job? ${written.value.length} caption(s) in “${job.value.set_name}” go back to how they were (any you edited since stay).`)) return
  try {
    await engine.captionAction(job.value.job_id, action)
    await load()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/jobs" class="crumb">Jobs</RouterLink> {{ title }}</h1>
        <p v-if="job" class="lede">
          {{ when(job.created) }}<template v-if="took"> · took {{ took }}</template>
        </p>
      </div>
      <template v-if="job">
        <StatusPill :status="job.status" :label="job.status === 'running' ? 'captioning' : undefined" />
        <RouterLink :to="`/datasets/${encodeURIComponent(job.dataset)}${!job.preview && written.length ? `?review=${job.job_id}` : ''}`">
          <ZenButton icon="mdi mdi-image-multiple-outline">{{ !job.preview && written.length ? 'Review in dataset' : 'Open dataset' }}</ZenButton>
        </RouterLink>
        <ZenButton v-if="active" icon="mdi mdi-stop" title="Stop at the next image; what's written stays (and can be undone)" @click="act('cancel')">Cancel</ZenButton>
        <ZenButton v-else-if="!job.preview && job.status !== 'undone' && written.length" icon="mdi mdi-undo-variant" @click="act('undo')">Undo</ZenButton>
      </template>
    </div>
    <div v-if="error" class="card err small mono">{{ error }}</div>

    <template v-if="job">
      <div v-if="active || job.message" class="card prog">
        <div class="line">
          <b class="small">{{ job.status === 'queued' ? 'Queued' : active ? `${job.done} of ${total || '…'}` : job.status }}</b>
          <span class="muted small">{{ job.message }}</span>
        </div>
        <div v-if="active" class="bar"><div :style="{ width: `${pct}%` }" /></div>
      </div>

      <div class="cols">
        <section class="card">
          <div class="eyebrow">{{ job.preview ? 'Try-on (not saved)' : 'Captions written' }} · {{ written.length }}</div>
          <div v-if="!written.length" class="muted small">None yet.</div>
          <div v-for="w in written" :key="w.path" class="cap">
            <img :src="thumbUrl(job.dataset, w.path, 128)" loading="lazy" alt="" />
            <div>
              <div class="mono muted small">{{ w.path }}</div>
              <p>{{ w.text }}</p>
            </div>
          </div>
        </section>
        <aside class="side">
          <section class="card">
            <div class="eyebrow">Job</div>
            <dl>
              <template v-for="[k, v] in facts" :key="k"><dt>{{ k }}</dt><dd :class="{ mono: k === 'Model' }">{{ v }}</dd></template>
            </dl>
            <details v-if="job.prompt" class="prompt">
              <summary class="small muted">Instruction</summary>
              <p class="small">{{ job.prompt }}</p>
            </details>
          </section>
          <section class="card">
            <div class="eyebrow">Log</div>
            <div class="log mono">
              <div v-for="(l, i) in job.log ?? []" :key="i" :class="l.level">
                <span class="t">{{ clock(l.t) }}</span> {{ l.msg }}
              </div>
              <div v-if="!job.log?.length" class="muted">No log for this job (it ran before jobs kept one).</div>
            </div>
          </section>
        </aside>
      </div>
    </template>
  </div>
</template>

<style scoped>
.titles h1 {
  display: flex;
  align-items: baseline;
  gap: 6px;
}
.prog {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.line {
  display: flex;
  gap: 10px;
  align-items: baseline;
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
  transition: width var(--zen-dur) ease;
}
.cols {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 380px);
  gap: 16px;
  align-items: start;
}
.cols > section,
.side > section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.side {
  display: flex;
  flex-direction: column;
  gap: 16px;
  position: sticky;
  top: 16px;
}
.cap {
  display: grid;
  grid-template-columns: 64px 1fr;
  gap: 12px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.cap img {
  width: 64px;
  height: 64px;
  object-fit: cover;
  border-radius: var(--r-md);
}
.cap p {
  margin: 4px 0 0;
  font-size: 13px;
  line-height: 1.55;
}
dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 12px;
  margin: 0;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
  min-width: 0;
  overflow-wrap: anywhere;
}
.prompt p {
  margin: 6px 0 0;
  line-height: 1.5;
}
.log {
  font-size: 11px;
  line-height: 1.6;
  max-height: 50vh;
  overflow: auto;
  scrollbar-width: thin;
}
.log > div {
  overflow-wrap: anywhere;
}
.log .t {
  color: var(--zen-muted);
}
.log .warn {
  color: var(--zen-warn);
}
.log .error {
  color: var(--zen-danger);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 900px) {
  .cols {
    grid-template-columns: 1fr;
  }
  .side {
    position: static;
  }
}
</style>
