<script setup lang="ts">
// Simple mode's view of a run: what it's doing in plain words, a big progress bar with the time
// left, and the samples as they come (before, then each one after). No loss chart, no knobs;
// the full run page is one click away.
import { ZenButton } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { ACTIVE, EVENTS, engine, type RunDetail, type SampleRow } from '@/api/engine'
import { eta } from '@/recipes'

const props = defineProps<{ id: string }>()
const d = ref<RunDetail | null>(null)
const sampling = ref<{ index?: number; count?: number; s_step?: number; s_total?: number; preview?: string } | null>(null)
const now = ref(Date.now() / 1000)
let timer: ReturnType<typeof setInterval> | undefined
let off = () => {}

async function load() {
  d.value = await engine.run(props.id).catch(() => d.value)
}
onMounted(() => {
  load()
  timer = setInterval(() => {
    now.value = Date.now() / 1000
    if (d.value && !ACTIVE.includes(d.value.run.status)) return
    if (Math.round(now.value) % 5 === 0) load() // status and messages, in case an event was missed
  }, 1000)
  off = onComfyMessage((m) => {
    if (!d.value || m.data.run_id !== props.id) return
    if (m.type === EVENTS.metrics) d.value.metrics.push(m.data as never)
    else if (m.type === EVENTS.sample) d.value.samples.push(m.data as unknown as SampleRow)
    else if (m.type === EVENTS.status) {
      d.value.run.status = m.data.status as never
      d.value.run.message = (m.data.message as string) ?? undefined
      if (!ACTIVE.includes(d.value.run.status)) load()
    } else if (m.type === EVENTS.sampleProgress) {
      const p = m.data as Record<string, unknown>
      sampling.value = p.phase === 'done' ? null : { ...sampling.value, ...(p as object) }
    }
  })
})
onUnmounted(() => {
  off()
  clearInterval(timer)
})

const status = computed(() => d.value?.run.status)
const total = computed(() => d.value?.spec.optim.steps ?? 0)
const last = computed(() => d.value?.metrics.at(-1))
const step = computed(() => last.value?.step ?? 0)
const pct = computed(() => (total.value ? Math.min(100, (step.value / total.value) * 100) : 0))
const left = computed(() => (last.value?.it_s && status.value === 'training' ? (total.value - step.value) / last.value.it_s : null))
const done = computed(() => status.value === 'done')
const failed = computed(() => status.value === 'error' || status.value === 'rejected')

const headline = computed(() => {
  const s = status.value
  if (!s) return 'Loading…'
  if (s === 'queued') return 'Waiting for its turn'
  if (s === 'loading') return 'Getting ready'
  if (sampling.value) return step.value === 0 ? 'Making a "before" picture' : 'Making a new preview'
  if (s === 'training') return step.value ? 'Learning' : 'Starting'
  if (s === 'paused') return 'Paused'
  if (s === 'done') return 'Your LoRA is ready'
  if (s === 'stopped' || s === 'interrupted') return 'Stopped'
  return 'Something went wrong'
})
const detailLine = computed(() => {
  const s = status.value
  if (s === 'loading') return d.value?.run.message || 'Loading the model and your photos'
  if (sampling.value?.s_total) return `${sampling.value.s_step ?? 0} of ${sampling.value.s_total}`
  if (s === 'training' && left.value != null) return `${eta(left.value)} left`
  if (failed.value) return d.value?.run.message ?? ''
  if (done.value) return 'Use the word below in a prompt to bring it in.'
  return ''
})

// the samples: the first prompt's, in order (before first); the newest big
const viewUrl = (x: SampleRow) => `/view?${new URLSearchParams({ filename: x.filename, subfolder: x.subfolder, type: x.type })}`
const isVideo = (x: SampleRow) => /\.(mp4|webm)$/i.test(x.filename)
const strip = computed(() => (d.value?.samples ?? []).filter((x) => x.index === 0).sort((a, b) => a.step - b.step))
const pick = ref<number | null>(null)
const big = computed(() => (pick.value != null ? strip.value[pick.value] : strip.value.at(-1)) ?? null)
const finalLora = computed(() => [...(d.value?.checkpoints ?? [])].reverse().find((c) => c.final)?.lora_name ?? null)
const tryLink = computed(() => (finalLora.value ? `/apps/text-to-image?run=${props.id}&lora=${encodeURIComponent(finalLora.value)}` : null))
</script>

<template>
  <div class="page srun">
    <div v-if="d" class="hero card" :class="{ done, failed }">
      <div class="top">
        <div>
          <div class="eyebrow">{{ d.spec.name }}</div>
          <h1>{{ headline }}</h1>
          <p class="muted">{{ detailLine }}</p>
        </div>
        <i class="big-ic" :class="done ? 'mdi mdi-check-circle' : failed ? 'mdi mdi-alert-circle-outline' : 'mdi mdi-sprout-outline'" />
      </div>
      <div v-if="!done && !failed" class="bar"><span :style="{ width: `${pct}%` }" /></div>
      <div class="meta small muted">
        <span v-if="!done && !failed">{{ Math.round(pct) }}%</span>
        <span>Trigger word <b class="mono">{{ d.spec.dataset.trigger || '–' }}</b></span>
        <span class="spacer" />
        <RouterLink :to="`/runs/${id}`" class="muted">Details</RouterLink>
      </div>
      <div v-if="done" class="acts">
        <RouterLink v-if="tryLink" :to="tryLink"><ZenButton variant="primary" icon="mdi mdi-creation-outline">Try it</ZenButton></RouterLink>
        <RouterLink to="/simple"><ZenButton icon="mdi mdi-plus">Train another</ZenButton></RouterLink>
      </div>
    </div>

    <section v-if="d" class="card previews">
      <div class="eyebrow">Previews</div>
      <div v-if="big" class="stage">
        <video v-if="isVideo(big)" :key="big.filename" :src="viewUrl(big)" autoplay loop muted playsinline controls />
        <img v-else :src="viewUrl(big)" alt="" />
        <span class="tag small">{{ big.step === 0 ? 'Before training' : `After ${big.step} steps` }}</span>
      </div>
      <div v-else-if="sampling?.preview" class="stage"><img :src="sampling.preview" alt="" /><span class="tag small">Drawing…</span></div>
      <div v-else class="muted small">The first preview shows up in a few minutes.</div>
      <div v-if="strip.length > 1" class="strip">
        <button v-for="(x, i) in strip" :key="x.filename" :class="{ on: big === x }" @click="pick = i === strip.length - 1 ? null : i">
          <video v-if="isVideo(x)" :src="viewUrl(x)" muted preload="metadata" />
          <img v-else :src="viewUrl(x)" loading="lazy" alt="" />
          <span class="small">{{ x.step === 0 ? 'Before' : x.step }}</span>
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.srun {
  max-width: 900px;
}
.hero {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 22px;
}
.top {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}
.top h1 {
  margin: 4px 0;
  font-size: 26px;
}
.top p {
  margin: 0;
}
.big-ic {
  font-size: 44px;
  color: var(--zen-accent);
}
.done .big-ic {
  color: var(--zen-ok);
}
.failed .big-ic {
  color: var(--zen-danger);
}
.bar {
  height: 12px;
  border-radius: 6px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
  transition: width 0.6s ease;
}
.meta {
  display: flex;
  gap: 16px;
  align-items: center;
}
.spacer {
  flex: 1;
}
.acts {
  display: flex;
  gap: 8px;
}
.previews {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.stage {
  position: relative;
  display: grid;
  place-items: center;
  background: var(--zen-surface-2);
  border-radius: var(--r-lg, 12px);
  overflow: hidden;
  max-height: 60vh;
}
.stage img,
.stage video {
  max-width: 100%;
  max-height: 60vh;
  object-fit: contain;
  display: block;
}
.tag {
  position: absolute;
  left: 10px;
  top: 10px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
}
.strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  scrollbar-width: thin;
}
.strip button {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: center;
  border: 2px solid transparent;
  background: none;
  padding: 2px;
  border-radius: var(--r-md);
  color: var(--zen-muted);
  cursor: pointer;
}
.strip button.on {
  border-color: var(--zen-accent);
  color: var(--zen-text);
}
.strip img,
.strip video {
  width: 88px;
  height: 88px;
  object-fit: cover;
  border-radius: var(--r-sm);
  display: block;
}
.small {
  font-size: 12px;
}
.muted {
  color: var(--zen-muted);
}
</style>
