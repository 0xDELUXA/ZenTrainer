<script setup lang="ts">
// One run. Header with the controls; the progress line; the loss and validation curves over the
// samples; a rail of numbers, setup, what's in VRAM and the checkpoints. The parts are in
// components/run/ and share the run through its context (context.ts).
import { ZenButton, ZenNumber } from '@nynxz/zenkit-ui'
import { computed, provide, ref, toRef } from 'vue'
import { useRouter } from 'vue-router'
import LossChart from '@/components/LossChart.vue'
import StatusPill from '@/components/StatusPill.vue'
import { RUN, useRun } from '@/components/run/context'
import RunCheckpoints from '@/components/run/RunCheckpoints.vue'
import RunLoaded from '@/components/run/RunLoaded.vue'
import RunNoise from '@/components/run/RunNoise.vue'
import RunProgress from '@/components/run/RunProgress.vue'
import RunSamples from '@/components/run/RunSamples.vue'
import RunSetup from '@/components/run/RunSetup.vue'
import RunStats from '@/components/run/RunStats.vue'

const props = defineProps<{ id: string }>()
const router = useRouter()

const run = useRun(toRef(props, 'id'))
provide(RUN, run)
const { detail, error, busy, lrInput, status, running, total, marks, likeSteps, send, retryable, canResume, resumePoint, resume } = run
const lrOpen = ref(false)
// resolution stages for the speed strip: each lower stage trains until its `until`, then the
// next resolution blends in over `stage_blend` of the run
const stages = computed(() => {
  const d = detail.value?.spec.dataset
  const steps = total.value
  if (!d?.stages?.length || !steps) return []
  const blend = d.stage_blend ?? 0.2
  const ladder = [...d.stages].sort((a, b) => a.until - b.until)
  return ladder.map((st, i) => ({
    start: Math.round(st.until * steps),
    end: Math.round(Math.min(1, st.until + blend) * steps),
    label: `→ ${ladder[i + 1]?.resolution ?? d.resolution}`,
  }))
})
const modelFile = computed(() => {
  const m = detail.value?.spec.model
  return (m?.kind === 'split' ? m.unet_name : m?.ckpt_name) ?? ''
})
</script>

<template>
  <div class="page wide">
    <div v-if="error && !detail" class="card muted">{{ error }}</div>
    <template v-else-if="detail">
      <div class="head">
        <div class="titles">
          <div class="title-row">
            <h1>{{ detail.run.name }}</h1>
            <StatusPill :status="detail.run.status" />
          </div>
          <p class="lede mono">
            <span class="lede-part" :title="detail.run.run_id">{{ detail.run.run_id }}</span>
            <span class="lede-part" :title="modelFile">{{ modelFile }}</span>
          </p>
        </div>
        <div class="actions">
          <template v-if="running">
            <ZenButton v-if="status === 'paused'" variant="primary" icon="mdi mdi-play" :disabled="!!busy" @click="send('resume')">Resume</ZenButton>
            <ZenButton v-else icon="mdi mdi-pause" :disabled="!!busy || status !== 'training'" @click="send('pause')">Pause</ZenButton>
            <ZenButton icon="mdi mdi-content-save-outline" :disabled="!!busy || status !== 'training'" title="Save a checkpoint now" @click="send('save')">Save</ZenButton>
            <div class="lr-wrap">
              <ZenButton icon="mdi mdi-tune-variant" title="Change the learning rate" @click="lrOpen = !lrOpen">LR</ZenButton>
              <div v-if="lrOpen" class="lr-pop card">
                <span class="eyebrow">Learning rate</span>
                <ZenNumber v-model="lrInput" :min="0" :step="0.00001" :precision="6" size="sm" />
                <ZenButton size="sm" variant="primary" @click="send('set_lr', { lr: lrInput }); lrOpen = false">Apply</ZenButton>
              </div>
            </div>
            <ZenButton variant="danger" icon="mdi mdi-stop" :disabled="!!busy" @click="send('stop')">Stop</ZenButton>
          </template>
          <RouterLink v-if="retryable" :to="`/runs/new?edit=${id}`"><ZenButton icon="mdi mdi-pencil-outline" title="Change its settings, then it goes back in the queue">Edit &amp; retry</ZenButton></RouterLink>
          <RouterLink v-else-if="canResume" :to="`/runs/new?edit=${id}`"><ZenButton icon="mdi mdi-tune-variant" title="Change memory, samples, learning rate or length, then resume from its last state">Edit &amp; resume</ZenButton></RouterLink>
          <ZenButton v-if="canResume" variant="primary" icon="mdi mdi-play-circle-outline" :disabled="!!busy" @click="resume()">
            {{ resumePoint ? `Resume from ${resumePoint}` : 'Restart' }}
          </ZenButton>
          <ZenButton icon="mdi mdi-content-copy" title="New run with these settings" @click="router.push(`/runs/new?from=${id}`)">Duplicate</ZenButton>
        </div>
      </div>
      <div v-if="error" class="card err mono">{{ error }}</div>
      <div v-if="detail.run.message && !running" class="card muted mono msg">{{ detail.run.message }}</div>

      <RunProgress />

      <div class="layout">
        <div class="main">
          <section class="card">
            <LossChart :rows="detail.metrics" :total="total" :marks="marks" :val="detail.validation" :like="likeSteps" :stages="stages" :height="200" />
          </section>
          <RunSamples />
          <RunNoise />
        </div>
        <aside class="rail">
          <RunStats />
          <RunSetup />
          <RunLoaded />
          <RunCheckpoints />
        </aside>
      </div>
    </template>
  </div>
</template>

<style scoped>
.wide {
  max-width: 1320px;
}
.head {
  display: flex;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
}
/* the title keeps a readable width: past it the actions wrap below instead of squeezing it into a
   column of single characters */
.titles {
  flex: 1 1 320px;
  min-width: min(320px, 100%);
}
.title-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.title-row h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 650;
  letter-spacing: -0.01em;
}
.lede {
  margin: 4px 0 0;
  font-size: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 0 12px;
}
/* the run id and the model wrap between them, each cut off with an ellipsis if it alone is too long */
.lede-part {
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}
.lr-wrap {
  position: relative;
}
.lr-pop {
  position: absolute;
  right: 0;
  top: calc(100% + 6px);
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 200px;
  padding: 12px;
  box-shadow: 0 10px 30px rgb(0 0 0 / 0.3);
  background: var(--zen-surface);
}
.err {
  border-color: var(--zen-danger);
  font-size: 12px;
}
.msg {
  font-size: 12px;
  white-space: pre-wrap;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 14px;
  align-items: start;
}
.main,
.rail {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}
.rail {
  position: sticky;
  top: 16px;
}
@media (max-width: 1000px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .rail {
    position: static;
  }
}
</style>
