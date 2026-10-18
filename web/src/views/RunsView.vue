<script setup lang="ts">
// Runs and the training queue on one page: what's training now, what's next (in order, still
// reorderable: those haven't been handed to ComfyUI yet), then everything that has finished.
import { ZenButton, ZenEmpty } from '@nynxz/zenkit-ui'
import { computed, onMounted } from 'vue'
import { engine } from '@/api/engine'
import StatusPill from '@/components/StatusPill.vue'
import { engineError, latest, queue, refreshQueue, refreshRuns, runs } from '@/runs'

onMounted(() => Promise.all([refreshRuns(), refreshQueue()]))

async function act(body: Parameters<typeof engine.editQueue>[0]) {
  await engine.editQueue(body)
  await Promise.all([refreshQueue(), refreshRuns()])
}
function move(i: number, by: number) {
  const order = queue.value.items.map((r) => r.run_id)
  const j = i + by
  if (j < 0 || j >= order.length) return
  ;[order[i], order[j]] = [order[j]!, order[i]!]
  act({ action: 'order', order })
}

const current = computed(() => queue.value.current)
const queued = computed(() => queue.value.items)
const history = computed(() => {
  const live = new Set([current.value?.run_id, ...queued.value.map((r) => r.run_id)])
  return runs.value.filter((r) => !live.has(r.run_id))
})
const progress = (id: string) => {
  const m = latest[id]
  return m?.total ? { step: m.step, total: m.total, pct: (m.step / m.total) * 100 } : null
}
const when = (t: number) => new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
const summary = computed(() => {
  const parts = [`${runs.value.length} run${runs.value.length === 1 ? '' : 's'}`]
  if (current.value) parts.push('1 training')
  if (queued.value.length) parts.push(`${queued.value.length} queued`)
  return parts.join(' · ')
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Runs</h1>
        <p class="lede">
          {{ summary }}
          <template v-if="queue.paused"> · <b class="warn">queue paused</b></template>
        </p>
      </div>
      <ZenButton
        v-if="queue.paused || queued.length"
        :variant="queue.paused ? 'primary' : 'default'"
        :icon="queue.paused ? 'mdi mdi-play' : 'mdi mdi-pause'"
        :title="queue.paused ? 'Let queued runs start again' : 'Let the current run finish, then hold the rest'"
        @click="act({ action: queue.paused ? 'resume' : 'pause' })"
        >{{ queue.paused ? 'Resume queue' : 'Pause queue' }}</ZenButton
      >
      <RouterLink to="/runs/new"><ZenButton variant="primary" icon="mdi mdi-plus">New run</ZenButton></RouterLink>
    </div>
    <div v-if="engineError" class="card muted">Can't reach the engine: {{ engineError }}</div>
    <ZenEmpty v-else-if="!runs.length" title="No runs yet" icon="mdi mdi-chart-line">
      Start one from <RouterLink to="/runs/new"><u>New run</u></RouterLink>.
    </ZenEmpty>

    <template v-else>
      <section v-if="current" class="sec">
        <div class="eyebrow">Now</div>
        <RouterLink :to="`/runs/${current.run_id}`" class="row card now">
          <div class="name">
            {{ current.name }}
            <div class="muted mono small">{{ current.run_id }}</div>
          </div>
          <div class="mini">
            <div class="bar"><div :style="{ width: `${progress(current.run_id)?.pct ?? 0}%` }" /></div>
            <span class="muted mono small">{{ progress(current.run_id) ? `${progress(current.run_id)!.step}/${progress(current.run_id)!.total}` : 'starting' }}</span>
          </div>
          <StatusPill :status="current.status" />
        </RouterLink>
      </section>

      <section v-if="queued.length" class="sec">
        <div class="eyebrow">Up next · {{ queued.length }}</div>
        <div v-for="(r, i) in queued" :key="r.run_id" class="row card queued">
          <RouterLink :to="`/runs/${r.run_id}`" class="name">
            <span class="pos mono">{{ i + 1 }}</span>
            <span>
              {{ r.name }}
              <span class="muted mono small block">{{ r.run_id }} · {{ r.resume_from ? 'resumes' : 'new' }}</span>
            </span>
          </RouterLink>
          <div class="acts">
            <RouterLink class="edit" :to="`/runs/new?edit=${r.run_id}`" title="Change its settings or name; it keeps its place"><i class="mdi mdi-pencil-outline" /></RouterLink>
            <button title="Move to top" :disabled="i === 0" @click="act({ action: 'top', run_id: r.run_id })"><i class="mdi mdi-format-vertical-align-top" /></button>
            <button title="Move up" :disabled="i === 0" @click="move(i, -1)"><i class="mdi mdi-chevron-up" /></button>
            <button title="Move down" :disabled="i === queued.length - 1" @click="move(i, 1)"><i class="mdi mdi-chevron-down" /></button>
            <button class="danger" title="Remove from the queue (the run stays, unstarted)" @click="act({ action: 'remove', run_id: r.run_id })"><i class="mdi mdi-close" /></button>
          </div>
          <StatusPill :status="r.status" />
        </div>
      </section>

      <section v-if="history.length" class="sec">
        <div v-if="current || queued.length" class="eyebrow">History</div>
        <RouterLink v-for="r in history" :key="r.run_id" :to="`/runs/${r.run_id}`" class="row card">
          <div class="name">
            {{ r.name }}
            <div class="muted mono small">{{ r.run_id }}</div>
          </div>
          <div class="muted small">{{ when(r.created) }}</div>
          <StatusPill :status="r.status" />
        </RouterLink>
      </section>
    </template>
  </div>
</template>

<style scoped>
.sec {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.warn {
  color: var(--zen-warn);
}
.row {
  display: grid;
  grid-template-columns: 1fr 200px 110px;
  gap: 16px;
  align-items: center;
  padding: 12px 16px;
}
a.row:hover,
.queued a.name:hover {
  border-color: var(--zen-accent);
  color: var(--zen-accent);
}
a.row:hover {
  color: inherit;
}
.now {
  border-color: color-mix(in srgb, var(--zen-accent) 50%, transparent);
}
.name {
  font-weight: 600;
  min-width: 0;
}
.queued .name {
  display: flex;
  align-items: center;
  gap: 12px;
}
.pos {
  width: 22px;
  flex: none;
  text-align: center;
  color: var(--zen-muted);
  font-weight: 400;
}
.block {
  display: block;
}
.small {
  font-size: 11px;
  font-weight: 400;
}
.mini {
  display: flex;
  align-items: center;
  gap: 8px;
}
.bar {
  flex: 1;
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
.acts {
  display: flex;
  gap: 2px;
  justify-content: flex-end;
}
.acts button,
.acts .edit {
  display: inline-grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font-size: 16px;
  cursor: pointer;
}
.acts .edit:hover,
.acts button:hover:not(:disabled) {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.acts button.danger:hover {
  color: var(--zen-danger);
}
.acts button:disabled {
  opacity: 0.3;
  cursor: default;
}
@media (max-width: 720px) {
  .row {
    grid-template-columns: 1fr auto;
  }
  .row > :nth-child(2) {
    display: none;
  }
}
</style>
