<script setup lang="ts">
// Split a dataset's long videos into short clips (engine: datasets.split_videos).
import { ZenButton, ZenCombo, ZenNumber } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { EngineError, engine, type DatasetInfo } from '@/api/engine'

const datasets = ref<DatasetInfo[]>([])
const name = ref('')
const seconds = ref(3.1)
const count = ref(20)
const longer = ref(20)
const busy = ref(false)
const error = ref<string | null>(null)
const result = ref<{ clips: number; moved: string[] } | null>(null)

onMounted(async () => {
  datasets.value = await engine.datasets().catch(() => [])
  name.value = datasets.value.find((d) => d.videos)?.folder ?? ''
})
const items = computed(() =>
  datasets.value.map((d) => ({ value: d.folder, label: `${d.folder}${d.videos ? ` · ${d.videos} video${d.videos === 1 ? '' : 's'}` : ' · no videos'}` })),
)
const frames = computed(() => Math.round(seconds.value * 24))

async function run() {
  busy.value = true
  error.value = null
  result.value = null
  try {
    result.value = await engine.splitVideos(name.value, { seconds: seconds.value, count: count.value, longer_than: longer.value })
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="page split">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/tools" class="crumb">Tools</RouterLink> Split videos into clips</h1>
        <p class="lede">Long videos become short clips spread through each one; the originals move to _long/.</p>
      </div>
    </div>
    <section class="card col">
      <label class="f"><span class="small muted">Dataset</span><ZenCombo v-model="name" :items="items" :sorted="false" placeholder="Pick a dataset…" /></label>
      <div class="grid-2">
        <label class="f">
          <span class="small muted">Clip length (seconds)</span>
          <ZenNumber v-model="seconds" :min="0.5" :max="30" :step="0.1" :precision="1" />
          <span class="small muted">{{ frames }} frames at 24 fps. MiniMax H3 trains 73-frame clips from 3.1 s.</span>
        </label>
        <label class="f">
          <span class="small muted">Clips per video</span>
          <ZenNumber v-model="count" :min="0" :max="500" :step="5" :precision="0" />
          <span class="small muted">Spread evenly through each video; 0 = every clip it holds.</span>
        </label>
        <label class="f">
          <span class="small muted">Only videos longer than (seconds)</span>
          <ZenNumber v-model="longer" :min="1" :max="3600" :step="5" :precision="0" />
        </label>
      </div>
      <div v-if="error" class="err small">{{ error }}</div>
      <div class="acts">
        <ZenButton variant="primary" icon="mdi mdi-content-cut" :disabled="!name || busy" @click="run">{{ busy ? 'Splitting…' : 'Split' }}</ZenButton>
      </div>
      <div v-if="result" class="done small">
        <i class="mdi mdi-check-circle-outline" />
        <span>{{ result.clips }} clips from {{ result.moved.length }} video{{ result.moved.length === 1 ? '' : 's' }}.</span>
        <RouterLink :to="`/datasets/${encodeURIComponent(name)}`"><ZenButton size="sm" icon="mdi mdi-image-multiple-outline">Open dataset</ZenButton></RouterLink>
      </div>
    </section>
  </div>
</template>

<style scoped>
.split {
  max-width: 900px;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.acts {
  display: flex;
}
.done {
  display: flex;
  align-items: center;
  gap: 10px;
}
.done i {
  color: var(--zen-ok);
  font-size: 18px;
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
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
</style>
