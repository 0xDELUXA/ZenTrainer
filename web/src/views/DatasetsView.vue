<script setup lang="ts">
import { ZenButton, ZenEmpty, ZenInput } from '@nynxz/zenkit-ui'
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { engine, thumbUrl, type DatasetInfo } from '@/api/engine'

const router = useRouter()
const datasets = ref<DatasetInfo[] | null>(null)
const error = ref<string | null>(null)
const creating = ref(false)
const newName = ref('')

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
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Datasets</h1>
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
    <div v-else class="grid">
      <article v-for="d in datasets ?? []" :key="d.folder" class="ds card">
        <RouterLink :to="`/datasets/${encodeURIComponent(d.folder)}`" class="mosaic">
          <img
            v-for="t in d.thumbs.slice(0, 4)"
            :key="t.filename"
            :src="thumbUrl(d.folder, `${t.subfolder.split('/').slice(1).concat(t.filename).join('/')}`, 256)"
            loading="lazy"
            alt=""
          />
        </RouterLink>
        <div class="body">
          <RouterLink :to="`/datasets/${encodeURIComponent(d.folder)}`" class="name">{{ d.folder }}</RouterLink>
          <div class="chips">
            <span class="chip"><i class="mdi mdi-image-outline" /> {{ d.images }}</span>
            <span class="chip" :class="{ warn: d.captioned < d.images }">
              <i class="mdi mdi-text-short" /> {{ d.captioned }}/{{ d.images }} captioned
            </span>
            <span v-if="d.steps_per_epoch !== d.images" class="chip"><i class="mdi mdi-repeat" /> {{ d.steps_per_epoch }}/epoch</span>
          </div>
          <RouterLink :to="{ path: '/runs/new', query: { dataset: d.folder } }">
            <ZenButton size="sm" icon="mdi mdi-play">Train on this</ZenButton>
          </RouterLink>
        </div>
      </article>
    </div>
  </div>
</template>

<style scoped>
.create {
  display: flex;
  gap: 8px;
  align-items: center;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
}
.ds {
  padding: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.mosaic {
  display: grid;
  transition: opacity 0.15s ease;
  grid-template-columns: 1fr 1fr;
  aspect-ratio: 4 / 3;
  gap: 2px;
  background: var(--zen-surface-2);
}
.mosaic:hover {
  opacity: 0.88;
}
.mosaic img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  min-height: 0;
}
.body {
  padding: 12px 14px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-start;
}
.name {
  font-weight: 650;
  font-size: 14px;
}
.chips {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.chip.warn i {
  color: var(--zen-warn);
}
</style>
