<script setup lang="ts">
// Apps: ComfyUI workflows with an interface built for the job. Each is a folder (app.json,
// its workflows, a cover): ours under engine/apps, yours under user/zentrainer/apps.
import { ZenEmpty } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { engine, type AppInfo } from '@/api/engine'
import TemplateCard from '@/components/TemplateCard.vue'

const apps = ref<AppInfo[]>([])
const errors = ref<string[]>([])
const loaded = ref(false)

onMounted(async () => {
  try {
    const r = await engine.apps()
    apps.value = r.apps
    errors.value = r.errors
  } finally {
    loaded.value = true
  }
})
// each card's picture: the app's own cover, else a ComfyUI template thumbnail of the same kind
// of job (never your own outputs: those can be anything, and live in the Gallery)
const FALLBACK: Record<string, string> = {
  'text-to-image': '/templates/image_z_image_turbo-1.webp',
  'test-lora': '/templates/image_krea2_turbo_int8_image_style_reference-1.webp',
}
const cover = (a: AppInfo) => (a.cover ? `/zentrainer/api/apps/${a.id}/cover` : (FALLBACK[a.id] ?? null))

const CATEGORIES: Record<string, string> = { generate: 'Generate', train: 'Training', edit: 'Edit', caption: 'Caption' }
const byCategory = computed(() => {
  const out = new Map<string, AppInfo[]>()
  for (const a of apps.value) out.set(a.category, [...(out.get(a.category) ?? []), a])
  return [...out.entries()]
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Apps</h1>
      </div>
    </div>
    <div v-if="errors.length" class="card warn">
      <b>Some apps didn't load</b>
      <div v-for="e in errors" :key="e" class="mono small">{{ e }}</div>
    </div>
    <ZenEmpty v-if="loaded && !apps.length" title="No apps" icon="mdi mdi-view-grid-outline" />
    <section v-for="[cat, list] in byCategory" :key="cat" class="cat">
      <div class="eyebrow">{{ CATEGORIES[cat] ?? cat }}</div>
      <div class="grid">
        <TemplateCard
          v-for="a in list"
          :key="a.id"
          :to="`/apps/${a.id}`"
          :title="a.name"
          :description="a.description"
          :image="cover(a)"
          :icon="a.icon ?? 'mdi mdi-application-outline'"
          :tags="[...(a.builtin ? [] : ['yours']), ...(a.families.length ? [`${a.families.length} models`] : [])]"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.cat {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}
.warn {
  border-color: var(--zen-warn);
}
</style>
