<script setup lang="ts">
// Apps: ComfyUI workflows with an interface built for the job. Each is a folder (app.json,
// its workflows, a cover): ours under engine/apps, yours under user/zentrainer/apps.
import { ZenEmpty } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { engine, type AppInfo } from '@/api/engine'

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
        <RouterLink v-for="a in list" :key="a.id" :to="`/apps/${a.id}`" class="app card">
          <img v-if="a.cover" :src="`/zentrainer/api/apps/${a.id}/cover`" alt="" class="cover" />
          <i v-else :class="a.icon ?? 'mdi mdi-application-outline'" class="icon" />
          <div>
            <div class="name">{{ a.name }} <span v-if="!a.builtin" class="chip">yours</span></div>
            <p class="muted desc">{{ a.description }}</p>
            <p v-if="a.families.length" class="muted small">{{ a.families.length }} models</p>
          </div>
        </RouterLink>
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
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.app {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  padding: 16px;
  transition: border-color 0.15s ease;
}
.app:hover {
  border-color: var(--zen-accent);
}
.icon {
  font-size: 26px;
  width: 44px;
  height: 44px;
  flex: none;
  display: grid;
  place-items: center;
  border-radius: var(--r-lg);
  color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 14%, transparent);
}
.cover {
  width: 64px;
  height: 64px;
  flex: none;
  object-fit: cover;
  border-radius: var(--r-lg);
}
.name {
  font-weight: 650;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.desc {
  margin: 4px 0 0;
  font-size: 12px;
  line-height: 1.45;
}
.warn {
  border-color: var(--zen-warn);
}
.small {
  font-size: 11px;
}
</style>
