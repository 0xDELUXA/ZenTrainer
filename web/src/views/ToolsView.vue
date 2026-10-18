<script setup lang="ts">
// Tools: utilities for LoRA files and datasets (tools.ts), laid out like Apps.
import { TOOL_CATEGORIES, TOOLS, type Tool } from '@/tools'

const byCategory = Object.entries(TOOL_CATEGORIES).map(([k, label]) => [label, TOOLS.filter((t) => t.category === k)] as [string, Tool[]])
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Tools</h1>
      </div>
    </div>
    <section v-for="[label, list] in byCategory" :key="label" class="cat">
      <div class="eyebrow">{{ label }}</div>
      <div class="grid">
        <RouterLink v-for="t in list" :key="t.id" :to="`/tools/${t.id}`" class="tool card">
          <i :class="t.icon" class="icon" />
          <div>
            <div class="name">{{ t.name }}</div>
            <p class="muted desc">{{ t.description }}</p>
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
.tool {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  padding: 16px;
  transition: border-color 0.15s ease;
}
.tool:hover {
  border-color: var(--zen-accent);
}
.icon {
  font-size: 24px;
  width: 44px;
  height: 44px;
  flex: none;
  display: grid;
  place-items: center;
  border-radius: var(--r-lg);
  color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 14%, transparent);
}
.name {
  font-weight: 650;
  font-size: 14px;
}
.desc {
  margin: 4px 0 0;
  font-size: 12px;
  line-height: 1.45;
}
</style>
