<script setup lang="ts">
// Tools: utilities for LoRA files and datasets (tools.ts), laid out like Apps.
import TemplateCard from '@/components/TemplateCard.vue'
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
        <TemplateCard v-for="t in list" :key="t.id" :to="`/tools/${t.id}`" :title="t.name" :description="t.description" :icon="t.icon" :image="t.image" :accent="t.accent" />
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
</style>
