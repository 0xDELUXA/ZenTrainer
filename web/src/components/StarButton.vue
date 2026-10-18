<script setup lang="ts">
// Star a model family. Safe inside a dropdown option: the click doesn't pick the option.
import { isStarred, toggleStar } from '@/favorites'

const props = defineProps<{ id: string; name?: string }>()
</script>

<template>
  <button
    type="button"
    class="star"
    :class="{ on: isStarred(props.id) }"
    :aria-pressed="isStarred(props.id)"
    :title="isStarred(props.id) ? `Unstar ${name ?? 'this model'}` : `Star ${name ?? 'this model'}: it goes to the top of model lists`"
    @click.stop.prevent="toggleStar(props.id)"
    @mousedown.stop
    @pointerdown.stop
  >
    <i :class="isStarred(props.id) ? 'mdi mdi-star' : 'mdi mdi-star-outline'" />
  </button>
</template>

<style scoped>
.star {
  display: inline-grid;
  place-items: center;
  width: 22px;
  height: 22px;
  flex: none;
  padding: 0;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  opacity: 0.55;
  cursor: pointer;
  font-size: 15px;
}
.star:hover,
.star:focus-visible {
  opacity: 1;
  color: var(--zen-text);
}
.star.on {
  opacity: 1;
  color: var(--zen-warn);
}
</style>
