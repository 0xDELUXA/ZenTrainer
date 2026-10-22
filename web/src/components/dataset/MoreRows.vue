<script setup lang="ts">
// The end of a paged list: asks for more as it scrolls near, and again while still in view after
// a page lands (an observer only fires on a change).
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

const emit = defineEmits<{ more: [] }>()
const el = ref<HTMLElement>()
let io: IntersectionObserver | undefined
onMounted(() => {
  io = new IntersectionObserver(
    (entries) => {
      if (!entries.some((e) => e.isIntersecting)) return
      emit('more')
      nextTick(() => {
        if (!el.value || !io) return
        io.unobserve(el.value)
        io.observe(el.value)
      })
    },
    { rootMargin: '1200px' },
  )
  if (el.value) io.observe(el.value)
})
onBeforeUnmount(() => io?.disconnect())
</script>

<template>
  <div ref="el" class="more-rows" aria-hidden="true" />
</template>

<style scoped>
.more-rows {
  grid-column: 1 / -1;
  height: 1px;
}
</style>
