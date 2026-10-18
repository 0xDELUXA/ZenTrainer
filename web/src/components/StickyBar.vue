<script setup lang="ts">
// A page's toolbar that sticks to the top as the page scrolls. It sits on a strip of the page's
// background from 10px above it down to its middle: nothing peeks over the top, and the strip's
// edge hides behind the bar, so below it the content just passes under (card borders meet the
// bar's rounded corners; no line anywhere). Stuck, the bar casts a shadow over what's under it.
// The negative margin cancels the strip's padding, so at rest the page lays out as without it.
//   <StickyBar><div class="toolbar card">…</div></StickyBar>
import { onBeforeUnmount, onMounted, ref } from 'vue'

const wrap = ref<HTMLElement | null>(null)
const stuck = ref(false)
// stuck when its top has reached the top of the scroll area (at rest there's a page header above)
function check() {
  const el = wrap.value
  if (!el) return
  const top = el.getBoundingClientRect().top
  const area = (el.closest('.viewport, .zen-scroll-viewport') as HTMLElement | null)?.getBoundingClientRect().top ?? 0
  stuck.value = top <= area + 1
}
onMounted(() => {
  document.addEventListener('scroll', check, { capture: true, passive: true })
  check()
})
onBeforeUnmount(() => document.removeEventListener('scroll', check, { capture: true }))
</script>

<template>
  <div ref="wrap" class="sticky-bar" :class="{ stuck }"><slot /></div>
</template>

<style scoped>
.sticky-bar {
  position: sticky;
  top: 0;
  z-index: 5;
  margin-top: -10px;
  padding-top: 10px;
  background: linear-gradient(to bottom, var(--zen-bg) calc(50% + 5px), transparent 0);
}
.sticky-bar > :slotted(*) {
  /* opaque, so content doesn't show through once it's stuck */
  background: color-mix(in srgb, var(--zen-surface) 70%, var(--zen-bg));
  transition: box-shadow 0.2s ease;
}
.sticky-bar.stuck > :slotted(*) {
  box-shadow: 0 10px 22px -6px rgba(0, 0, 0, 0.75);
}
</style>
