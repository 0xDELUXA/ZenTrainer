<script setup lang="ts">
import { computed } from 'vue'
import type { RunStatus } from '@/api/engine'

const props = defineProps<{ status: RunStatus }>()
// Status colour always comes with an icon and the word, never colour alone.
const look = computed(
  () =>
    ({
      queued: { icon: 'mdi mdi-clock-outline', tone: 'info' },
      loading: { icon: 'mdi mdi-database-arrow-down-outline', tone: 'info' },
      training: { icon: 'mdi mdi-progress-helper', tone: 'ok' },
      paused: { icon: 'mdi mdi-pause-circle-outline', tone: 'warn' },
      stopped: { icon: 'mdi mdi-stop-circle-outline', tone: 'warn' },
      done: { icon: 'mdi mdi-check-circle-outline', tone: 'ok' },
      interrupted: { icon: 'mdi mdi-stop-circle-outline', tone: 'warn' },
      error: { icon: 'mdi mdi-alert-circle-outline', tone: 'danger' },
      rejected: { icon: 'mdi mdi-alert-circle-outline', tone: 'danger' },
    })[props.status] ?? { icon: 'mdi mdi-help-circle-outline', tone: 'info' },
)
</script>

<template>
  <span class="pill" :class="look.tone"><i :class="look.icon" /> {{ status }}</span>
</template>

<style scoped>
.pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--zen-text);
}
.pill i {
  font-size: 14px;
}
.info i {
  color: var(--zen-info);
}
.ok i {
  color: var(--zen-ok);
}
.warn i {
  color: var(--zen-warn);
}
.danger i {
  color: var(--zen-danger);
}
</style>
