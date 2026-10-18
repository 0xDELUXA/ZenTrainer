<script setup lang="ts">
import { computed } from 'vue'
import type { RunStatus } from '@/api/engine'

// runs and caption jobs share it: running/undone are caption job states
const props = defineProps<{ status: RunStatus | 'running' | 'undone'; label?: string }>()
// Status colour always comes with an icon and the word, never colour alone.
const look = computed(
  () =>
    ({
      queued: { icon: 'mdi mdi-clock-outline', tone: 'info' },
      loading: { icon: 'mdi mdi-database-arrow-down-outline', tone: 'info' },
      training: { icon: 'mdi mdi-progress-helper', tone: 'ok' },
      running: { icon: 'mdi mdi-progress-helper', tone: 'ok' },
      undone: { icon: 'mdi mdi-undo-variant', tone: 'info' },
      paused: { icon: 'mdi mdi-pause-circle-outline', tone: 'warn' },
      sampling: { icon: 'mdi mdi-image-refresh-outline', tone: 'info' },
      stopped: { icon: 'mdi mdi-stop-circle-outline', tone: 'warn' },
      done: { icon: 'mdi mdi-check-circle-outline', tone: 'ok' },
      interrupted: { icon: 'mdi mdi-stop-circle-outline', tone: 'warn' },
      error: { icon: 'mdi mdi-alert-circle-outline', tone: 'danger' },
      rejected: { icon: 'mdi mdi-alert-circle-outline', tone: 'danger' },
    })[props.status] ?? { icon: 'mdi mdi-help-circle-outline', tone: 'info' },
)
</script>

<template>
  <span class="pill" :class="look.tone"><i :class="look.icon" /> {{ label ?? status }}</span>
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
