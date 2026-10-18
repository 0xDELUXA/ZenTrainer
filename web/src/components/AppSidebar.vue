<script setup lang="ts">
// The app rail. Every row keeps its icon in a fixed 44px column, so folding only
// changes the width: nothing moves, labels just fade. The right edge (highlighted on
// hover) or "[" folds it. Themes live in Settings. Colours come from the pack's --sidebar-* tokens when it
// has them, else from the general surface tokens.
import { ZenLotus } from '@nynxz/zenkit-ui'
import ScrollArea from './ScrollArea.vue'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { activeRuns, latest, lossTrail } from '@/runs'
import { activeCaptions, startCaptionSync } from '@/captions'
import { settings } from '@/settings'
import { comfyBehind, comfyOld, engineStale, restartEngine, restarting, startStatusSync } from '@/status'
import { activeDownloads, fmtBytes, startDownloadSync } from '@/downloads'
import { disks, gb, memLatest, startSystemSync } from '@/system'
import { drawerOpen, toggleDrawer, unseenErrors } from '@/logs'

defineProps<{ collapsed: boolean; mobileOpen: boolean }>()
const emit = defineEmits<{ toggle: []; navigate: [] }>()

// the mark: blooms when clicked, breathes while something trains
const lotus = ref<InstanceType<typeof ZenLotus> | null>(null)
const simple = computed(() => settings.mode === 'simple')
const training = computed(() => activeRuns.value.some((r) => r.status === 'training'))
const route = useRoute()
const router = useRouter()

interface NavItem {
  to: string
  icon: string
  label: string
  count?: () => number
  /** behind Settings → Experimental pages */
  experimental?: boolean
}
// six places: no group headings needed
const items: NavItem[] = [
  // everything that runs: training (and its queue) and caption jobs
  { to: '/jobs', icon: 'mdi mdi-chart-timeline-variant', label: 'Jobs', count: () => activeRuns.value.length + activeCaptions.value.length },
  { to: '/datasets', icon: 'mdi mdi-image-multiple-outline', label: 'Datasets' },
  { to: '/models', icon: 'mdi mdi-cube-outline', label: 'Models' },
  { to: '/apps', icon: 'mdi mdi-view-grid-outline', label: 'Apps', experimental: true },
  { to: '/tools', icon: 'mdi mdi-toolbox-outline', label: 'Tools', experimental: true },
  { to: '/gallery', icon: 'mdi mdi-image-album', label: 'Gallery', experimental: true },
]
const isActive = (to: string) =>
  to === '/jobs' ? route.path.startsWith('/jobs') || (route.path.startsWith('/runs/') && route.path !== '/runs/new') : route.path === to || route.path.startsWith(to + '/')

// ---- runs -------------------------------------------------------------------------
const live = computed(() =>
  activeRuns.value.map((r) => {
    const m = latest[r.run_id]
    const frac = m?.total ? m.step / m.total : 0
    return { ...r, step: m?.step ?? 0, total: m?.total ?? 0, it_s: m?.it_s, frac, spark: spark(lossTrail[r.run_id]) }
  }),
)

function spark(values: number[] | undefined): string {
  if (!values || values.length < 2) return ''
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  return values
    .map(
      (v, i) =>
        `${i ? 'L' : 'M'}${((i / (values.length - 1)) * 100).toFixed(1)},${(22 - ((v - lo) / (hi - lo || 1)) * 20).toFixed(1)}`,
    )
    .join('')
}

// ---- memory --------------------------------------------------------------------------
const meters = computed(() => {
  const m = memLatest.value
  if (!m) return []
  const out: { label: string; used: number; total: number; text?: string }[] = []
  if (m.vram_total) out.push({ label: 'VRAM', used: m.vram_used ?? 0, total: m.vram_total })
  if (m.ram_total) out.push({ label: 'RAM', used: m.ram_used ?? 0, total: m.ram_total })
  // the disk outputs land on: checkpoints and resume states fill it fast
  const d = disks.value[0]
  if (d) out.push({ label: 'Disk', used: d.used, total: d.total, text: `${gb(d.free)} GB free` })
  return out
})

// ---- downloads: the one in progress, and how many wait behind it -----------------------
const dl = computed(() => {
  const list = activeDownloads.value
  const cur = list.find((d) => d.status === 'downloading') ?? list[0]
  if (!cur) return null
  return { name: cur.name, pct: cur.total ? cur.done / cur.total : 0, done: cur.done, total: cur.total, speed: cur.speed, more: list.length - 1 }
})

// ---- caption jobs: the one running (or first waiting), and how many more ---------------
const cap = computed(() => {
  const list = activeCaptions.value
  const cur = list.find((j) => j.status === 'running') ?? list[0]
  if (!cur) return null
  const total = cur.total || cur.preview || 0
  const meta = cur.status === 'queued' ? 'waiting' : cur.message.startsWith('loading') ? 'loading' : `${cur.done}/${total}`
  return { job: cur, pct: total ? cur.done / total : 0, meta, more: list.length - 1 }
})

// ---- housekeeping: restart to load an update, or ComfyUI has one -----------------------
const trainingNow = computed(() => activeRuns.value.some((r) => r.status === 'training' || r.status === 'loading'))
async function onRestart() {
  if (trainingNow.value) return
  if (!confirm('Restart ComfyUI to load the updated engine? It takes about a minute.')) return
  await restartEngine().catch((e) => alert(String((e as Error).message ?? e)))
}

// ---- folding & keys ------------------------------------------------------------------
function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (e.metaKey || e.ctrlKey || e.altKey || t.closest('input, textarea, select, [contenteditable]')) return
  if (e.key === '[') emit('toggle')
  else if (e.key === 'n') router.push(simple.value ? '/simple' : '/runs/new')
}

onMounted(() => {
  startSystemSync()
  startDownloadSync()
  startCaptionSync()
  startStatusSync()
  addEventListener('keydown', onKey)
})
onUnmounted(() => {
  removeEventListener('keydown', onKey)
})
</script>

<template>
  <aside class="rail" :class="{ collapsed: collapsed && !mobileOpen, open: mobileOpen }" aria-label="Sidebar">
    <button
      class="edge"
      :title="collapsed ? 'Expand sidebar ([)' : 'Fold sidebar ([)'"
      :aria-label="collapsed ? 'Expand sidebar' : 'Fold sidebar'"
      :aria-expanded="!collapsed"
      @click="emit('toggle')"
    />
    <RouterLink to="/jobs" class="row brand" :title="training ? 'Training…' : undefined" @click="lotus?.bloom(); emit('navigate')">
      <span class="ic"><ZenLotus ref="lotus" class="lotus" :breathing="training" /></span>
      <span class="lbl">ZenTrainer</span>
    </RouterLink>

    <RouterLink :to="simple ? '/simple' : '/runs/new'" class="row cta" :title="`${simple ? 'Train a LoRA' : 'New run'} (N)`" @click="emit('navigate')">
      <span class="ic"><i class="mdi mdi-plus" /></span>
      <span class="lbl">{{ simple ? 'Train a LoRA' : 'New run' }}</span>
      <kbd class="lbl">N</kbd>
    </RouterLink>
    <ScrollArea class="nav-scroll"><nav>
      <RouterLink
        v-for="item in items.filter((i) => !i.experimental || settings.experimental)"
        :key="item.to"
        :to="item.to"
        class="row item"
        :class="{ active: isActive(item.to) }"
        :title="collapsed ? item.label : undefined"
        @click="emit('navigate')"
      >
        <span class="ic">
          <i :class="item.icon" />
          <span v-if="item.count?.()" class="dot-count" />
        </span>
        <span class="lbl">{{ item.label }}</span>
        <span v-if="item.count?.()" class="lbl count">{{ item.count() }}</span>
      </RouterLink>
    </nav></ScrollArea>

    <!-- what's training, above the memory meters: one click to it -->
    <RouterLink
      v-for="r in live"
      :key="r.run_id"
      :to="simple ? `/simple/run/${r.run_id}` : `/runs/${r.run_id}`"
      class="live"
      :title="`${r.name}: ${r.status} ${Math.round(r.frac * 100)}%`"
      @click="emit('navigate')"
    >
      <span class="ic">
        <svg viewBox="0 0 36 36" class="ring" :class="r.status">
          <circle cx="18" cy="18" r="15" class="track" />
          <circle cx="18" cy="18" r="15" class="arc" :stroke-dasharray="`${r.frac * 94.25} 94.25`" />
        </svg>
        <span class="pct">{{ Math.round(r.frac * 100) }}</span>
      </span>
      <span class="lbl live-body">
        <span class="live-name ellipsis">{{ r.name }}</span>
        <svg v-if="r.spark" viewBox="0 0 100 24" preserveAspectRatio="none" class="spark"><path :d="r.spark" /></svg>
        <span class="live-meta">
          {{ r.status === 'training' ? `${r.step}/${r.total} · ${r.it_s?.toFixed(2) ?? '–'} it/s` : r.status }}
        </span>
      </span>
    </RouterLink>


    <RouterLink
      v-if="cap"
      :to="`/jobs/caption/${cap.job.job_id}`"
      class="row mem"
      :title="cap.job.message || `${cap.job.preview ? 'Trying captions on' : 'Captioning'} ${cap.job.dataset}${cap.job.file ? `: ${cap.job.file}` : ''}`"
      @click="emit('navigate')"
    >
      <span class="ic"><i class="mdi mdi-creation-outline" /></span>
      <span class="lbl mem-body">
        <span class="mem-row">
          <span class="mem-head"><span class="ellipsis">{{ cap.job.preview ? 'Try' : 'Caption' }} {{ cap.job.dataset }}</span><span class="mono">{{ cap.meta }}{{ cap.more ? ` +${cap.more}` : '' }}</span></span>
          <span class="meter"><span :style="{ width: `${cap.pct * 100}%` }" /></span>
        </span>
      </span>
    </RouterLink>

    <div v-if="dl" class="row mem" :title="`${dl.name}: ${fmtBytes(dl.done)} of ${fmtBytes(dl.total)} · ${fmtBytes(dl.speed)}/s`">
      <span class="ic"><i class="mdi mdi-download" /></span>
      <span class="lbl mem-body">
        <span class="mem-row">
          <span class="mem-head"><span class="ellipsis">{{ dl.name }}</span><span class="mono">{{ Math.floor(dl.pct * 100) }}%{{ dl.more ? ` +${dl.more}` : '' }}</span></span>
          <span class="meter"><span :style="{ width: `${dl.pct * 100}%` }" /></span>
        </span>
      </span>
    </div>

    <div v-if="comfyOld" class="row mem status-row warn-row" :title="comfyOld">
      <span class="ic"><i class="mdi mdi-alert-outline" /></span>
      <span class="lbl status-lbl">Update ComfyUI for ZenTrainer</span>
    </div>
    <button
      v-else-if="engineStale"
      class="row mem status-row"
      :disabled="trainingNow || restarting"
      :title="trainingNow ? 'Updated engine code is waiting: restart once nothing is training' : 'The engine’s code changed since it started: restart ComfyUI to load it'"
      @click="onRestart"
    >
      <span class="ic"><i :class="restarting ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-restart'" /></span>
      <span class="lbl status-lbl">{{ restarting ? 'Restarting…' : trainingNow ? 'Update waiting (after the run)' : 'Restart to load updates' }}</span>
    </button>
    <RouterLink v-else-if="comfyBehind" to="/settings" class="row mem status-row" :title="`ComfyUI is ${comfyBehind} commits behind upstream`" @click="emit('navigate')">
      <span class="ic"><i class="mdi mdi-arrow-up-circle-outline" /></span>
      <span class="lbl status-lbl">ComfyUI update available</span>
    </RouterLink>

    <RouterLink v-if="meters.length" to="/system" class="row mem" :class="{ active: isActive('/system') }" title="System memory" @click="emit('navigate')">
      <span class="ic"><i class="mdi mdi-memory" /></span>
      <span class="lbl mem-body">
        <span v-for="m in meters" :key="m.label" class="mem-row">
          <span class="mem-head"><span>{{ m.label }}</span><span class="mono">{{ 'text' in m ? m.text : `${gb(m.used)}/${gb(m.total)} GB` }}</span></span>
          <span class="meter"><span :style="{ width: `${(m.used / m.total) * 100}%` }" :class="{ hot: m.used / m.total > 0.9 }" /></span>
        </span>
      </span>
    </RouterLink>

    <div class="foot">
      <button
        class="row item log-row"
        :class="{ active: drawerOpen }"
        :title="collapsed ? 'Log (Ctrl+`)' : 'ComfyUI log (Ctrl+`)'"
        @click="toggleDrawer()"
      >
        <span class="ic">
          <i class="mdi mdi-console-line" />
          <span v-if="unseenErrors" class="dot-count err" />
        </span>
        <span class="lbl">Log</span>
        <span v-if="unseenErrors" class="lbl count err" :title="`${unseenErrors} new error${unseenErrors === 1 ? '' : 's'}`">{{ unseenErrors > 99 ? '99+' : unseenErrors }}</span>
        <kbd v-else class="lbl key">Ctrl `</kbd>
      </button>
      <RouterLink
        to="/settings"
        class="row item"
        :class="{ active: isActive('/settings') }"
        :title="collapsed ? 'Settings' : undefined"
        @click="emit('navigate')"
      >
        <span class="ic"><i class="mdi mdi-cog-outline" /></span>
        <span class="lbl">Settings</span>
      </RouterLink>
    </div>
  </aside>
</template>

<style scoped>
.rail {
  /* pack sidebar tokens first, general surfaces as fallback */
  --sb-bg: var(--sidebar, color-mix(in srgb, var(--zen-surface) 70%, var(--zen-bg)));
  --sb-fg: var(--sidebar-foreground, var(--zen-text));
  --sb-muted: var(--zen-muted);
  --sb-primary: var(--sidebar-primary, var(--zen-accent));
  --sb-primary-fg: var(--sidebar-primary-foreground, var(--zen-accent-text));
  --sb-accent: var(--sidebar-accent, color-mix(in srgb, var(--zen-accent) 14%, transparent));
  --sb-accent-fg: var(--sidebar-accent-foreground, var(--zen-text));
  --sb-border: var(--sidebar-border, color-mix(in srgb, var(--zen-border) 70%, transparent));
  --w: 244px;
  --ic: 44px;
  width: var(--w);
  flex: none;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 10px 10px;
  background: var(--sb-bg);
  color: var(--sb-fg);
  border-right: 1px solid var(--sb-border);
  transition: width 0.2s cubic-bezier(0.2, 0, 0, 1);
  overflow: hidden;
  position: relative;
  user-select: none;
}
.rail.collapsed {
  --w: calc(var(--ic) + 20px);
}
/* the fold handle: a strip over the right border that lights up on hover */
.edge {
  position: absolute;
  top: 0;
  right: -1px;
  bottom: 0;
  width: 7px;
  z-index: 5;
  padding: 0;
  border: none;
  background: none;
  cursor: w-resize !important;
}
.collapsed .edge {
  cursor: e-resize !important;
}
.edge::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 2px;
  background: var(--sb-primary);
  opacity: 0;
  transition: opacity 0.15s ease;
}
.edge:hover::after,
.edge:focus-visible::after {
  opacity: 0.7;
}
.rail a,
.rail button {
  cursor: pointer;
}

/* every row: [44px icon column][label]; the icon never moves */
.row,
.live {
  display: flex;
  align-items: center;
  min-height: 36px;
  border-radius: var(--r-md);
  color: inherit;
  flex: none;
}
.ic {
  position: relative;
  width: var(--ic);
  height: 36px;
  flex: none;
  display: grid;
  place-items: center;
  font-size: 18px;
}
.lbl {
  white-space: nowrap;
  transition: opacity 0.15s ease;
}
.collapsed .lbl {
  opacity: 0;
  pointer-events: none;
}
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}

.brand {
  min-height: 40px;
  margin-bottom: 6px;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: -0.01em;
}
.lotus {
  width: 24px;
  height: 24px;
  color: var(--zen-accent);
}
.cta {
  background: var(--sb-primary);
  color: var(--sb-primary-fg);
  font-weight: 600;
  margin-bottom: 8px;
}
.cta:hover {
  filter: brightness(1.07);
}
.cta kbd {
  margin-left: auto;
  margin-right: 10px;
  font: 600 10px var(--zen-mono);
  padding: 1px 5px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, currentColor 16%, transparent);
}

.nav-scroll {
  flex: 1;
  height: auto;
  margin-bottom: 8px;
}
nav {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding-top: 4px;
}
.item {
  position: relative;
  color: var(--sb-muted);
}
.item:hover {
  color: var(--sb-fg);
  background: color-mix(in srgb, var(--sb-accent) 60%, transparent);
}
.item.active {
  color: var(--sb-accent-fg);
  background: var(--sb-accent);
}
.item.active::before {
  content: '';
  position: absolute;
  left: -10px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  background: var(--sb-primary);
}
.item.active .ic i {
  color: var(--sb-primary);
}
.count {
  margin-left: auto;
  margin-right: 10px;
  min-width: 20px;
  height: 18px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  display: grid;
  place-items: center;
  font-size: 11px;
  font-weight: 650;
  background: var(--sb-primary);
  color: var(--sb-primary-fg);
}
.dot-count {
  position: absolute;
  top: 7px;
  right: 10px;
  width: 7px;
  height: 7px;
  border-radius: var(--r-sm);
  background: var(--sb-primary);
  opacity: 0;
  transition: opacity 0.15s ease;
}
.collapsed .dot-count {
  opacity: 1;
}
.log-row {
  width: 100%;
  border: none;
  background: none;
  font: inherit;
  text-align: left;
  padding: 0;
}
.count.err,
.dot-count.err {
  background: var(--zen-danger);
  color: #fff;
}
.key {
  margin-left: auto;
  margin-right: 10px;
  font: 600 10px var(--zen-mono);
  padding: 1px 5px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, currentColor 12%, transparent);
  opacity: 0.8;
}
.collapsed .key {
  opacity: 0;
}

/* live run: progress ring in the icon column, details beside it */
.live {
  min-height: 56px;
  margin-bottom: 4px;
  border: 1px solid var(--sb-border);
  background: color-mix(in srgb, var(--zen-surface) 70%, transparent);
  border-radius: var(--r-lg);
  transition: border-color 0.15s ease, background 0.15s ease;
}
.collapsed .live {
  border-color: transparent;
  background: none;
}
.live:hover {
  border-color: var(--sb-primary);
}
.live .ic {
  height: 54px;
}
.ring {
  width: 32px;
  height: 32px;
  transform: rotate(-90deg);
}
.ring circle {
  fill: none;
  stroke-width: 3;
}
.ring .track {
  stroke: color-mix(in srgb, var(--sb-muted) 25%, transparent);
}
.ring .arc {
  stroke: var(--sb-primary);
  stroke-linecap: round;
  transition: stroke-dasharray 0.4s ease;
}
.ring.paused .arc {
  stroke: var(--zen-warn);
}
.pct {
  position: absolute;
  font: 600 9px var(--zen-mono);
}
.live-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-right: 10px;
}
.live-name {
  font-weight: 650;
  font-size: 12px;
}
.spark {
  width: 100%;
  height: 16px;
}
.spark path {
  fill: none;
  stroke: var(--sb-primary);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
  stroke-linejoin: round;
}
.live-meta {
  font: 10px var(--zen-mono);
  color: var(--sb-muted);
}

/* memory */
.mem {
  color: var(--sb-muted);
  min-height: 52px;
}
.mem:hover,
.mem.active {
  color: var(--sb-fg);
  background: color-mix(in srgb, var(--sb-accent) 60%, transparent);
}
.mem-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 6px 10px 6px 0;
  font-size: 11px;
}
.status-row {
  border: none;
  background: none;
  font: inherit;
  text-align: left;
  width: 100%;
  cursor: pointer;
  color: var(--zen-accent);
}
.status-row:disabled {
  cursor: default;
  color: var(--zen-muted);
}
.status-lbl {
  font-size: 12px;
}
.mem-row {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.mem-head {
  display: flex;
  justify-content: space-between;
  gap: 6px;
}
.meter {
  height: 4px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--sb-muted) 22%, transparent);
  overflow: hidden;
}
.meter span {
  display: block;
  height: 100%;
  background: var(--sb-primary);
  transition: width 0.5s ease;
}
.meter span.hot {
  background: var(--zen-warn);
}

/* footer */
.foot {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-top: 6px;
  margin-top: 4px;
  border-top: 1px solid var(--sb-border);
}
@media (max-width: 720px) {
  .rail,
  .rail.collapsed {
    --w: 272px;
    position: fixed;
    inset: 0 auto 0 0;
    z-index: 50;
    transform: translateX(-100%);
    transition: transform 0.2s ease;
    box-shadow: 0 0 40px rgb(0 0 0 / 0.35);
  }
  .edge {
    display: none;
  }
  .rail.open {
    transform: none;
  }
}
.warn-row {
  color: var(--zen-warn);
  cursor: help;
}
</style>
