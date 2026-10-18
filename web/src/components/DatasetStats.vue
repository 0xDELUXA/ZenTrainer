<script setup lang="ts">
// A dataset's health: a tile per measure (the real number, the change since the last snapshot, a
// sparkline of its score over the dataset's history) beside the pentagon of the five scores. A
// snapshot is kept whenever the dataset changed, so the history shows what each edit did.
import { computed, onMounted, ref } from 'vue'
import { engine, type DatasetStats } from '@/api/engine'
import HealthPentagon from './HealthPentagon.vue'

const props = defineProps<{ name: string }>()
const emit = defineEmits<{ loaded: [stats: DatasetStats] }>()
const stats = ref<DatasetStats | null>(null)
const busy = ref(false)
const error = ref('')

async function load() {
  busy.value = true
  error.value = ''
  try {
    stats.value = await engine.datasetStats(props.name)
    emit('loaded', stats.value)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
onMounted(load)
defineExpose({ load })

const m = computed(() => (stats.value?.current.metrics ?? {}) as Record<string, number | null>)
const scores = computed(() => stats.value?.current.scores ?? {})
const pct = (x: number | null | undefined) => `${Math.round((x ?? 0) * 100)}%`

interface Tile {
  key: string
  label: string
  value: string
  sub: string
}
const tiles = computed<Tile[]>(() => {
  const v = m.value
  if (!stats.value) return []
  const out: Tile[] = [
    { key: 'captions', label: 'Captions', value: `${v.captioned}/${v.pictures}`, sub: 'have a default caption' },
    {
      key: 'consistency',
      label: 'Consistency',
      value: `${v.caption_words_median} words`,
      sub: `median · spread ${pct(v.caption_words_spread)} · ${v.caption_identical} identical · ${v.caption_short} under 5 words`,
    },
    {
      key: 'quality',
      label: 'Quality',
      value: `${v.image_issues} flagged`,
      sub: [
        v.below_target && `${v.below_target} small`,
        v.blurry && `${v.blurry} blurry`,
        v.bars && `${v.bars} with bars`,
        v.flat && `${v.flat} flat`,
        v.greyscale && `${v.greyscale} greyscale`,
      ]
        .filter(Boolean)
        .join(' · ') || `all sharp and ${v.short_side_median}px+ (median short side)`,
    },
    {
      key: 'exposure',
      label: 'Exposure',
      value: pct(v.brightness_median),
      sub: `median brightness · spread ${pct(v.brightness_spread)}${v.too_dark ? ` · ${v.too_dark} dark` : ''}${v.blown_out ? ` · ${v.blown_out} blown out` : ''}`,
    },
    { key: 'uniqueness', label: 'Uniqueness', value: `${v.near_duplicates}`, sub: v.near_duplicates ? 'pictures with a near-identical twin' : 'no near-duplicates' },
  ]
  if (scores.value.faces != null)
    out.push({
      key: 'faces',
      label: 'Faces',
      value: v.face_one_identity ? `alike ${v.face_likeness_median ?? '–'}` : 'mixed people',
      sub: `${v.face_one_identity ? 'faces agree (each vs the others; one person ≳ 0.5)' : `faces don't agree (median ${v.face_likeness_median ?? '–'}): likeness for runs is off`} · ${v.face_issues} issue${v.face_issues === 1 ? '' : 's'} · ${v.face_coverage}/${v.pictures} with a face`,
    })
  else
    out.push({
      key: 'balance',
      label: 'Balance',
      value: pct(v.aspect_top_share),
      sub: 'share in the most common aspect ratio · run Faces for a likeness check',
    })
  return out
})

// a score's history and its change since the snapshot before this one
const series = (key: string): { at: number; v: number }[] =>
  (stats.value?.history ?? []).flatMap((h) => (h.scores[key] != null ? [{ at: h.at, v: h.scores[key]! }] : []))
function delta(key: string): number | null {
  const s = series(key)
  return s.length >= 2 ? s[s.length - 1]!.v - s[s.length - 2]!.v : null
}
const W = 64
const H = 26
function spark(key: string) {
  const s = series(key)
  if (!s.length) return { line: '', dots: [] as { x: number; y: number; tip: string }[] }
  const x = (i: number) => (s.length === 1 ? W / 2 : 4 + (i * (W - 8)) / (s.length - 1))
  const y = (v: number) => H - 3 - v * (H - 6)
  const dots = s.map((p, i) => ({ x: x(i), y: y(p.v), tip: `${new Date(p.at * 1000).toLocaleString()}: ${Math.round(p.v * 100)}` }))
  return { line: dots.map((d) => `${d.x},${d.y}`).join(' '), dots }
}
</script>

<template>
  <section class="dstats card">
    <div class="pent-col">
      <HealthPentagon v-if="stats" :scores="scores" :size="190" labels class="big-pent" />
      <div v-else class="pent-empty muted small">{{ busy ? 'Measuring…' : error || 'No stats yet' }}</div>
      <div class="muted small foot">
        {{ stats ? `${stats.history.length} snapshot${stats.history.length === 1 ? '' : 's'} · one is kept each time the dataset changes` : '' }}
        <button class="link" :disabled="busy" @click="load">{{ busy ? 'measuring…' : 'measure again' }}</button>
      </div>
    </div>
    <div class="tiles">
      <div v-for="t in tiles" :key="t.key" class="tile">
        <div class="t-head">
          <span class="t-label">{{ t.label }}</span>
          <span v-if="scores[t.key] != null" class="t-score mono" :title="`Score ${Math.round(scores[t.key]! * 100)} of 100`">{{ Math.round(scores[t.key]! * 100) }}</span>
        </div>
        <div class="t-value">{{ t.value }}</div>
        <div class="t-sub muted">{{ t.sub }}</div>
        <div v-if="scores[t.key] != null" class="t-foot">
          <svg :width="W" :height="H" class="spark" role="img" :aria-label="`${t.label} score over time`">
            <polyline v-if="series(t.key).length > 1" :points="spark(t.key).line" class="spark-line" />
            <circle v-for="(d, i) in spark(t.key).dots" :key="i" :cx="d.x" :cy="d.y" :r="i === spark(t.key).dots.length - 1 ? 3 : 2" class="spark-dot">
              <title>{{ d.tip }}</title>
            </circle>
          </svg>
          <span v-if="delta(t.key) != null && Math.round(delta(t.key)! * 100) !== 0" class="t-delta small">
            <i :class="delta(t.key)! > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" />{{ Math.abs(Math.round(delta(t.key)! * 100)) }} since last
          </span>
          <span v-else class="t-delta small muted">{{ series(t.key).length > 1 ? 'no change' : 'first look' }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.dstats {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 16px;
  padding: 14px;
  align-items: start;
}
@media (max-width: 900px) {
  .dstats {
    grid-template-columns: 1fr;
  }
}
.pent-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.pent-empty {
  height: 210px;
  display: grid;
  place-items: center;
}
.foot {
  text-align: center;
  line-height: 1.5;
}
.link {
  border: none;
  background: none;
  color: var(--zen-accent);
  font: inherit;
  cursor: pointer;
  padding: 0;
}
.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 10px;
}
.tile {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 10px 12px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.t-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}
.t-label {
  font-size: 12px;
  color: var(--zen-muted);
}
.t-score {
  font-size: 12px;
  color: var(--zen-text);
}
.t-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--zen-text);
}
.t-sub {
  font-size: 11px;
  line-height: 1.4;
  min-height: 2.8em;
}
.t-foot {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
}
.spark {
  display: block;
  overflow: visible;
}
.spark-line {
  fill: none;
  stroke: var(--zen-accent);
  stroke-width: 2;
  stroke-linejoin: round;
}
.spark-dot {
  fill: var(--zen-accent);
  stroke: var(--zen-surface);
  stroke-width: 1;
}
.t-delta {
  color: var(--zen-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
.spark {
  flex: none;
}
.big-pent {
  margin: 18px 40px 10px; /* room for the axis labels outside the shape */
}
</style>
