<script setup lang="ts">
// What the LoRA spends its steps on: a flow model learns layout and pose at heavy noise
// (composition), shapes and features in the middle (form), faces, texture and fine detail at
// light noise (detail). One slider leans training toward either end (it sets the timestep shift),
// and bars show the share of steps in each band: as set, and the model's own schedule.
import { ZenSlider } from '@nynxz/zenkit-ui'
import { computed } from 'vue'
import type { NoiseProfile, TrainSpec } from '@/api/engine'

const props = defineProps<{ objective: TrainSpec['objective']; profile: NoiseProfile }>()

const BASE = 9 // focus -1..1 -> shift 9..1/9: an even split at 0, ~82% in one end band at the ends
const shiftOf = (f: number) => Number((BASE ** -f).toFixed(2))
const focusOf = (shift: number) => -Math.log(shift) / Math.log(BASE)

// Acklam's inverse normal CDF: logit-normal timesteps from evenly spaced quantiles
function ndtri(p: number): number {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
  const lo = 0.02425
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p))
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
  }
  if (p > 1 - lo) return -ndtri(1 - p)
  const q = p - 0.5
  const r = q * q
  return ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
}

/** [composition, form, detail] shares of the steps (σ above 2/3, between, below 1/3) */
function bands(o: Pick<TrainSpec['objective'], 'timesteps' | 'logit_mean' | 'logit_std' | 'uniform_mix' | 'shift'>): number[] {
  const table = props.profile.sigmas ?? []
  const sigmaAt = (t: number) => {
    // t: noise fraction. A shift reshapes it as training does (objective.sigmas_for); without one,
    // the model's own schedule at that point (its table runs from full noise to clean)
    if (o.shift > 0) return (o.shift * t) / (1 + (o.shift - 1) * t)
    const x = (1 - t) * (table.length - 1)
    const i = Math.min(table.length - 2, Math.floor(x))
    return table[i]! + (table[i + 1]! - table[i]!) * (x - i)
  }
  const N = 1000
  const out = [0, 0, 0]
  const add = (t: number, w: number) => {
    const s = sigmaAt(Math.min(1, Math.max(0, t)))
    out[s > 2 / 3 ? 0 : s < 1 / 3 ? 2 : 1]! += w
  }
  const mix = o.timesteps === 'logit_normal' ? (o.uniform_mix ?? 0) : 1
  for (let i = 0; i < N; i++) {
    const u = (i + 0.5) / N
    add(u, mix / N)
    if (mix < 1) add(1 / (1 + Math.exp(-(o.logit_mean + o.logit_std * ndtri(u)))), (1 - mix) / N)
  }
  return out
}

const own = computed(() => bands({ ...props.objective, timesteps: 'uniform', shift: 0 }))
const now = computed(() => bands(props.objective))
// the slider's position for the model's own schedule: the shift that gives its detail share
const ownFocus = computed(() => focusOf(Math.max(1e-3, (1 / Math.max(own.value[2]!, 1e-3) - 1) / 2)))
const focus = computed({
  get: () => (props.objective.shift > 0 ? focusOf(props.objective.shift) : ownFocus.value),
  set: (f: number) => (props.objective.shift = shiftOf(f)),
})
const pct = (x: number) => `${Math.round(x * 100)}%`
const NAMES = ['Composition', 'Form', 'Detail']
</script>

<template>
  <div class="focus">
    <div class="ends small muted"><span>Composition</span><span>Detail</span></div>
    <ZenSlider v-model="focus" :min="-1" :max="1" :step="0.05" />
    <div class="mark" :style="{ left: `${((ownFocus + 1) / 2) * 100}%` }" title="The model's own" />
    <div v-for="(row, k) in [{ name: 'This run', b: now }, { name: profile.model ? `${profile.model}'s own` : `Model's own`, b: own }]" :key="k" class="bar-row">
      <span class="small muted who">{{ row.name }}</span>
      <div class="bar">
        <span v-for="(share, i) in row.b" :key="i" class="seg" :class="`b${i}`" :style="{ flexGrow: share }" :title="`${NAMES[i]}: ${pct(share)} of steps`">
          <template v-if="share >= 0.12">{{ NAMES[i] }} {{ pct(share) }}</template>
        </span>
      </div>
    </div>
    <div class="foot small muted">
      <span>{{ objective.shift > 0 ? `shift ${objective.shift}` : "the model's own schedule" }}</span>
      <button v-if="objective.shift > 0" class="link" @click="objective.shift = 0">Use the model's own</button>
    </div>
  </div>
</template>

<style scoped>
.focus {
  position: relative;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ends {
  display: flex;
  justify-content: space-between;
}
.mark {
  position: absolute;
  top: 22px;
  width: 2px;
  height: 14px;
  margin-left: -1px;
  background: var(--zen-muted);
  border-radius: 1px;
  pointer-events: none;
}
.bar-row {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  align-items: center;
  gap: 8px;
}
.who {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar {
  display: flex;
  height: 20px;
  border-radius: var(--r-sm);
  overflow: hidden;
  gap: 2px;
}
.seg {
  flex-basis: 0;
  min-width: 2px;
  display: flex;
  align-items: center;
  padding: 0 6px;
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  color: var(--zen-text);
}
/* one hue, light to strong: the bands run from heavy noise to light */
.b0 {
  background: color-mix(in srgb, var(--zen-accent) 22%, var(--zen-surface-2));
}
.b1 {
  background: color-mix(in srgb, var(--zen-accent) 42%, var(--zen-surface-2));
}
.b2 {
  background: color-mix(in srgb, var(--zen-accent) 70%, var(--zen-surface-2));
}
.foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
}
</style>
