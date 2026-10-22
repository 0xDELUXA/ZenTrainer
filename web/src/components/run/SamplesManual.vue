<script setup lang="ts">
// Manual: try any prompt at the current step (or, finished, on the newest checkpoint), apart
// from the run's samples. One row per try: its prompt, then its renders across (no LoRA, then
// each step), with the tries still waiting their turn.
import { ZenButton, ZenCombo, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, ref } from 'vue'
import { viewUrl, type SampleRow } from '@/api/engine'
import { vAutosize } from '@/lib/autosize'
import { useRunContext } from './context'
import { isVideo, rowHasVideo, sampleItem, useSamplesContext } from './samples'

const { detail, status, step, running, sampleBusy, sampleCkpt, newestCkpt, sampleFinished, queuedTries, queueTry, queueFinishedTries } = useRunContext()
const { liveManual, lb, syncing, toggleSync, hoverPlay, hoverPause } = useSamplesContext()

// each try is a card: its no-LoRA render (if asked for) and every step it was rendered at
const manualGroups = computed(() => {
  const by = new Map<number, { index: number; base: SampleRow | null; steps: SampleRow[] }>()
  for (const x of detail.value?.samples ?? []) {
    if (!x.manual) continue
    const g = by.get(x.index) ?? { index: x.index, base: null, steps: [] }
    if (x.base) g.base = x
    else g.steps.push(x)
    by.set(x.index, g)
  }
  for (const g of by.values()) g.steps.sort((a, b) => a.step - b.step)
  return [...by.values()].filter((g) => g.steps.length || g.base).sort((a, b) => b.index - a.index)
})
async function again(g: { index: number; base: SampleRow | null; steps: SampleRow[] }) {
  const x = g.steps.at(-1) ?? g.base
  if (!x) return
  await queueTry({ prompt: x.prompt, width: x.width, height: x.height, frames: x.frames, seed: x.seed, base: false, index: g.index }, x.prompt, g.index)
}
// the cards for tries not landed yet: the first one is live while a manual render runs (even
// one this page didn't queue, e.g. after a reload)
// a prompt as samples render it: [trigger] substituted (engine: callbacks.sample_text), so a waiting
// try can find the row its first render (the no-LoRA one) already started
const asRendered = (p: string) => {
  const t = detail.value?.spec.dataset.trigger ?? ''
  return p.replaceAll('[trigger]', t).trim()
}
const waitingCards = computed(() => {
  const live = liveManual.value
  const phase = live
    ? `${live.base ? 'Without the LoRA' : `At step ${live.step ?? step.value}`} · ${
        live.s_total ? ((live.s_step ?? 0) >= live.s_total ? 'decoding' : `${live.s_step ?? 0}/${live.s_total}`) : 'starting'
      }`
    : ''
  const frac = live?.s_total ? Math.min(1, (live.s_step ?? 0) / live.s_total) : 0
  const cards = queuedTries.value.map((q, i) => ({
    key: `q${q.id}`,
    prompt: asRendered(q.prompt),
    again: q.again != null,
    // a new try whose no-LoRA render landed waits in that row, not in a row of its own
    againIndex: q.again ?? manualGroups.value.find((g) => !g.steps.length && rowOf(g).prompt === asRendered(q.prompt))?.index,
    live: i === 0 && !!live,
    // a finished run's job loads the whole model before its first render
    loading: i === 0 && !live && sampleBusy.value,
    frac: i === 0 ? frac : 0,
    state: i === 0 && live ? phase : i === 0 && sampleBusy.value ? 'Loading the model…' : i === (live ? 1 : 0) ? 'Next' : `Queued · ${i - (live ? 1 : 0)} ahead`,
  }))
  if (live && !cards.length)
    cards.push({ key: 'live', prompt: live.prompt ?? '', again: false, againIndex: undefined, live: true, loading: false, frac, state: phase })
  return cards
})
// "Try a prompt": one render at the current step, apart from the run's samples
const tryPrompt = ref('')
const trySize = ref('')
const tryFrames = ref<number | null>(null)
const tryBase = ref(true)
async function renderTry() {
  // one try per line: several prompts queue at once
  const lines = tryPrompt.value.split('\n').map((l) => l.trim()).filter(Boolean)
  if (!lines.length) return
  const [w, h] = trySize.value ? trySize.value.split('x').map(Number) : [undefined, undefined]
  tryPrompt.value = ''
  if (sampleCkpt.value) {
    // finished: one job renders them all on the newest checkpoint
    queueFinishedTries(lines)
    await sampleFinished(lines.map((prompt) => ({ prompt, width: w, height: h, frames: tryFrames.value ?? undefined, base: tryBase.value })))
    return
  }
  for (const prompt of lines)
    await queueTry({ prompt, width: w, height: h, frames: tryFrames.value ?? undefined, base: tryBase.value }, prompt)
}
const trySizes = computed(() => {
  const sp = detail.value?.spec.samples
  return [
    { value: '', label: sp ? `Run size (${sp.width}×${sp.height})` : 'Run size' },
    ...[[1024, 1024], [832, 1216], [1216, 832], [480, 832], [832, 480], [512, 512]].map(([w, h]) => ({ value: `${w}x${h}`, label: `${w}×${h}` })),
  ]
})
const rowOf = (g: { base: SampleRow | null; steps: SampleRow[] }) => (g.steps[0] ?? g.base)!
const copyText = (t: string) => navigator.clipboard?.writeText(t).catch(() => {})
function openManual(i: number, row?: SampleRow | null) {
  // the viewer pages through every try (its no-LoRA render first, then each step) and opens on
  // the render that was clicked
  const rows = (g: (typeof manualGroups.value)[number]) => [...(g.base ? [g.base] : []), ...g.steps]
  const flat = manualGroups.value.flatMap(rows)
  const start = manualGroups.value.slice(0, i).reduce((n, g) => n + rows(g).length, 0)
  const own = rows(manualGroups.value[i]!)
  const at = start + Math.max(0, row ? own.indexOf(row) : own.length - 1)
  lb.value = { items: flat.map((x) => ({ ...sampleItem(x), label: x.base ? 'Without the LoRA' : `Step ${x.step}` })), index: at }
}
// a tile shaped like its render (pairs side by side: twice as wide), so nothing letterboxes
const shapeOf = (x: SampleRow | null | undefined, pair = false) =>
  x?.width && x?.height ? { aspectRatio: `${x.width * (pair ? 2 : 1)} / ${x.height}` } : undefined
// a Manual tile: 96px high, but a portrait render grows taller so the tile stays at least
// 80px wide and its "Step N" label fits
const tileOf = (x: SampleRow | null | undefined) => {
  const shape = shapeOf(x)
  if (!x?.width || !x?.height) return shape
  const r = x.width / x.height
  return 96 * r < 80 ? { ...shape, height: `${Math.min(170, Math.round(80 / r))}px` } : shape
}
</script>

<template>
  <div v-if="detail" class="manual">
    <div class="try">
      <textarea
        v-model="tryPrompt"
        v-autosize
        rows="1"
        class="try-in"
        :placeholder="`Try a prompt ${running ? 'at this step' : newestCkpt ? `on the step ${newestCkpt.step} checkpoint` : ''}, e.g. [trigger] reading a book in a cafe (one per line for several)`"
        @keydown.ctrl.enter="renderTry"
      />
      <div class="try-opts">
        <ZenCombo v-model="trySize" :items="trySizes" :sorted="false" :searchable="false" size="sm" />
        <ZenNumber v-if="(detail.spec.samples.frames ?? 1) > 1" :model-value="tryFrames ?? detail.spec.samples.frames ?? 1" :min="1" :max="361" :step="17" :precision="0" size="sm" title="Frames" @update:model-value="tryFrames = $event" />
        <label class="base-sw small" title="Also render the same prompt and seed without the LoRA, to compare"><ZenSwitch v-model="tryBase" size="sm" /> Compare with base</label>
        <ZenButton size="sm" variant="primary" icon="mdi mdi-creation-outline" :disabled="(status !== 'training' && !sampleCkpt) || !tryPrompt.trim()" :title="status === 'training' ? 'Queues it (Ctrl+Enter). One prompt per line queues several' : sampleCkpt ? `Renders with the step ${sampleCkpt.step} checkpoint (Ctrl+Enter). One prompt per line for several` : 'Only while it trains, or once it has a checkpoint'" @click="renderTry">Render</ZenButton>
      </div>
    </div>
    <div v-if="!manualGroups.length && !waitingCards.length" class="muted small empty-manual">
      Prompts you try here render with {{ running ? "the LoRA as it is now, at the step it's on" : 'the newest checkpoint' }}. They don't join the run's samples.
    </div>
    <!-- one row per try: its prompt, then its renders across (no LoRA, then each step) -->
    <div class="mrows">
      <div v-for="t in waitingCards.filter((w) => w.againIndex == null)" :key="t.key" class="mrow2">
        <div class="mprompt"><span class="mtext">{{ t.prompt }}</span></div>
        <div class="mcells">
          <div class="tcell pending" :class="{ live: t.live }">
            <img v-if="t.live && liveManual?.preview" :src="liveManual.preview" alt="" />
            <i v-else :class="t.live || t.loading ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-clock-outline'" />
            <div v-if="t.live" class="qbar"><span :style="{ width: `${t.frac * 100}%` }" /></div>
            <span class="clabel">{{ t.state }}</span>
          </div>
        </div>
      </div>
      <div v-for="(g, gi) in manualGroups" :key="g.index" class="mrow2" :data-syncing="syncing === `m${g.index}` || undefined">
        <div class="mprompt">
          <span class="mtext" :title="rowOf(g).prompt">{{ rowOf(g).prompt }}</span>
          <span class="mmeta mono">
            {{ rowOf(g).width }}×{{ rowOf(g).height }}<template v-if="(rowOf(g).frames ?? 1) > 1"> · {{ rowOf(g).frames }}f</template> · seed {{ rowOf(g).seed }}
          </span>
          <span class="macts" :class="{ held: syncing === `m${g.index}` }">
            <button
              v-if="rowHasVideo([g.base, ...g.steps])"
              :class="{ on: syncing === `m${g.index}` }"
              :title="syncing === `m${g.index}` ? 'Stop' : 'Play the whole row together'"
              @click="toggleSync(`m${g.index}`, ($event.currentTarget as HTMLElement).closest('.mrow2')?.querySelector('.mcells'))"
            >
              <i :class="syncing === `m${g.index}` ? 'mdi mdi-stop' : 'mdi mdi-play'" />
            </button>
            <button title="Put this prompt back in the box to change it" @click="tryPrompt = rowOf(g).prompt"><i class="mdi mdi-pencil-outline" /></button>
            <button title="Copy the prompt" @click="copyText(rowOf(g).prompt)"><i class="mdi mdi-content-copy" /></button>
          </span>
        </div>
        <div class="mcells">
          <button v-for="x in [...(g.base ? [g.base] : []), ...g.steps]" :key="x.filename" class="tcell" :style="tileOf(x)" @click="openManual(gi, x)">
            <video v-if="isVideo(x)" :src="viewUrl(x)" muted loop playsinline preload="metadata" @mouseenter="hoverPlay" @mouseleave="hoverPause" />
            <img v-else :src="viewUrl(x)" loading="lazy" alt="" />
            <span class="clabel">{{ x.base ? 'No LoRA' : `Step ${x.step}` }}</span>
          </button>
          <div v-for="t in waitingCards.filter((w) => w.againIndex === g.index)" :key="t.key" class="tcell pending" :class="{ live: t.live }" :style="tileOf(g.steps.at(-1) ?? g.base)">
            <img v-if="t.live && liveManual?.preview" :src="liveManual.preview" alt="" />
            <i v-else :class="t.live ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-clock-outline'" />
            <div v-if="t.live" class="qbar"><span :style="{ width: `${t.frac * 100}%` }" /></div>
            <span class="clabel">{{ t.state }}</span>
          </div>
          <!-- the next render of this prompt: same seed, the LoRA as it is now -->
          <button
            v-if="status === 'training' && !g.steps.some((x) => x.step === step) && !queuedTries.some((q) => q.again === g.index)"
            class="tcell add"
            :style="tileOf(g.steps.at(-1) ?? g.base)"
            title="Render this prompt again (same seed) with the LoRA as it is now"
            @click="again(g)"
          >
            <i class="mdi mdi-plus" />
            <span>Step {{ step }}</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.manual {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.try {
  /* one row: the prompt grows, its options and Render sit at its right (they wrap below when narrow) */
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 8px;
}
.try-in {
  flex: 1 1 320px;
  min-width: 0;
}
.try-in {
  display: block;
  width: 100%;
  resize: none;
  overflow-y: auto;
  font: inherit;
  font-size: 13px;
  line-height: 1.5;
  padding: 8px 10px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
  max-height: 10em;
}
.try-in:focus {
  outline: none;
  border-color: var(--zen-accent);
}
.try-opts {
  display: flex;
  gap: 8px;
  align-items: center;
  flex: 0 0 auto;
  min-height: 38px; /* the prompt box's one-line height, so they line up with it */
}
.try-opts > :first-child {
  width: 170px;
}
.empty-manual {
  padding: 8px 0;
}
.mrows {
  display: flex;
  flex-direction: column;
}
.mrow2 {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 12px;
  padding: 8px 6px;
  border-radius: var(--r-md);
}
.mrow2 + .mrow2 {
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 40%, transparent);
}
.mrow2:hover {
  background: color-mix(in srgb, var(--zen-text) 3%, transparent);
}
.mprompt {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.mtext {
  font-size: 11.5px;
  line-height: 1.4;
  color: var(--zen-text);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.mmeta {
  font-size: 10px;
  color: var(--zen-muted);
}
.macts {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.mrow2:hover .macts,
.macts:focus-within,
.macts.held {
  opacity: 1;
}
.macts button.on {
  color: var(--zen-selected, var(--zen-accent));
}
.macts button {
  width: 24px;
  height: 22px;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.macts button:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.tcell.add {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
}
.tcell.add i {
  font-size: 18px;
}
.tcell.add:hover {
  border-color: var(--zen-accent);
  color: var(--zen-accent);
}
.mcells {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.tcell {
  position: relative;
  flex: none;
  height: 96px;
  aspect-ratio: 1;
  border: none;
  padding: 0;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
}
.tcell img,
.tcell video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.tcell.pending {
  display: grid;
  place-items: center;
  cursor: default;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font-size: 20px;
}
.tcell.pending.live {
  border: 1px solid color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
.tcell.pending img {
  position: absolute;
  inset: 0;
}
.clabel {
  position: absolute;
  left: 6px;
  bottom: 6px;
  max-width: calc(100% - 12px);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  pointer-events: none;
}
.qbar {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 3px;
  background: color-mix(in srgb, var(--zen-text) 10%, transparent);
}
.qbar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
  transition: width var(--zen-dur) ease;
}
.base-sw {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
  white-space: nowrap;
}
</style>
