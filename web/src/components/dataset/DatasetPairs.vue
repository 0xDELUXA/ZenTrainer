<script setup lang="ts">
// Edit pairs: every input, then the result, the instruction beside them (one picture in and one
// out, the same shape, as a drag-across compare).
import { ZenCompare } from '@nynxz/zenkit-ui'
import { vAutosize } from '@/lib/autosize'
import { aspect, isSimplePair, isVideoPath, useDatasetContext } from './context'

const { zoom, inputs, cap, k, drafts, saving, save, selected, shown, toggleSelect, onItemClick, onItemMenu, ctrlView, full, thumb } = useDatasetContext()
</script>

<template>
  <div class="pairs" :style="{ '--ph': `${zoom}px` }">
    <article v-for="(it, i) in shown" :key="it.path" class="pair card" :class="{ sel: selected.has(it.path), dirty: drafts[k(it)] != null }" @contextmenu="onItemMenu(it, i, $event)">
      <label class="check" :class="{ on: selected.has(it.path) }" @click.prevent="toggleSelect(it, i, $event)">
        <i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
      </label>
      <div class="pair-media">
        <!-- one picture in, one out, same shape: drag across to compare -->
        <div v-if="isSimplePair(it, inputs(it))" class="pane compare" :style="{ aspectRatio: aspect(it) }">
          <ZenCompare :before="thumb({ path: inputs(it)[0]! }, 768)" :after="thumb(it, 768)" before-label="Input" after-label="Result" track-closest=".pair" />
        </div>
        <template v-else>
          <button v-for="(c, n) in inputs(it)" :key="c" class="pane" :title="c" :data-peek="ctrlView(it, c)" @click="onItemClick(it, i, $event)">
            <video v-if="isVideoPath(c)" :src="ctrlView(it, c)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
            <img v-else :src="thumb({ path: c }, 512)" loading="lazy" alt="" />
            <span class="pane-tag">{{ inputs(it).length > 1 ? `Input ${n + 1}` : 'Input' }}</span>
          </button>
          <i v-if="inputs(it).length" class="mdi mdi-arrow-right pair-arrow" />
          <button class="pane result" :title="it.path" :data-peek="full(it)" @click="onItemClick(it, i, $event)">
            <video v-if="it.kind === 'video'" :src="ctrlView(it, it.path)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
            <img v-else :src="thumb(it, 512)" loading="lazy" alt="" />
            <span class="pane-tag">{{ inputs(it).length ? 'Result' : 'No input' }}</span>
          </button>
        </template>
      </div>
      <div class="pair-text">
        <textarea
          v-autosize
          class="cap-edit"
          :class="{ empty: !cap(it) && drafts[k(it)] == null }"
          :value="drafts[k(it)] ?? cap(it)"
          placeholder="The instruction: what to change"
          rows="2"
          @input="drafts[k(it)] = ($event.target as HTMLTextAreaElement).value"
          @blur="save(it)"
          @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
        />
        <span class="card-meta mono">
          {{ it.path.split('/').at(-1) }}<span v-if="it.width"> · {{ it.width }}×{{ it.height }}</span>
          <span v-if="inputs(it).length > 1"> · {{ inputs(it).length }} inputs</span>
          <span v-if="saving[k(it)]"> · saving…</span>
        </span>
      </div>
    </article>
  </div>
</template>

<style scoped src="./items.css"></style>
<style scoped>
.pairs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(max(260px, calc(var(--ph) * 1.25)), 1fr));
  gap: 12px;
  align-items: start;
}
.pair {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
}
.pair.sel {
  border-color: var(--zen-accent);
}
.pair .check {
  top: 16px;
  left: 16px;
  z-index: 2;
}
.pair-media {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: var(--ph);
  overflow-x: auto;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.pane {
  position: relative;
  height: 100%;
  flex: none;
  padding: 0;
  border: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
}
.pane img,
.pane video {
  height: 100%;
  width: auto;
  max-width: 100%;
  object-fit: contain;
  display: block;
}
.pane.compare {
  max-width: 100%;
}
.pane.compare {
  cursor: ew-resize;
}
.pane.compare :deep(.zen-compare) {
  width: 100%;
  height: 100%;
}
.pane-tag {
  position: absolute;
  left: 6px;
  bottom: 6px;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  pointer-events: none;
}
.pair-arrow {
  color: var(--zen-muted);
  font-size: 18px;
  flex: none;
}
.pair-text {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  padding: 0 2px;
}
.card-meta {
  font-size: 10px;
  color: var(--zen-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
