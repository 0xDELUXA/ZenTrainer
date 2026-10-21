<script setup lang="ts">
// The dataset as a grid of uniform cards (zoomable): the picture with its badges (mask, video,
// repeats, image and face issues, no caption), mute and select on hover, the caption below.
import { ZenTip } from '@nynxz/zenkit-ui'
import { fileInfo, pct, secs, useDatasetContext } from './context'

const { zoom, inputs, fit, showMasks, activeSet, cap, selected, isMuted, setMuted, imageFlags, faceOf, faceIssue, faceNote, shown, toggleSelect, onItemClick, onItemMenu, full, thumb, thumbSize } = useDatasetContext()
</script>

<template>
  <div class="grid" :style="{ '--card': `${zoom}px` }">
    <article
      v-for="(it, i) in shown"
      :key="it.path"
      class="card-item"
      :class="{ sel: selected.has(it.path), muted: isMuted(it) }"
      @contextmenu="onItemMenu(it, i, $event)"
    >
      <button class="thumb" :title="it.path" :data-peek="full(it)" @click="onItemClick(it, i, $event)">
        <img :src="thumb(it, thumbSize)" loading="lazy" :alt="cap(it)" :class="{ 'mask-dim': showMasks && it.mask }" :style="{ objectFit: fit }" />
        <!-- the mask view: what the mask leaves out is a purple/black checker (Source's missing
             texture: unmistakable on any picture), and only what it trains shows -->
        <span v-if="showMasks && it.mask" class="mask-checker" />
        <img
          v-if="showMasks && it.mask"
          :src="thumb(it, thumbSize)"
          loading="lazy"
          alt=""
          class="mask-lit"
          :style="{ objectFit: fit, maskImage: `url(${thumb({ path: it.mask }, thumbSize)})`, WebkitMaskImage: `url(${thumb({ path: it.mask }, thumbSize)})`, maskSize: fit, WebkitMaskSize: fit }"
        />
        <!-- an edit pair: its input picture in the corner (the big one is the target) -->
        <img v-if="inputs(it).length" :src="thumb({ path: inputs(it)[0]! }, 256)" loading="lazy" alt="" class="control-inset" :title="`Input: ${inputs(it).join(', ')}`" />
        <span class="badges">
          <span v-if="it.mask" class="badge" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'This mask is a different shape from its image: it won\'t line up' : `Masked: ${pct(it.mask_coverage)} trains at full weight`">
            <i :class="it.mask_mismatch ? 'mdi mdi-alert-outline' : 'mdi mdi-layers-outline'" /> {{ pct(it.mask_coverage) }}
          </span>
          <span v-if="it.kind === 'video'" class="badge" :title="`Video: ${it.frames ?? '?'} frames at ${it.fps ?? '?'} fps`"><i class="mdi mdi-play" /> {{ secs(it.duration) }}</span>
          <span v-if="it.repeats > 1" class="badge" :title="`Repeats ${it.repeats}×`">×{{ it.repeats }}</span>
          <span v-if="imageFlags[it.path]?.length" class="badge warn" :title="imageFlags[it.path]!.join(', ')"><i class="mdi mdi-image-filter-center-focus-weak" /> {{ imageFlags[it.path]!.length }}</span>
          <span v-if="faceOf(it)" class="badge" :class="{ warn: !!faceIssue(it) }" :title="faceIssue(it) ?? faceNote(it)">
            <i :class="faceOf(it)!.faces === 0 ? 'mdi mdi-face-man-outline' : 'mdi mdi-face-recognition'" />
            {{ faceOf(it)!.faces === 0 ? '0' : faceOf(it)!.likeness != null ? faceOf(it)!.likeness!.toFixed(2) : faceOf(it)!.faces }}
          </span>
          <span v-if="!cap(it)" class="badge warn" title="No caption"><i class="mdi mdi-text-box-remove-outline" /></span>
        </span>
      </button>
      <span class="info-btn" @click.stop><ZenTip :text="fileInfo(it)" label="File details" /></span>
      <button
        class="mute-btn"
        :class="{ on: isMuted(it) }"
        :title="isMuted(it) ? `Muted in “${activeSet}”: click to train on it again` : `Mute in “${activeSet}”: kept, but not trained on`"
        @click.stop="setMuted([it.path], !isMuted(it))"
      >
        <i :class="isMuted(it) ? 'mdi mdi-eye-off' : 'mdi mdi-eye-outline'" />
      </button>
      <label class="check" :class="{ on: selected.has(it.path) }" @click.prevent="toggleSelect(it, i, $event)">
        <i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
      </label>
      <button class="card-body" :title="cap(it) || 'No caption'" @click="onItemClick(it, i, $event)">
        <span class="cap-text" :class="{ empty: !cap(it) }">{{ cap(it) || 'No caption' }}</span>
      </button>
    </article>
  </div>
</template>

<style scoped src="./items.css"></style>
<style scoped>
/* grid: the slider sets the smallest card; columns stretch to fill the row evenly */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(var(--card), 1fr));
  gap: 12px;
  align-items: start;
}
.card-item {
  transform-origin: top left;
  position: relative;
  display: flex;
  flex-direction: column;
  border-radius: var(--r-xl);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 60%, transparent);
  overflow: hidden;
  transition:
    border-color 0.15s ease,
    box-shadow 0.15s ease;
}
/* hover in the primary colour, softer than a selection's full accent ring */
.card-item:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
.card-item.sel {
  border-color: var(--zen-accent);
  box-shadow: 0 0 0 1px var(--zen-accent);
}
/* every card the same size: a square thumbnail (image positioned inside, so its own
   aspect can't stretch the card) over a fixed-height caption block */
.thumb {
  position: relative;
  display: block;
  width: 100%;
  aspect-ratio: 1;
  padding: 0;
  border: none;
  background: var(--zen-surface-2);
  cursor: zoom-in;
  overflow: hidden;
}
.thumb img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.card-item.muted .thumb img:not(.control-inset):not(.mask-ov) {
  opacity: 0.35;
  filter: grayscale(0.6);
}
.info-btn {
  position: absolute;
  top: 8px;
  right: 40px;
  z-index: 2;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border-radius: var(--r-sm);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.info-btn :deep(*) {
  color: inherit;
}
.card-item:hover .info-btn,
.info-btn:focus-within {
  opacity: 1;
}
.mute-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 2;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.card-item:hover .mute-btn,
.mute-btn.on,
.mute-btn:focus-visible {
  opacity: 1;
}
.mute-btn.on {
  color: var(--zen-warn);
}
.thumb .control-inset {
  position: absolute;
  left: 6px;
  top: 6px;
  width: 34%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-sm);
  border: 2px solid var(--zen-surface);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
.badges {
  position: absolute;
  right: 6px;
  bottom: 6px;
  display: flex;
  gap: 4px;
}
.badge {
  font-size: 11px;
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.6);
  color: #fff;
}
.badge.warn {
  color: #fbbf24;
}
.card-item:hover .check,
.grid:has(.check.on) .check {
  opacity: 1;
}
.card-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  height: 52px;
  flex: none;
  padding: 9px 10px 8px;
  border: none;
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.cap-text {
  font-size: 12px;
  line-height: 1.4;
  height: 2.8em; /* exactly two lines; the clamp lives here, padding stays outside */
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.cap-text.empty {
  color: var(--zen-warn);
}
/* the mask view */
.thumb img.mask-dim {
  opacity: 0;
}
.thumb .mask-checker {
  position: absolute;
  inset: 0;
  background: repeating-conic-gradient(#f0f 0 25%, #000 0 50%) 0 0 / 16px 16px;
  pointer-events: none;
}
.thumb img.mask-lit {
  mask-mode: luminance;
  mask-repeat: no-repeat;
  mask-position: center;
  pointer-events: none;
}
</style>
