<script setup lang="ts">
// The dataset as a table: sortable columns, every caption editable in place.
import { vAutosize } from '@/autosize'
import { kb, pct, useDatasetContext } from './context'

const { cap, k, drafts, saving, save, sortKey, sortDir, selected, isMuted, faceOf, faceIssue, faceNote, shown, sortBy, toggleSelect, onItemClick, onItemMenu, allShownSelected, toggleAll, full, thumb } = useDatasetContext()
</script>

<template>
  <div class="table card">
    <div class="tr th">
      <label class="c-check" @click.prevent="toggleAll"><i :class="allShownSelected ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></label>
      <span class="c-thumb" />
      <button class="c-name sortable" @click="sortBy('name')">Name <i v-if="sortKey === 'name'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
      <button class="c-res sortable" @click="sortBy('res')">Size <i v-if="sortKey === 'res'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
      <button class="c-bytes sortable" @click="sortBy('size')">File <i v-if="sortKey === 'size'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
      <span class="c-mask">Mask</span>
      <span class="c-face">Face</span>
      <button class="c-cap sortable" @click="sortBy('caption')">Caption <i v-if="sortKey === 'caption'" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" /></button>
      <span class="c-act" />
    </div>
    <div v-for="(it, i) in shown" :key="it.path" class="tr" :class="{ sel: selected.has(it.path), dirty: drafts[k(it)] != null, muted: isMuted(it) }" @contextmenu="onItemMenu(it, i, $event)">
      <label class="c-check" @click.prevent="toggleSelect(it, i, $event)"><i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></label>
      <button class="c-thumb" :data-peek="full(it)" @click="onItemClick(it, i, $event)"><img :src="thumb(it, 128)" loading="lazy" alt="" /></button>
      <span class="c-name mono ellipsis" :title="it.path">{{ it.path }}</span>
      <span class="c-res mono muted">{{ it.width ? `${it.width}×${it.height}` : '–' }}</span>
      <span class="c-bytes mono muted">{{ kb(it.bytes) }}</span>
      <span class="c-mask small" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'Different shape from its image: it won\'t line up' : ''">{{ it.mask ? pct(it.mask_coverage) : '' }}</span>
      <span
        class="c-face small mono"
        :class="{ warn: !!faceIssue(it) }"
        :title="faceOf(it) ? faceIssue(it) ?? faceNote(it) : 'Run Faces to check'"
      >
        <template v-if="faceOf(it)">
          <i v-if="faceIssue(it)" class="mdi mdi-alert-outline" />
          {{ faceOf(it)!.faces === 0 ? 'none' : faceOf(it)!.faces > 1 ? `${faceOf(it)!.faces} faces` : faceOf(it)!.likeness?.toFixed(2) }}
        </template>
      </span>
      <textarea
        v-autosize
        class="c-cap cap-edit"
        :class="{ empty: !cap(it) && drafts[k(it)] == null }"
        :value="drafts[k(it)] ?? cap(it)"
        placeholder="No caption: add one"
        rows="1"
        @input="drafts[k(it)] = ($event.target as HTMLTextAreaElement).value"
        @blur="save(it)"
        @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
      />
      <span class="c-act">
        <span v-if="saving[k(it)]" class="muted small">saving…</span>
      </span>
    </div>
  </div>
</template>

<style scoped src="./items.css"></style>
<style scoped>
.tr.muted .c-thumb img {
  opacity: 0.35;
}
.table {
  padding: 0;
  overflow: hidden;
}
.tr {
  display: grid;
  grid-template-columns: 32px 64px minmax(120px, 0.9fr) 84px 64px 44px 70px minmax(280px, 4fr) 48px;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.tr.th {
  border-top: none;
  position: sticky;
  top: 0;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--zen-muted);
  background: color-mix(in srgb, var(--zen-surface) 90%, transparent);
}
.tr.sel {
  background: color-mix(in srgb, var(--zen-accent) 10%, transparent);
}
.tr.dirty .cap-edit {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
.sortable {
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  text-transform: inherit;
  letter-spacing: inherit;
  text-align: left;
  padding: 0;
  cursor: pointer;
}
.c-check {
  font-size: 18px;
  color: var(--zen-muted);
  cursor: pointer;
  display: grid;
  place-items: center;
}
.tr.sel .c-check {
  color: var(--zen-accent);
}
.c-thumb {
  width: 56px;
  height: 56px;
  padding: 0;
  border: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: zoom-in;
}
.c-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.c-name {
  font-size: 12px;
}
.c-res,
.c-bytes {
  font-size: 11px;
}
.c-mask {
  text-align: center;
  color: var(--zen-accent);
}
@media (max-width: 900px) {
  .tr {
    grid-template-columns: 32px 56px 1fr 56px;
  }
  .c-res,
  .c-bytes,
  .c-mask {
    display: none;
  }
  .c-cap {
    grid-column: 2 / -1;
  }
}
</style>
