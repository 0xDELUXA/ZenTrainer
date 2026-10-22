<script setup lang="ts">
// Appearance, after lora.moe's settings: a mode toggle and a grid of theme-pack cards.
// Previews are drawn from each pack's own tokens, so every ZenKit pack and user
// drop-in gets one without shipping banner images.
import { ZenButton, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { engine, type ComfyVersion, type EngineSettings } from '@/api/engine'
import { SETTINGS, setSetting, settings, type SettingId } from '@/stores/settings'
import { currentModes, currentTheme, packCards, setMode, setTheme } from '@/lib/theme'

const mode = computed(() => currentTheme.value.mode)
// registry settings, by category (Appearance renders inside its own section)
const settingsIn = (category: string) => (Object.keys(SETTINGS) as SettingId[]).filter((id) => SETTINGS[id].category === category)
const currentName = computed(() => packCards.value.find((p) => p.id === currentTheme.value.id)?.name)
// keep the user's light/dark preference; a one-mode pack shows its own mode meanwhile
const pick = (id: string) => setTheme(id, currentTheme.value.preferred)
// ComfyUI's own palettes, first in the list
const COMFY: Record<string, string> = { 'maroon-volt': 'ComfyUI', 'comfyui-brand': 'ComfyUI' }
const packFilter = ref<'all' | 'dark' | 'light'>('all')
const shownPacks = computed(() =>
  packCards.value
    .filter((p) => packFilter.value === 'all' || p.modes.includes(packFilter.value))
    .sort((a, b) => Number(!!COMFY[b.id]) - Number(!!COMFY[a.id]) || (a.id === 'maroon-volt' ? -1 : b.id === 'maroon-volt' ? 1 : 0)),
)
// the swatch shows the pack in the current mode when it has it
const sw = (p: (typeof packCards.value)[number]) => (p.modes.includes(mode.value) ? p.swatch[mode.value] : p.swatch[p.modes[0]!])!

// ComfyUI: version, the tested pin, and (on request) how far upstream has moved
const comfy = ref<ComfyVersion | null>(null)
const checking = ref(false)
onMounted(async () => (comfy.value = await engine.comfyui().catch(() => null)))
// Storage: the ZenTrainer folder (datasets and results out of ComfyUI's input/output). The engine
// reads it at start, so a saved change shows as pending until ComfyUI restarts.
const storage = ref<EngineSettings | null>(null)
const folderDraft = ref('')
const savingFolder = ref(false)
const folderNote = ref('')
const folderError = ref('')
onMounted(async () => {
  storage.value = await engine.settings().catch(() => null)
  folderDraft.value = storage.value?.settings.folder ?? ''
})
async function saveFolder() {
  savingFolder.value = true
  folderError.value = folderNote.value = ''
  try {
    storage.value = await engine.putSettings({ folder: folderDraft.value.trim() })
    folderDraft.value = storage.value.settings.folder
    folderNote.value = storage.value.message || 'Saved'
  } catch (e) {
    folderError.value = String((e as Error).message ?? e)
  } finally {
    savingFolder.value = false
  }
}

async function checkUpdates() {
  checking.value = true
  try {
    comfy.value = await engine.comfyui(true)
  } finally {
    checking.value = false
  }
}
</script>

<template>
  <div class="page narrow">
    <div class="page-head">
      <div class="titles">
        <h1>Settings</h1>
      </div>
    </div>

    <section v-if="comfy" class="card section">
      <div>
        <h2>ComfyUI</h2>
        <p class="muted">ZenTrainer uses ComfyUI's internals, so it's tested against a pinned version (comfy.lock).</p>
      </div>
      <dl class="kv">
        <dt>Version</dt><dd class="mono">{{ comfy.version ?? '–' }} <span class="muted">{{ comfy.describe }}</span></dd>
        <dt>Location</dt><dd class="mono small-t">{{ comfy.path }}</dd>
        <template v-if="comfy.pin">
          <dt>Tested pin</dt>
          <dd>
            <span class="mono">{{ comfy.pin.describe }}</span>
            <span v-if="comfy.matches_pin" class="ok"><i class="mdi mdi-check-circle-outline" /> running the pin</span>
            <span v-else class="warn"><i class="mdi mdi-alert-outline" /> not the pin: run <span class="mono">scripts/zt update</span></span>
          </dd>
        </template>
        <template v-if="comfy.upstream">
          <dt>Upstream</dt>
          <dd>
            <span class="mono">{{ comfy.upstream.describe }}</span>
            <span v-if="comfy.upstream.behind" class="muted"> · {{ comfy.upstream.behind }} commits newer than this checkout</span>
            <span v-else class="ok"> · up to date</span>
          </dd>
        </template>
        <template v-if="comfy.upstream_error"><dt>Upstream</dt><dd class="warn small-t">{{ comfy.upstream_error }}</dd></template>
      </dl>
      <div class="row-actions">
        <ZenButton size="sm" icon="mdi mdi-cloud-sync-outline" :disabled="checking" @click="checkUpdates">{{ checking ? 'Checking…' : 'Check for updates' }}</ZenButton>
        <span v-if="comfy.upstream?.behind" class="muted small-t">Newer isn't automatically better here: move the pin with <span class="mono">scripts/zt update --ref &lt;tag&gt;</span> and test a short run first.</span>
      </div>
    </section>

    <section v-if="storage" class="card section">
      <div>
        <h2>Storage</h2>
        <p class="muted">
          By default datasets live in ComfyUI's input folder and LoRAs, samples and merges under its output/zentrainer.
          A ZenTrainer folder keeps them out of there: datasets in <span class="mono">datasets/</span>, results in
          <span class="mono">output/</span>. Nothing already there moves.
        </p>
      </div>
      <div class="stack">
        <div class="eyebrow">ZenTrainer folder</div>
        <div class="row-actions">
          <ZenInput v-model="folderDraft" class="grow mono" placeholder="Empty: ComfyUI's input and output folders" @keydown.enter="saveFolder" />
          <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="savingFolder || folderDraft.trim() === storage.settings.folder" @click="saveFolder">Save</ZenButton>
        </div>
        <p class="muted hint">An absolute path outside ComfyUI's input and output folders. Takes effect after restarting ComfyUI.</p>
        <p v-if="folderError" class="warn">{{ folderError }}</p>
        <p v-else-if="storage.restart_needed" class="warn"><i class="mdi mdi-restart" /> Saved: restart ComfyUI (System → Restart ComfyUI) to use it. Until then the folders below stay in use.</p>
        <p v-else-if="folderNote" class="ok">{{ folderNote }}</p>
      </div>
      <div class="stack">
        <div class="eyebrow">In use now</div>
        <dl class="kv">
          <dt>Datasets</dt><dd class="mono small-t">{{ storage.active.datasets }}</dd>
          <dt>Outputs</dt><dd class="mono small-t">{{ storage.active.output }}</dd>
          <dt>LoRAs</dt><dd class="mono small-t">{{ storage.active.loras }}</dd>
        </dl>
      </div>
    </section>

    <section class="card section">
      <div>
        <h2>Appearance</h2>
        <p class="muted">Each pack brings its own palette and rounding. Some support light and dark; some are one-mode.</p>
      </div>

      <div class="stack">
        <div class="eyebrow">Mode</div>
        <div class="segmented">
          <button
            v-for="m in ['light', 'dark'] as const"
            :key="m"
            :class="{ on: mode === m }"
            :disabled="!currentModes.includes(m)"
            @click="setMode(m)"
          >
            <i :class="m === 'light' ? 'mdi mdi-white-balance-sunny' : 'mdi mdi-weather-night'" />
            {{ m === 'light' ? 'Light' : 'Dark' }}
          </button>
        </div>
        <p v-if="currentModes.length < 2" class="muted hint">
          {{ currentName }} is {{ mode }}-only. Packs with both modes follow your
          {{ currentTheme.preferred }} preference.
        </p>
      </div>

      <div v-for="id in settingsIn('Appearance')" :key="id" class="stack">
        <div class="eyebrow">{{ SETTINGS[id].label }}</div>
        <div class="segmented">
          <button
            v-for="o in SETTINGS[id].options"
            :key="String(o.value)"
            :class="{ on: settings[id] === o.value }"
            @click="setSetting(id, o.value)"
          >
            <i v-if="o.icon" :class="o.icon" /> {{ o.label }}
          </button>
        </div>
        <p v-if="SETTINGS[id].description" class="muted hint">{{ SETTINGS[id].description }}</p>
      </div>

      <div class="stack">
        <div class="pack-head">
          <span class="eyebrow">Theme pack</span>
          <span class="muted small">{{ shownPacks.length }} of {{ packCards.length }}</span>
          <span class="spacer" />
          <div class="segmented small-seg">
            <button v-for="f in (['all', 'dark', 'light'] as const)" :key="f" :class="{ on: packFilter === f }" @click="packFilter = f">
              {{ { all: 'All', dark: 'Dark', light: 'Light' }[f] }}
            </button>
          </div>
        </div>
        <div class="packs">
          <button
            v-for="p in shownPacks"
            :key="p.id"
            class="pack"
            :class="{ active: currentTheme.id === p.id }"
            :title="p.description || p.name"
            @click="pick(p.id)"
          >
            <span
              class="sw"
              :style="{
                background: sw(p).bg,
                borderColor: sw(p).border,
                '--c': sw(p).card,
                '--p': sw(p).primary,
                '--f': sw(p).fg,
                '--rr': sw(p).radius,
              }"
            >
              <span class="sw-card"><span class="sw-line" /><span class="sw-btn" /></span>
            </span>
            <span class="pack-text">
              <span class="name">{{ p.name }}</span>
              <span class="tags">
                <span v-if="COMFY[p.id]" class="tag">{{ COMFY[p.id] }}</span>
                <i v-if="p.modes.includes('light')" class="mdi mdi-white-balance-sunny" title="Light" />
                <i v-if="p.modes.includes('dark')" class="mdi mdi-weather-night" title="Dark" />
              </span>
            </span>
            <i v-if="currentTheme.id === p.id" class="mdi mdi-check check" />
          </button>
        </div>
      </div>
    </section>

    <section class="card section">
      <div>
        <h2>Training</h2>
        <p class="muted">Defaults for new runs. Each run keeps the values it started with; where LoRAs are saved is picked in New run.</p>
      </div>
      <div v-for="id in settingsIn('Training').filter((x) => SETTINGS[x].options.length)" :key="id" class="stack">
        <div class="eyebrow">{{ SETTINGS[id].label }}</div>
        <div class="segmented">
          <button
            v-for="o in SETTINGS[id].options"
            :key="String(o.value)"
            :class="{ on: settings[id] === o.value }"
            @click="setSetting(id, o.value as never)"
          >
            <i v-if="o.icon" :class="o.icon" /> {{ o.label }}
          </button>
        </div>
        <p v-if="SETTINGS[id].description" class="muted hint">{{ SETTINGS[id].description }}</p>
      </div>
    </section>
  </div>
</template>

<style scoped>
.narrow {
  max-width: 880px;
}
.section {
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 22px;
}
h2 {
  margin: 0 0 4px;
  font-size: 16px;
}
p {
  margin: 0;
}
.stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.kv {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px 16px;
  margin: 0;
  font-size: 13px;
}
.kv dt {
  color: var(--zen-muted);
}
.kv dd {
  margin: 0;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: baseline;
}
.small-t {
  font-size: 11px;
  word-break: break-all;
}
.ok {
  color: var(--zen-ok);
  font-size: 12px;
}
.warn {
  color: var(--zen-warn);
  font-size: 12px;
}
.row-actions .grow {
  flex: 1;
  min-width: 240px;
}
.row-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.segmented {
  display: inline-flex;
  align-self: flex-start;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 40%, transparent);
}
.segmented button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.segmented button:disabled {
  opacity: 0.4;
  cursor: default;
}
.hint {
  font-size: 12px;
}
.segmented button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.pack-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.small-seg button {
  padding: 3px 10px;
  font-size: 12px;
}
.packs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 8px;
}
.pack {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 10px 7px 7px;
  text-align: left;
  font: inherit;
  color: inherit;
  cursor: pointer;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 40%, transparent);
  transition: border-color 0.15s ease;
  min-width: 0;
}
.pack:hover {
  border-color: var(--zen-muted);
}
.pack.active {
  border-color: var(--zen-selected, var(--zen-accent));
  box-shadow: 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
/* the pack in miniature: its background, a card, a line of text, its primary button */
.sw {
  width: 46px;
  height: 36px;
  flex: none;
  box-sizing: border-box;
  padding: 5px;
  border: 1px solid;
  border-radius: calc(var(--rr) * 0.8);
}
.sw-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: 100%;
  box-sizing: border-box;
  padding: 3px;
  border-radius: calc(var(--rr) * 0.5);
  background: var(--c);
}
.sw-line {
  width: 70%;
  height: 3px;
  border-radius: 2px;
  background: var(--f);
  opacity: 0.7;
}
.sw-btn {
  align-self: flex-end;
  width: 45%;
  height: 5px;
  border-radius: calc(var(--rr) * 0.4);
  background: var(--p);
}
.pack-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1;
}
.name {
  font-weight: 600;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tags {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--zen-muted);
}
.tag {
  white-space: nowrap;
  font-size: 10px;
  padding: 0 5px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-accent) 18%, transparent);
  color: var(--zen-text);
}
.check {
  width: 20px;
  height: 20px;
  flex: none;
  display: grid;
  place-items: center;
  border-radius: var(--r-md);
  background: var(--zen-accent);
  color: var(--zen-accent-text);
  font-size: 13px;
}
</style>
