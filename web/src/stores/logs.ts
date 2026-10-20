// ComfyUI's console, live: backfilled from /internal/logs/raw (ComfyUI keeps the last few
// hundred writes), then streamed over the websocket after PATCH /internal/logs/subscribe.
// ComfyUI drops the subscription when the socket closes, so it's renewed on every
// reconnect (the "status" message that carries our sid).
//
// Entries are raw stdout/stderr writes, not lines: they're stitched into lines here, ANSI
// codes stripped, progress bars (\r rewrites) collapsed to their latest state, and a
// Python traceback kept together as one block.

import { computed, ref } from 'vue'
import { getClientId, onComfyMessage } from '@/api/comfy'

export type LogLevel = 'info' | 'warn' | 'error'
export interface LogLine {
  id: number
  t: string // ISO time of the write that finished the line
  text: string // may span several lines (a traceback)
  level: LogLevel
}
interface RawEntry {
  t: string
  m: string
}

const MAX = 5000
const ERROR = /\b(error|exception|traceback|critical|fatal)\b|^!!!/i
const WARN = /\b(warn(ing)?|deprecated)\b/i
// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07/g

export const logLines = ref<LogLine[]>([])
/** the unfinished last line (a progress bar mid-update), shown live */
export const logPartial = ref('')
/** errors that arrived while the drawer was closed */
export const unseenErrors = ref(0)
export const drawerOpen = ref(false)
export const DRAWER_HEIGHT_KEY = 'zentrainer.logs.height'
export const DRAWER_MIN_H = 140
/** the drawer's height (per viewer); the page reserves this much scroll room while it's open */
export const drawerHeight = ref(
  (() => {
    try {
      const v = Number(localStorage.getItem(DRAWER_HEIGHT_KEY))
      return v >= DRAWER_MIN_H ? v : 300
    } catch {
      return 300
    }
  })(),
)

export const logCounts = computed(() => {
  let warn = 0
  let error = 0
  for (const l of logLines.value) {
    if (l.level === 'warn') warn++
    else if (l.level === 'error') error++
  }
  return { warn, error }
})

let nextId = 1
let partial = '' // raw text of the current line, before \r collapse
let traceback: { text: string[]; t: string } | null = null
let lastT = '' // newest write already taken in (backfill and stream overlap)

// ComfyUI's logger tags its lines; untagged output (prints, tqdm, libraries) is guessed from words
const TAG = /^\[(DEBUG|DETAIL|INFO|WARNING|ERROR|CRITICAL)\]\s?/
const TAGGED: Record<string, LogLevel> = { DEBUG: 'info', DETAIL: 'info', INFO: 'info', WARNING: 'warn', ERROR: 'error', CRITICAL: 'error' }
const levelOf = (s: string): LogLevel => {
  const tag = TAG.exec(s)
  if (tag) return TAGGED[tag[1] ?? 'INFO'] ?? 'info'
  return ERROR.test(s) ? 'error' : WARN.test(s) ? 'warn' : 'info'
}

function push(text: string, t: string, level = levelOf(text)) {
  const list = logLines.value
  list.push({ id: nextId++, t, text, level })
  if (list.length > MAX) list.splice(0, list.length - MAX)
  if (level === 'error' && !drawerOpen.value) unseenErrors.value++
}

// a finished line: either part of an open traceback or a line of its own
function finish(line: string, t: string) {
  if (traceback) {
    // a traceback runs on through indented frames, blank lines and the chained-exception
    // banners, and ends at the first unindented line: the exception itself
    const chained = /^(During handling of the above exception|The above exception was the direct cause)/.test(line)
    if (/^\s/.test(line) || line === '' || chained || /^Traceback \(most recent call last\)/.test(line)) {
      traceback.text.push(line)
      return
    }
    traceback.text.push(line)
    push(traceback.text.join('\n'), t, 'error')
    traceback = null
    return
  }
  if (/^(\[ERROR\]\s?)?Traceback \(most recent call last\)/.test(line)) {
    traceback = { text: [line], t }
    return
  }
  const level = levelOf(line)
  const text = line.replace(/^\[(INFO|DEBUG|DETAIL)\]\s?/, '') // the colour says it; WARNING/ERROR tags stay
  if (text.trim() === '') return
  push(text, t, level)
}

const collapse = (s: string) => s.slice(s.lastIndexOf('\r') + 1) // a \r rewrite keeps only its latest state

function take(entries: RawEntry[]) {
  let changed = false
  for (const e of entries) {
    if (!e || typeof e.m !== 'string') continue
    if (e.t && lastT && e.t <= lastT) continue
    if (e.t) lastT = e.t
    const parts = (partial + e.m.replace(ANSI, '')).split('\n')
    partial = parts.pop() ?? ''
    for (const p of parts) finish(collapse(p).replace(/\s+$/, ''), e.t)
    changed = true
  }
  if (changed) logPartial.value = collapse(partial)
}

async function subscribe() {
  try {
    await fetch('/internal/logs/subscribe', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: true, clientId: getClientId() }),
    })
  } catch {
    // ComfyUI unreachable: the next reconnect tries again
  }
}

let started = false
export async function startLogSync(): Promise<void> {
  if (started) return
  started = true
  onComfyMessage((msg) => {
    if (msg.type === 'logs') take(((msg.data as { entries?: RawEntry[] }).entries ?? []) as RawEntry[])
    else if (msg.type === 'status' && (msg.data as { sid?: string }).sid) subscribe() // (re)connected
  })
  try {
    const raw = (await (await fetch('/internal/logs/raw')).json()) as { entries?: RawEntry[] }
    take(raw.entries ?? [])
    unseenErrors.value = 0 // history isn't news
  } catch {
    // no backfill; the stream still works
  }
  subscribe()
}

/** Empty the view (ComfyUI's own buffer is untouched). */
export function clearLogs(): void {
  logLines.value = []
  logPartial.value = ''
  unseenErrors.value = 0
}

export function toggleDrawer(open = !drawerOpen.value): void {
  drawerOpen.value = open
  if (open) unseenErrors.value = 0
}
