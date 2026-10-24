// A run's metric rows as one continuous history, across resumes. Each session's `elapsed` starts
// again at 0 (it's that session's wall-clock), so a resumed run on the time axis drew its second
// half over its first, with a stroke back to the start; and a resume from an earlier checkpoint
// logs those steps again, so the abandoned stretch and its redo overlapped on either axis. Here
// a redone step replaces the rows from there on, and later sessions' times carry on from where
// the history they continue stops. Rows are never changed: a shifted one is a copy.

import type { MetricRow } from '@/api/engine'

export function resumed<T extends MetricRow>(rows: readonly T[]): T[] {
  const out: T[] = []
  let offset = 0 // seconds added to this session's elapsed
  let rawPrev = -Infinity // the previous row's own elapsed, to see a session start
  for (const r of rows) {
    while (out.length && out.at(-1)!.step >= r.step) out.pop() // resumed from before them: they were abandoned
    const raw = r.elapsed ?? 0
    if (raw < rawPrev) offset = out.at(-1)?.elapsed ?? 0 // a new session: its clock restarted
    rawPrev = raw
    out.push(r.elapsed == null || !offset ? r : { ...r, elapsed: raw + offset })
  }
  return out
}

// A page left open on a long run gets a row every step: past LIVE_ROWS its older rows are folded
// two into one (the later row, with both losses' mean), as the engine thins them (runs/thin.py),
// so it doesn't grow back to every step it ever saw. The latest LIVE_TAIL stay exact.
const LIVE_ROWS = 6000
const LIVE_TAIL = 400

export function keepCompact<T extends MetricRow>(rows: T[]): void {
  if (rows.length <= LIVE_ROWS) return
  const head = rows.length - LIVE_TAIL
  const folded: T[] = []
  for (let i = 0; i < head; i += 2) {
    const a = rows[i]!
    const b = rows[i + 1]
    if (!b || i + 1 >= head) folded.push(a)
    else folded.push(a.loss != null && b.loss != null ? { ...b, loss: (a.loss + b.loss) / 2 } : b)
  }
  rows.splice(0, head, ...folded)
}
