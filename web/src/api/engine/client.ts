// The fetch helper every area's calls go through, and what they share: errors, image URLs, websocket events.

export class EngineError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown,
  ) {
    super(message)
  }
}

export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/zentrainer/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new EngineError(body?.error ?? res.statusText, res.status, body)
  return body as T
}

/** ComfyUI's /view URL for an image in its input/output folders. */
export function viewUrl(f: { filename: string; subfolder: string; type?: string }): string {
  const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder, type: f.type ?? 'output' })
  return `/view?${q}`
}

/** Websocket event names; mirror engine/zentrainer/events.py. */
export const EVENTS = {
  metrics: 'zentrainer.metrics',
  checkpoint: 'zentrainer.checkpoint',
  status: 'zentrainer.status',
  sample: 'zentrainer.sample',
  sampleProgress: 'zentrainer.sample_progress',
  validation: 'zentrainer.validation',
  likeness: 'zentrainer.likeness',
  caption: 'zentrainer.caption',
  download: 'zentrainer.download',
} as const
