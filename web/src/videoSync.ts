// Plays a group of videos together from the first frame and keeps them in step: the first
// video is the clock, the rest are nudged back when they drift (decoders start at slightly
// different times, and a stall in one would otherwise put it behind for good).
export function playTogether(videos: HTMLVideoElement[]): () => void {
  let frame = 0
  const [clock, ...rest] = videos
  if (!clock) return () => {}
  for (const v of videos) {
    v.pause()
    v.currentTime = 0
  }
  void Promise.all(videos.map((v) => v.play().catch(() => {})))
  const tick = () => {
    for (const v of rest) {
      if (v.readyState >= 2 && Math.abs(v.currentTime - clock.currentTime) > 0.06 && clock.currentTime < v.duration) {
        v.currentTime = clock.currentTime
      }
    }
    frame = requestAnimationFrame(tick)
  }
  frame = requestAnimationFrame(tick)
  return () => {
    cancelAnimationFrame(frame)
    for (const v of videos) v.pause()
  }
}
