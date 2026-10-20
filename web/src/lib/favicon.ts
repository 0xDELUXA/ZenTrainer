// The tab icon follows the theme: ZenKit's lotus (the sidebar's mark) on a tile of the pack's
// primary colour, cornered by the pack's own radius, so a square pack gets a square icon.
// public/favicon.svg is the same mark in the default pack, for before the app loads.

// ZenLotus's petals (ZenKit packages/ui/src/primitives/ZenLotus.vue)
const petal = (h: number, w: number) =>
  `M0,0 C${w},${-h * 0.3} ${w * 0.7},${-h * 0.78} 0,${-h} C${-w * 0.7},${-h * 0.78} ${-w},${-h * 0.3} 0,0Z`
const PETALS: [string, number, number][] = [
  [petal(54, 24), -70, 0.35],
  [petal(54, 24), 70, 0.35],
  [petal(66, 26), -36, 0.6],
  [petal(66, 26), 36, 0.6],
  [petal(78, 28), 0, 1],
]

const esc = (c: string) => c.replace(/[<>&"]/g, '')

export function faviconSvg(bg: string, fg: string, radiusPx: number): string {
  // a 64px tile; the pack radius is sized for ~40px controls, so scale it up, capped at a circle-ish 24
  const rx = Math.min(24, Math.max(0, radiusPx * 1.6))
  const petals = PETALS.map(([d, r, o]) => `<path d="${d}" transform="rotate(${r})" opacity="${o}"/>`).join('')
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">` +
    `<rect width="64" height="64" rx="${rx}" fill="${esc(bg)}"/>` +
    `<g transform="translate(32 47) scale(0.4)" fill="${esc(fg)}">${petals}</g></svg>`
  )
}

/** Redraw the tab icon from the theme tokens now on <html>. */
export function updateFavicon(): void {
  const cs = getComputedStyle(document.documentElement)
  const bg = cs.getPropertyValue('--primary').trim() || '#7c3aed'
  const fg = cs.getPropertyValue('--primary-foreground').trim() || '#ffffff'
  // --radius may be rem: let the browser resolve it
  const probe = document.createElement('div')
  probe.style.cssText = 'position:absolute;visibility:hidden;width:var(--radius,8px)'
  document.body.append(probe)
  const radius = probe.getBoundingClientRect().width
  probe.remove()

  let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]')
  if (!link) {
    link = document.createElement('link')
    link.rel = 'icon'
    document.head.append(link)
  }
  link.type = 'image/svg+xml'
  link.href = `data:image/svg+xml,${encodeURIComponent(faviconSvg(bg, fg, radius))}`
}
