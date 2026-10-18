// Adds ZenTrainer to ComfyUI: a top bar button, an "Open ZenTrainer" command and menu entry.
// The app itself is served at /zentrainer/.
import { app } from '../../scripts/app.js'

const open = () => window.open(new URL('zentrainer/', window.location.href), '_blank')

// ComfyUI also uses the tooltip as the button's aria-label: the only stable handle on a button it
// renders for us. It reads `actionBarButtons` once, at registration, so the setting hides the
// button with a style rather than unregistering it.
const TOOLTIP = 'Open ZenTrainer'
const HIDE_STYLE_ID = 'zentrainer-hide-topbar-button'

// Our lotus (the favicon's petals, without its tile) as an icon class: a mask filled with the
// current text colour, so it sits in the top bar like ComfyUI's own icons and follows the theme.
const PETAL = (s) => `M0,0 C${26 * s},${-19.8 * s} ${18.2 * s},${-51.48 * s} 0,${-66 * s} C${-18.2 * s},${-51.48 * s} ${-26 * s},${-19.8 * s} 0,0Z`
const LOTUS = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><g transform="translate(32 55) scale(0.62)">${
  [[-70, 0.82, 0.45], [70, 0.82, 0.45], [-36, 0.91, 0.7], [36, 0.91, 0.7], [0, 1.0, 1]]
    .map(([r, s, o]) => `<path d="${PETAL(s * 1.18)}" transform="rotate(${r})" opacity="${o}"/>`).join('')
}</g></svg>`
const icon = document.createElement('style')
icon.textContent = `.zentrainer-logo::before { content: ''; display: inline-block; width: 1.15em; height: 1.15em;
  vertical-align: -0.2em; background: currentColor;
  -webkit-mask: url('data:image/svg+xml,${encodeURIComponent(LOTUS)}') center / contain no-repeat;
  mask: url('data:image/svg+xml,${encodeURIComponent(LOTUS)}') center / contain no-repeat; }`
document.head.appendChild(icon)

function showButton(show) {
  document.getElementById(HIDE_STYLE_ID)?.remove()
  if (show) return
  const style = document.createElement('style')
  style.id = HIDE_STYLE_ID
  style.textContent = `button[aria-label="${TOOLTIP}"] { display: none !important; }`
  document.head.appendChild(style)
}

app.registerExtension({
  name: 'nynxz.zentrainer',
  actionBarButtons: [{ icon: 'zentrainer-logo', tooltip: TOOLTIP, onClick: open }],
  commands: [{ id: 'zentrainer.open', label: 'Open ZenTrainer', icon: 'zentrainer-logo', function: open }],
  menuCommands: [{ path: ['Extensions'], commands: ['zentrainer.open'] }],
  settings: [
    {
      id: 'ZenTrainer.topbarButton',
      name: 'Show the ZenTrainer top bar button',
      category: ['ZenTrainer', 'Top bar', 'Button'],
      type: 'boolean',
      defaultValue: true,
      tooltip: 'Show the button that opens ZenTrainer in the top bar. Extensions → Open ZenTrainer and the command palette work either way.',
      onChange: (value) => showButton(value !== false),
    },
  ],
})
