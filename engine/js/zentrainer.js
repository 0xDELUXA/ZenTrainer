// Adds an "Open ZenTrainer" command to ComfyUI. The app itself is served at /zentrainer/.
import { app } from '../../scripts/app.js'

const open = () => window.open(new URL('zentrainer/', window.location.href), '_blank')

app.registerExtension({
  name: 'nynxz.zentrainer',
  commands: [{ id: 'zentrainer.open', label: 'Open ZenTrainer', icon: 'pi pi-bolt', function: open }],
  menuCommands: [{ path: ['Extensions'], commands: ['zentrainer.open'] }],
})
