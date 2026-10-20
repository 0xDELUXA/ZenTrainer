// The VRAM and RAM series the System page and the log drawer chart. Colours come from the
// pack's chart tokens when it has them; dashes carry identity too (never colour alone).
import type { Series } from '@/components/MemChart.vue'

export const VRAM_SERIES: Series[] = [
  { key: 'vram_used', label: 'Device used', color: 'var(--chart-1, var(--zen-accent))' },
  { key: 'torch_reserved', label: 'PyTorch reserved', color: 'var(--chart-2, var(--zen-info))', dash: '6 3' },
  { key: 'torch_allocated', label: 'PyTorch allocated', color: 'var(--chart-3, var(--zen-ok))', dash: '2 3' },
  { key: 'aimdo_used', label: 'Dynamic VRAM (aimdo)', color: 'var(--chart-4, var(--zen-warn))', dash: '10 3 2 3' },
]
export const RAM_SERIES: Series[] = [
  { key: 'ram_used', label: 'System used', color: 'var(--chart-1, var(--zen-accent))' },
  { key: 'proc_rss', label: 'ComfyUI process', color: 'var(--chart-2, var(--zen-info))', dash: '6 3' },
]
