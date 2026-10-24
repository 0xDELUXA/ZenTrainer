// The Tools page's list: small utilities that work on files (LoRAs, datasets), beside the Apps
// that generate. Add one here and give it a route under /tools.

export interface Tool {
  id: string
  name: string
  description: string
  icon: string
  category: 'loras' | 'datasets'
  /** the card's picture (web/public or ComfyUI's /templates); none = an icon panel */
  image?: string
  accent?: string
}

export const TOOLS: Tool[] = [
  {
    id: 'merge',
    name: 'Merge LoRAs',
    description: 'Bake several LoRAs into one at the strengths you choose: a soup of a few runs on one subject, or a whole stack in one file.',
    icon: 'mdi mdi-call-merge',
    category: 'loras',
    accent: '#a855f7',
  },
  {
    id: 'compact',
    name: 'Compact a LoRA',
    description: 'Shrink a LoRA or LoKr to the size you want, keeping rank where the layers change most. A LoKr can stay one or become a LoRA.',
    icon: 'mdi mdi-arrow-collapse-all',
    category: 'loras',
    accent: '#22c55e',
  },
  {
    id: 'inspect',
    name: 'Inspect a model file',
    description: 'Look inside any .safetensors file, a LoRA or a whole model: layers, shapes, precisions, metadata. Save a copy changed.',
    icon: 'mdi mdi-file-tree-outline',
    category: 'loras',
    accent: '#f59e0b',
  },
  {
    id: 'split-videos',
    name: 'Split videos into clips',
    description: 'Cut a dataset’s long videos into short clips spread through each one, soundtrack kept. The originals move aside, so training uses the clips.',
    icon: 'mdi mdi-content-cut',
    category: 'datasets',
    accent: '#0ea5e9',
  },
]

export const TOOL_CATEGORIES: Record<Tool['category'], string> = { loras: 'LoRAs and models', datasets: 'Datasets' }
