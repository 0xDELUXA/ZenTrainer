// The Tools page's list: small utilities that work on files (LoRAs, datasets), beside the Apps
// that generate. Add one here and give it a route under /tools.

export interface Tool {
  id: string
  name: string
  description: string
  icon: string
  category: 'loras' | 'datasets'
}

export const TOOLS: Tool[] = [
  {
    id: 'merge',
    name: 'Merge LoRAs',
    description: 'Bake several LoRAs into one at the strengths you choose, and optionally shrink it to a small rank. A soup of a few runs on one subject, or a stack in one file.',
    icon: 'mdi mdi-call-merge',
    category: 'loras',
  },
  {
    id: 'split-videos',
    name: 'Split videos into clips',
    description: 'Cut a dataset’s long videos into short clips spread through each one, soundtrack kept. The originals move aside, so training uses the clips.',
    icon: 'mdi mdi-content-cut',
    category: 'datasets',
  },
]

export const TOOL_CATEGORIES: Record<Tool['category'], string> = { loras: 'LoRAs', datasets: 'Datasets' }
