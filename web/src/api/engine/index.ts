// Client for the ZenTrainer engine's HTTP API (engine/zentrainer/server/routes/), one file per
// area as on the engine. Everything is re-exported here: import from '@/api/engine'.

import { appsApi } from './apps'
import { datasetsApi } from './datasets'
import { hostApi } from './host'
import { modelsApi } from './models'
import { runsApi } from './runs'

export { EngineError, EVENTS, viewUrl } from './client'
export * from './spec'
export * from './runs'
export * from './datasets'
export * from './apps'
export * from './models'
export * from './host'

export const engine = { ...hostApi, ...runsApi, ...datasetsApi, ...appsApi, ...modelsApi }
