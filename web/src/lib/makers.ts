// Who makes a model family, by its logo: engine/catalog/families.json (the catalog's hand
// corrections, shared with the engine) says each family's maker and logo; the logos are in
// web/public/logos/ (64 px, transparent). Read at build time, so a logo needs no engine restart.
// "mono" logos are black ink on transparent: shown white on dark themes.
import catalog from '../../../engine/catalog/families.json'

interface Entry {
  match: string
  maker?: string
  logo?: string
  mono?: boolean
  other?: boolean
}
const ENTRIES = ((catalog as { families?: Entry[] }).families ?? []).map((e) => ({ ...e, re: new RegExp(e.match) }))

/** a family few people train: listed under "Other models" in the picker, out of the main list */
export const isOther = (familyId: string | null | undefined) => !!ENTRIES.find((e) => e.re.test((familyId ?? '').toLowerCase()))?.other

/** the family's maker and logo (served from the app's web/public/logos/), or null */
export function makerLogo(familyId: string | null | undefined): { src: string; mono: boolean; maker: string } | null {
  const id = (familyId ?? '').toLowerCase()
  const hit = ENTRIES.find((e) => e.re.test(id))
  if (!hit?.logo) return null
  return { src: `${import.meta.env.BASE_URL}logos/${hit.logo}`, mono: !!hit.mono, maker: hit.maker ?? '' }
}
