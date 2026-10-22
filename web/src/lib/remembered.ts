import { ref, watch } from 'vue'

/** a section's open/folded, remembered per browser (blocked storage: it just isn't remembered) */
export function remembered(key: string, open: boolean) {
  const r = ref((() => {
    try {
      const v = localStorage.getItem(key)
      return v == null ? open : v !== '0'
    } catch {
      return open
    }
  })())
  watch(r, (v) => {
    try {
      localStorage.setItem(key, v ? '1' : '0')
    } catch {
      // storage blocked
    }
  })
  return r
}
