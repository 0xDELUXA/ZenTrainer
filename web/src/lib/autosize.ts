// v-autosize: a textarea as tall as its text (up to its CSS max-height, then it scrolls), so
// captions read in full instead of through a two-line slot. Re-fits on input, on value
// changes from outside (another image in the lightbox) and on width changes.

import type { Directive } from 'vue'

const fit = (el: HTMLTextAreaElement) => {
  el.style.height = 'auto'
  el.style.height = `${el.scrollHeight + (el.offsetHeight - el.clientHeight)}px`
  // taller than its max-height: CSS can fade the cut (.clamped)
  el.classList.toggle('clamped', el.scrollHeight > el.clientHeight + 1)
}
const observers = new WeakMap<HTMLTextAreaElement, ResizeObserver>()

export const vAutosize: Directive<HTMLTextAreaElement> = {
  mounted(el) {
    el.addEventListener('input', () => fit(el))
    el.addEventListener('blur', () => ((el.scrollTop = 0), fit(el))) // a clamped box shows its start again
    el.addEventListener('focus', () => requestAnimationFrame(() => fit(el)))
    let width = el.clientWidth
    const ro = new ResizeObserver(() => {
      if (el.clientWidth !== width) {
        width = el.clientWidth
        fit(el)
      }
    })
    ro.observe(el)
    observers.set(el, ro)
    requestAnimationFrame(() => fit(el))
  },
  updated(el) {
    fit(el)
  },
  unmounted(el) {
    observers.get(el)?.disconnect()
  },
}
