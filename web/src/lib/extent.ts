// The largest and smallest of a series, by a loop: Math.max(...values) passes every value as an
// argument, and past ~130k (a long run's steps) that overflows the call stack, which stopped the
// run page drawing at all.

export function maxOf(values: Iterable<number>, floor = -Infinity): number {
  let m = floor
  for (const v of values) if (v > m) m = v
  return m
}

export function minOf(values: Iterable<number>, ceiling = Infinity): number {
  let m = ceiling
  for (const v of values) if (v < m) m = v
  return m
}
