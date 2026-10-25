// Likeness as a share of what it can reach: the dataset's own photos only look so much like each
// other (their median against the rest, the `ceiling` on each likeness row), and a LoRA that
// caught the face perfectly would still score about that. Rows from before it have none.

export interface Ceilinged {
  ceiling?: number | null
}

/** The run's ceiling: from its newest row that has one. */
export const likeCeiling = (rows?: Ceilinged[] | null): number | null =>
  [...(rows ?? [])].reverse().find((r) => r.ceiling != null)?.ceiling ?? null

/** "93%", or "" without a ceiling. */
export const likeShare = (score: number, ceiling: number | null | undefined): string =>
  ceiling ? `${Math.round((score / ceiling) * 100)}%` : ''

export const SHARE_TIP = 'As a share of how alike your dataset\'s own photos are to each other (100% is as alike as they are)'
