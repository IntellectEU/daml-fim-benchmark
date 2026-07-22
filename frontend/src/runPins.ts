/**
 * "Pinning" lets the user stick chosen runs to the top of the runs table so they
 * stay visible while scrolling or filtering. This file holds the small bit of
 * state and helpers behind that: which run ids are pinned (persisted in
 * localStorage so it survives reloads), how to toggle one, and how to reorder
 * rows so pinned runs come first.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'

type RunLike = {
  run_id: string
}

function loadPins(storageKey: string): Set<string> {
  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) return new Set<string>()
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return new Set<string>()
    return new Set(parsed.map(String))
  } catch {
    return new Set<string>()
  }
}

function persistPins(storageKey: string, pins: Set<string>): void {
  window.localStorage.setItem(storageKey, JSON.stringify(Array.from(pins)))
}

/**
 * Hook owning the set of pinned run ids for a given localStorage key. Returns the
 * current `pins` set, a `setPins` setter, a `togglePin(runId)` action, and an
 * `isPinned(runId)` check; changes are written back to localStorage automatically.
 */
export function useRunPins(storageKey: string) {
  const [pins, setPins] = useState<Set<string>>(() => loadPins(storageKey))

  useEffect(() => {
    persistPins(storageKey, pins)
  }, [pins, storageKey])

  const togglePin = useCallback((runId: string): void => {
    setPins((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })
  }, [])

  const isPinned = useCallback((runId: string): boolean => pins.has(runId), [pins])

  return { pins, setPins, togglePin, isPinned }
}

/** Reorder rows so pinned runs come first, each group keeping its original order. */
export function orderPinnedRuns<T extends RunLike>(rows: T[], pins: Set<string>): T[] {
  const pinnedRuns: T[] = []
  const unpinnedRuns: T[] = []
  for (const row of rows) {
    if (pins.has(row.run_id)) pinnedRuns.push(row)
    else unpinnedRuns.push(row)
  }
  return [...pinnedRuns, ...unpinnedRuns]
}

/**
 * The id of the last pinned run in row order, or null if none are pinned — used to
 * draw a divider after the pinned block (so the table can separate pinned from the rest).
 */
export function useLastPinnedRunId<T extends RunLike>(rows: T[], pins: Set<string>): string | null {
  return useMemo(() => {
    let lastPinnedId: string | null = null
    for (const row of rows) {
      if (pins.has(row.run_id)) lastPinnedId = row.run_id
    }
    return lastPinnedId
  }, [rows, pins])
}
