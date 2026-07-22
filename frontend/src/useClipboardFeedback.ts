/**
 * Hook for "copy, then briefly show a ✅" buttons. It copies text to the clipboard
 * and remembers which value was just copied so the UI can flash confirmation on the
 * matching button, then clears that state after `resetMs`.
 */
import { useEffect, useRef, useState } from 'react'

import { copyTextToClipboard } from './clipboard'

/**
 * Returns `copiedValue` (the most recently copied text, or null once it expires) and
 * `copyWithFeedback(value)` which copies and sets that state; the flag auto-resets
 * after `resetMs`. Resolves true on success, false if the value was empty or the copy failed.
 */
export function useClipboardFeedback(resetMs = 1200): {
  copiedValue: string | null
  copyWithFeedback: (value: string) => Promise<boolean>
} {
  const [copiedValue, setCopiedValue] = useState<string | null>(null)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
    }
  }, [])

  async function copyWithFeedback(value: string): Promise<boolean> {
    const text = String(value || '')
    if (!text) return false

    try {
      await copyTextToClipboard(text)
      setCopiedValue(text)
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
      timerRef.current = window.setTimeout(() => {
        setCopiedValue((current) => (current === text ? null : current))
        timerRef.current = null
      }, resetMs)
      return true
    } catch {
      return false
    }
  }

  return {
    copiedValue,
    copyWithFeedback,
  }
}
