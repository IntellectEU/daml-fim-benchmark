/**
 * Copy a string to the system clipboard, working in more places than the modern
 * clipboard API alone: it tries `navigator.clipboard` first, then falls back to a
 * hidden <textarea> + `execCommand('copy')` for older or non-secure contexts.
 * Throws if neither path is available; copying an empty string is a no-op.
 */
export async function copyTextToClipboard(value: string): Promise<void> {
  const text = String(value ?? '')
  if (!text) return

  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return
  }

  if (typeof document !== 'undefined') {
    const input = document.createElement('textarea')
    input.value = text
    input.setAttribute('readonly', '')
    input.style.position = 'fixed'
    input.style.opacity = '0'
    document.body.appendChild(input)
    input.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(input)
    if (ok) return
  }

  throw new Error('Clipboard not available')
}
