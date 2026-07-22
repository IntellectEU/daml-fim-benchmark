/**
 * The FIM (fill-in-the-middle) runs page — the main and only screen of this UI.
 *
 * It lists the benchmark's evaluation runs in a table (with pinning, filtering by
 * eval kind / free-text, and selectable metric columns), lets you act on runs
 * (edit notes, archive, delete, compare two side by side), and drills into a single
 * run's per-sample results — showing each example's prompt, model prediction, and
 * target, with copy-to-clipboard helpers throughout. All data comes from the
 * backend via the functions in api.ts; this file is the page's UI and local state.
 *
 * It's a large single-file component on purpose (one page, many small handlers);
 * the helper constants and pure helpers above `FimPage` keep the component body
 * focused on rendering and wiring.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Key, KeyboardEvent, ReactNode } from 'react'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Dropdown,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import SyntaxHighlighter from 'react-syntax-highlighter'
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs'

import {
  archiveRuns,
  compareRuns,
  deleteRuns,
  fetchHealth,
  fetchRunResultDetail,
  fetchRunResults,
  fetchRuns,
  restartBackend,
  saveRunNotes,
} from './api'
import { RunNameCell } from './components/RunNameCell'
import { orderPinnedRuns, useLastPinnedRunId, useRunPins } from './runPins'
import { useClipboardFeedback } from './useClipboardFeedback'
import type { HealthResponse } from './api'
import type {
  CompareResponse,
  DetailMode,
  EvalKind,
  ResponseFilter,
  ResultDetailResponse,
  ResultListItem,
  ResultListResponse,
  RunListItem,
} from './types'

const { Text } = Typography

const PIN_STORAGE_KEY = 'dca_fim_frontend_pins'
const RUN_COLUMN_STORAGE_KEY = 'dca_fim_frontend.columns'
const RUN_COLUMN_KNOWN_STORAGE_KEY = 'dca_fim_frontend.columns_known'
const DEFAULT_HIDDEN_RUN_COLUMNS = new Set(['run_id'])
const LINT_COLUMN_KEYS = ['lint_pass_rate', 'lint_id', 'lint_ood'] as const
const LINT_COLUMN_KEY_SET = new Set<string>(LINT_COLUMN_KEYS)
const RUN_TABLE_SELECTION_WIDTH = 46
const STATIC_RUN_COLUMN_WIDTHS = {
  pin: 64,
  age: 116,
  eval_kind: 100,
  version: 120,
  examples: 78,
  weighted_total_avg: 106,
  total_avg: 122,
  id_avg: 122,
  ood_avg: 122,
  undisclosed_avg: 122,
  lint_pass_rate: 62,
  lint_id: 58,
  lint_ood: 62,
  actions: 96,
  repo: 68,
} as const
const FLEX_RUN_COLUMN_SPECS = {
  run_name: { min: 230, base: 330, max: 520 },
  notes: { min: 120, base: 200, max: 340 },
  run_id: { min: 240, base: 330, max: 520 },
  model: { min: 180, base: 260, max: 380 },
} as const

type FlexRunColumnKey = keyof typeof FLEX_RUN_COLUMN_SPECS

function formatPercent(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

// Backend timestamps use the compact `YYMMDD-HHMMSS` form (UTC); the run_id also
// embeds the same pattern, so fall back to it when the explicit field is missing.
function parseCompactTimestamp(text: string | null | undefined): Date | null {
  const m = /^(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/.exec((text ?? '').trim())
  if (!m) return null
  const [, yy, mo, dd, hh, mi, ss] = m
  const ms = Date.UTC(2000 + Number(yy), Number(mo) - 1, Number(dd), Number(hh), Number(mi), Number(ss))
  return Number.isNaN(ms) ? null : new Date(ms)
}

function parseRunTimestamp(row: { timestamp?: string | null; run_id?: string }): Date | null {
  const fromField = parseCompactTimestamp(row.timestamp)
  if (fromField) return fromField
  const match = /(\d{6}-\d{6})/.exec(row.run_id ?? '')
  return match ? parseCompactTimestamp(match[1]) : null
}

function formatRelativeAge(date: Date | null, now: number): string {
  if (!date) return '—'
  const diffSec = Math.floor((now - date.getTime()) / 1000)
  if (diffSec < 0) return 'just now'
  if (diffSec < 60) return `${diffSec}s ago`
  const min = Math.floor(diffSec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.floor(hr / 24)
  if (day < 7) return `${day}d ago`
  if (day < 30) return `${Math.floor(day / 7)}w ago`
  if (day < 365) return `${Math.floor(day / 30)}mo ago`
  return `${Math.floor(day / 365)}y ago`
}

type AgeFilter = 'all' | '1h' | 'today' | '24h' | '7d' | '30d'

const AGE_FILTER_OPTIONS: Array<{ value: AgeFilter; label: string }> = [
  { value: 'all', label: 'Any time' },
  { value: '1h', label: 'Last hour' },
  { value: 'today', label: 'Today' },
  { value: '24h', label: 'Last 24h' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
]

const AGE_FILTER_WINDOWS_MS: Record<Exclude<AgeFilter, 'all' | 'today'>, number> = {
  '1h': 60 * 60 * 1000,
  '24h': 24 * 60 * 60 * 1000,
  '7d': 7 * 24 * 60 * 60 * 1000,
  '30d': 30 * 24 * 60 * 60 * 1000,
}

function runMatchesAgeFilter(
  row: { timestamp?: string | null; run_id?: string },
  filter: AgeFilter,
  now: number
): boolean {
  if (filter === 'all') return true
  const date = parseRunTimestamp(row)
  if (!date) return false
  if (filter === 'today') {
    const start = new Date()
    start.setHours(0, 0, 0, 0)
    return date.getTime() >= start.getTime()
  }
  return now - date.getTime() <= AGE_FILTER_WINDOWS_MS[filter]
}

function numericSort(a: number | null, b: number | null): number {
  const av = a ?? Number.NEGATIVE_INFINITY
  const bv = b ?? Number.NEGATIVE_INFINITY
  return av - bv
}

function loadVisibleColumns(storageKey: string): string[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map(String).filter(Boolean)
  } catch {
    return []
  }
}

function getColumnKey(
  col: { key?: Key; dataIndex?: unknown }
): string | null {
  if (col.key !== undefined && col.key !== null) return String(col.key)
  const dataIndex = col.dataIndex
  if (typeof dataIndex === 'string' || typeof dataIndex === 'number') return String(dataIndex)
  if (Array.isArray(dataIndex)) return dataIndex.map(String).join('.')
  return null
}

function filterVisibleColumns<T extends object>(
  cols: ColumnsType<T>,
  visibleKeys: string[]
): ColumnsType<T> {
  const visible = new Set(visibleKeys)
  return cols.filter((col) => {
    const key = getColumnKey(col)
    return key ? visible.has(key) : true
  })
}

function readErrorDetail(err: unknown, fallback: string): string {
  if (typeof err === 'object' && err !== null) {
    const anyErr = err as { message?: string; code?: string; response?: { data?: unknown } }
    const responseData = anyErr.response?.data
    const responseText =
      typeof responseData === 'string'
        ? responseData
        : responseData && typeof responseData === 'object'
          ? JSON.stringify(responseData)
          : ''
    return [anyErr.message, anyErr.code, responseText].filter(Boolean).join(' | ') || fallback
  }
  return fallback
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

function renderHighlightedCode(value: unknown, maxHeightPx = 180): ReactNode {
  const source = asText(value)
  if (!source) {
    return <pre className="code-preview" style={{ maxHeight: `${maxHeightPx}px` }}> </pre>
  }
  return (
    <SyntaxHighlighter
      language="haskell"
      style={atomOneDark}
      wrapLongLines
      customStyle={{
        margin: '6px 0 0 0',
        maxHeight: `${maxHeightPx}px`,
        overflow: 'auto',
        borderRadius: '8px',
        padding: '10px',
        fontFamily: '"IBM Plex Mono", "SFMono-Regular", Menlo, monospace',
        fontSize: '12px',
        lineHeight: 1.45,
      }}
      codeTagProps={{ style: { fontFamily: 'inherit' } }}
    >
      {source}
    </SyntaxHighlighter>
  )
}

type ModelPillStyle = {
  background: string
  color: string
  borderColor: string
}

function runModelLabel(row: Pick<RunListItem, 'base_model' | 'model'>): string {
  const label = String(row.base_model || row.model || '').trim()
  return label || '—'
}

// The key a run's pill color is grouped by: the displayed model label, with any
// trailing "(...)" annotation stripped. So "Qwen2.5-Coder-7B (daml3)" and
// "Qwen2.5-Coder-7B" share a color, while different models (incl. different
// sizes/versions) get different colors. Independent of the provider field,
// which is inconsistently populated.
function modelPillKey(row: RunListItem): string {
  return runModelLabel(row).replace(/\s*\([^)]*\)\s*$/, '').trim().toLowerCase() || 'model'
}

// Assign each distinct model key a hue spread evenly around the color wheel, so
// every model present is maximally separated from the others (no hash collisions
// making unrelated models look alike). Sorted for deterministic assignment.
function buildPillHueByKey(rows: RunListItem[]): Map<string, number> {
  const keys = Array.from(new Set(rows.map(modelPillKey))).sort()
  const step = keys.length ? 360 / keys.length : 0
  const out = new Map<string, number>()
  keys.forEach((key, i) => out.set(key, Math.round(i * step)))
  return out
}

// Readable pastel pill style for a hue (fixed saturation/lightness keeps the
// text legible on the background).
function pillStyleFromHue(hue: number): ModelPillStyle {
  return {
    background: `hsl(${hue}, 70%, 92%)`,
    color: `hsl(${hue}, 60%, 25%)`,
    borderColor: `hsl(${hue}, 65%, 62%)`,
  }
}

function scoreToColor(rate: number): string {
  if (rate < 0.5) {
    const p = rate * 2
    const r = 255
    const g = Math.round(100 + (135 * p))
    const b = Math.round(100 - (41 * p))
    return `rgb(${r}, ${g}, ${b})`
  }
  const p = (rate - 0.5) * 2
  const r = Math.round(255 - (155 * p))
  const g = Math.round(235 + (20 * p))
  const b = Math.round(59 + (41 * p))
  return `rgb(${r}, ${g}, ${b})`
}

function scoreBadgeStyle(rateRaw: number | null): { background: string; color: string } {
  const parsed = typeof rateRaw === 'number' ? rateRaw : 0
  const rate = Math.max(0, Math.min(1, Number.isFinite(parsed) ? parsed : 0))
  if (rate >= 0.999999) return { background: 'rgb(24, 110, 56)', color: '#ffffff' }
  if (rate >= 0.99) return { background: 'rgb(44, 146, 76)', color: '#ffffff' }
  if (rate >= 0.95) return { background: 'rgb(86, 184, 98)', color: '#102312' }
  if (rate >= 0.9) return { background: 'rgb(166, 214, 88)', color: '#1b2510' }
  const visualRate = 1 - (1 - rate) ** 0.55
  return { background: scoreToColor(visualRate), color: '#111111' }
}

const SCORE_PILL_RIM_EXPLAINER: Record<'id' | 'ood', string> = {
  id: 'Blue rim: score over in-domain repos (in the model’s training data).',
  ood: 'Cyan rim: score over out-of-domain repos (not in training data).',
}

function scoreBadge(rate: number | null, rimKind?: 'id' | 'ood', decimals = 1): ReactNode {
  if (rate === null || Number.isNaN(rate)) return '—'
  const style = scoreBadgeStyle(rate)
  const rimClass = rimKind === 'id' ? ' score-pill-id' : rimKind === 'ood' ? ' score-pill-ood' : ''
  const pill = (
    <span className={`score-pill${rimClass}`} style={{ background: style.background, color: style.color }}>
      {`${(rate * 100).toFixed(decimals)}%`}
    </span>
  )
  if (!rimKind) return pill
  return (
    <Tooltip title={SCORE_PILL_RIM_EXPLAINER[rimKind]} mouseEnterDelay={0.2}>
      {pill}
    </Tooltip>
  )
}

const OVERGEN_EXPLAINER =
  'Over-generation: the prediction starts with the correct target but adds extra ' +
  'trailing text after it (exact matches are excluded). These near-misses are ' +
  'often still useful completions — the +x% shows how many there are on top of ' +
  'the exact-match score.'

function scoreCell(
  rate: number | null,
  overgenRate: number | null,
  rimKind?: 'id' | 'ood',
  rateDecimals = 1
): ReactNode {
  return (
    <div className="score-cell-wrap">
      {scoreBadge(rate, rimKind, rateDecimals)}
      {overgenRate !== null && !Number.isNaN(overgenRate) && (
        <Tooltip title={OVERGEN_EXPLAINER} mouseEnterDelay={0.2}>
          <span className="score-subtext">{`+${Math.round(overgenRate * 100)}%`}</span>
        </Tooltip>
      )}
    </div>
  )
}

function scoreHeader(label: string, subtext: string, explainer?: string): ReactNode {
  const labelNode = explainer ? (
    <Tooltip title={explainer} mouseEnterDelay={0.2}>
      <strong>{label}</strong>
    </Tooltip>
  ) : (
    <strong>{label}</strong>
  )
  const subtextNode =
    subtext === '+over-generation' ? (
      <Tooltip title={OVERGEN_EXPLAINER} mouseEnterDelay={0.2}>
        <span className="score-header-subtext">{subtext}</span>
      </Tooltip>
    ) : (
      <span className="score-header-subtext">{subtext}</span>
    )
  return (
    <div className="score-header-wrap">
      {labelNode}
      {subtextNode}
    </div>
  )
}

type CompareMetricRow = {
  key: string
  metric: string
  left: string
  right: string
  delta: string
  emphasized: boolean
}

type DisplayRunRow = RunListItem & {
  _cv_is_child?: boolean
  _cv_expandable?: boolean
  _cv_is_expanded?: boolean
}

/**
 * Alternate U OOD value for CV parent rows.
 * Motivation:
 * - A repo in a fold's training split can end up with no examples actually
 *   sampled into training (small datasets), so that fold evaluates it as OOD.
 * - The repo then appears as OOD in more than one fold.
 * - That blocks the "single OOD coverage" highlight even when the run is otherwise comparable.
 */
type CvParentUOodAdjustment = {
  adjustedRate: number
  duplicateOodRepos: string[]
  accidentalOodRepos: string[]
}

const MATCH_OUTCOME_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'match', label: '✅ Match' },
  { value: 'prefix', label: '➕ Over-gen' },
  { value: 'mismatch', label: '❌ Mismatch' },
]

const SYNTAX_OUTCOME_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'pass', label: '✅ Pass' },
  { value: 'fail', label: '❌ Fail' },
  { value: 'inconclusive', label: '⚪ Inconclusive' },
]

const RESPONSE_FILTER_OPTIONS = [
  { value: 'all', label: 'All Responses' },
  { value: 'non_empty', label: 'Non-empty' },
  { value: 'empty', label: 'Empty' },
]

export function FimPage() {
  const runTableContainerRef = useRef<HTMLDivElement | null>(null)
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null)
  const [restartPending, setRestartPending] = useState(false)

  const [includeArchived, setIncludeArchived] = useState(false)
  const [evalFilter, setEvalFilter] = useState<EvalKind>('All')
  const [runSort, setRunSort] = useState<{ key: string; order: 'ascend' | 'descend' } | null>(null)
  const [ageFilter, setAgeFilter] = useState<AgeFilter>('all')
  const [query, setQuery] = useState('')
  const [modelFilter, setModelFilter] = useState<string>('all')
  const [showRepoDetails, setShowRepoDetails] = useState(false)
  const [showLintColumns, setShowLintColumns] = useState(false)
  const [visibleRunColumns, setVisibleRunColumns] = useState<string[]>(() =>
    loadVisibleColumns(RUN_COLUMN_STORAGE_KEY)
  )
  // Columns the user has already been offered. Anything available but not yet
  // known is a newly-introduced column and gets shown by default once.
  const knownRunColumnsRef = useRef<Set<string>>(
    new Set(loadVisibleColumns(RUN_COLUMN_KNOWN_STORAGE_KEY))
  )
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([])
  const { pins, setPins, togglePin, isPinned } = useRunPins(PIN_STORAGE_KEY)
  const [cvExpandedParents, setCvExpandedParents] = useState<Set<string>>(new Set())
  const { copiedValue: copiedRunId, copyWithFeedback } = useClipboardFeedback()

  const [editingNoteRunId, setEditingNoteRunId] = useState<string | null>(null)
  const [editingNoteDraft, setEditingNoteDraft] = useState('')
  const [editingNoteOriginal, setEditingNoteOriginal] = useState('')
  const [editingNoteSaving, setEditingNoteSaving] = useState(false)

  const [archiveModalOpen, setArchiveModalOpen] = useState(false)
  const [archiveReason, setArchiveReason] = useState('')
  const [archiveSaving, setArchiveSaving] = useState(false)
  const [bulkDeletePending, setBulkDeletePending] = useState(false)

  const [compareModalOpen, setCompareModalOpen] = useState(false)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareData, setCompareData] = useState<CompareResponse | null>(null)

  // Phase 2: details/results overlay state
  const [detailOpen, setDetailOpen] = useState(false)
  const [detailRunId, setDetailRunId] = useState<string>('')
  const [detailMode, setDetailMode] = useState<DetailMode>('match')
  const [detailOutcomeFilter, setDetailOutcomeFilter] = useState<string>('all')
  const [detailSampleType, setDetailSampleType] = useState<string>('all')
  const [detailSource, setDetailSource] = useState<string>('all')
  const [detailResponseFilter, setDetailResponseFilter] = useState<ResponseFilter>('all')
  const [detailQuery, setDetailQuery] = useState('')
  const [detailPage, setDetailPage] = useState(1)
  const [detailPageSize, setDetailPageSize] = useState(25)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [detailData, setDetailData] = useState<ResultListResponse | null>(null)

  const [selectedResultId, setSelectedResultId] = useState<string | null>(null)
  const [selectedResultDetail, setSelectedResultDetail] = useState<ResultDetailResponse | null>(null)
  const [selectedResultLoading, setSelectedResultLoading] = useState(false)
  const [selectedResultError, setSelectedResultError] = useState<string | null>(null)
  const [runTableContainerWidth, setRunTableContainerWidth] = useState(0)

  useEffect(() => {
    const element = runTableContainerRef.current
    if (!element) return

    const updateWidth = () => {
      setRunTableContainerWidth(Math.floor(element.clientWidth))
    }

    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    let cancelled = false

    async function ping(): Promise<void> {
      try {
        await fetchHealth()
        if (!cancelled) setApiHealthy(true)
      } catch {
        if (!cancelled) setApiHealthy(false)
      }
    }

    void ping()
    const timer = setInterval(() => {
      void ping()
    }, 5000)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    const timeout = setTimeout(() => {
      void loadRuns()
    }, 180)
    return () => clearTimeout(timeout)
  }, [includeArchived, evalFilter, query])

  useEffect(() => {
    if (!detailOpen || !detailRunId) return
    const timeout = setTimeout(() => {
      void loadDetailResults()
    }, 180)
    return () => clearTimeout(timeout)
  }, [
    detailOpen,
    detailRunId,
    detailMode,
    detailOutcomeFilter,
    detailSampleType,
    detailSource,
    detailResponseFilter,
    detailQuery,
    detailPage,
    detailPageSize,
  ])

  async function loadRuns(options?: { suppressError?: boolean }): Promise<boolean> {
    setLoading(true)
    setLoadError(null)
    try {
      const res = await fetchRuns({ includeArchived, evalFilter, query })
      setRuns(res.items)
      setSelectedRunIds((prev) => prev.filter((id) => res.items.some((r) => r.run_id === id)))
      setApiHealthy(true)
      return true
    } catch (err) {
      console.error(err)
      setApiHealthy(false)
      if (!options?.suppressError) {
        setLoadError(readErrorDetail(err, 'Failed to load FIM runs'))
        message.error('Failed to load FIM runs')
      }
      return false
    } finally {
      setLoading(false)
    }
  }

  function healthPid(snapshot: HealthResponse | null | undefined): number | null {
    const pid = snapshot?.pid
    return typeof pid === 'number' && Number.isFinite(pid) ? pid : null
  }

  async function waitForRestartedBackend(beforePid: number | null): Promise<boolean> {
    let sawUnavailable = false
    for (let i = 0; i < 45; i += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 300))
      try {
        const health = await fetchHealth()
        const currentPid = healthPid(health)
        if (beforePid !== null && currentPid !== null) {
          if (currentPid !== beforePid) return true
          continue
        }
        if (beforePid === null && currentPid !== null) return true
        if (sawUnavailable) return true
      } catch {
        sawUnavailable = true
      }
    }
    return false
  }

  async function refreshRunsAfterRestart(): Promise<boolean> {
    for (let i = 0; i < 6; i += 1) {
      const ok = await loadRuns({ suppressError: true })
      if (ok) return true
      await new Promise((resolve) => window.setTimeout(resolve, 300))
    }
    return loadRuns()
  }

  async function onRestartBackend(): Promise<void> {
    if (restartPending) return
    setRestartPending(true)
    const msgKey = 'fim-server-restart'
    message.loading({ content: 'Restarting backend...', key: msgKey, duration: 0 })
    try {
      let beforePid: number | null = null
      try {
        beforePid = healthPid(await fetchHealth())
      } catch {
        // Backend may already be unavailable; restart flow below handles this.
      }

      await restartBackend()
      const healthy = await waitForRestartedBackend(beforePid)

      if (healthy) {
        message.success({ content: 'Backend restarted', key: msgKey, duration: 2.5 })
      } else {
        message.warning({ content: 'Restart requested but backend is still unavailable', key: msgKey, duration: 4 })
      }

      await refreshRunsAfterRestart()
    } catch (err) {
      console.error(err)
      message.error({ content: readErrorDetail(err, 'Failed to restart backend'), key: msgKey, duration: 4 })
    } finally {
      setRestartPending(false)
    }
  }

  async function loadDetailResults(): Promise<void> {
    if (!detailRunId) return
    setDetailLoading(true)
    setDetailError(null)
    try {
      const res = await fetchRunResults({
        runId: detailRunId,
        mode: detailMode,
        outcomeFilter: detailOutcomeFilter,
        sampleType: detailSampleType,
        source: detailSource,
        responseFilter: detailResponseFilter,
        query: detailQuery,
        page: detailPage,
        pageSize: detailPageSize,
      })
      setDetailData(res)
    } catch (err) {
      console.error(err)
      setDetailError(readErrorDetail(err, 'Failed to load run results'))
      message.error('Failed to load run results')
    } finally {
      setDetailLoading(false)
    }
  }

  const repoColumns = useMemo(() => {
    const seen = new Set<string>()
    const out: string[] = []
    for (const run of runs) {
      for (const entry of run.repo_scores || []) {
        const repo = String(entry.repo || '').trim()
        if (!repo || seen.has(repo)) continue
        seen.add(repo)
        out.push(repo)
      }
      for (const repo of run.train_repos || []) {
        const name = String(repo || '').trim()
        if (!name || seen.has(name)) continue
        seen.add(name)
        out.push(name)
      }
    }
    out.sort((a, b) => a.localeCompare(b))
    return out
  }, [runs])

  const anyRunHasLintData = useMemo(
    () =>
      runs.some(
        (run) =>
          run.lint_pass_rate !== null ||
          run.lint_in_domain_pass_rate !== null ||
          run.lint_out_of_domain_pass_rate !== null
      ),
    [runs]
  )

  // The lint toggle/columns are dead ends when no loaded run has any lint field
  // populated (true for all shipped runs); gate on this rather than raw
  // `showLintColumns` so persisted state from before this data shipped can't
  // resurrect the columns.
  const lintColumnsActive = showLintColumns && anyRunHasLintData

  const pillHueByKey = useMemo(() => buildPillHueByKey(runs), [runs])

  const modelFilterOptions = useMemo(() => {
    const seen = new Set<string>()
    const out: Array<{ value: string; label: string }> = [{ value: 'all', label: 'All models' }]
    for (const run of runs) {
      const label = runModelLabel(run)
      if (seen.has(label)) continue
      seen.add(label)
      out.push({ value: label, label })
    }
    out.sort((a, b) => {
      if (a.value === 'all') return -1
      if (b.value === 'all') return 1
      return a.label.localeCompare(b.label)
    })
    return out
  }, [runs])

  const modelFilterLabelByValue = useMemo(() => {
    const out = new Map<string, string>()
    for (const option of modelFilterOptions) out.set(option.value, option.label)
    return out
  }, [modelFilterOptions])

  const modelFilterMenuItems = useMemo(
    () => modelFilterOptions.map((option) => ({ key: option.value, label: option.label })),
    [modelFilterOptions]
  )

  const cvParentHasSingleOodCoverage = useMemo(() => {
    type CoverageAcc = {
      subrunIds: Set<string>
      oodCounts: Map<string, number>
      missingOodDomainLabels: boolean
    }
    const parentById = new Map<string, RunListItem>()
    const byParentId = new Map<string, CoverageAcc>()

    for (const row of runs) {
      if (row.is_cv_parent) parentById.set(row.run_id, row)

      const parentId = String(row.cv_parent_run_id || '').trim()
      if (!row.is_cv_subrun || !parentId) continue

      let acc = byParentId.get(parentId)
      if (!acc) {
        acc = {
          subrunIds: new Set<string>(),
          oodCounts: new Map<string, number>(),
          missingOodDomainLabels: false,
        }
        byParentId.set(parentId, acc)
      }

      acc.subrunIds.add(row.run_id)
      const oodRepos = new Set<string>()
      for (const entry of row.repo_scores || []) {
        const repo = String(entry.repo || '').trim()
        if (!repo) continue
        const domain = String(entry.domain || '').trim().toLowerCase()
        if (domain === 'out_of_domain') {
          oodRepos.add(repo)
          continue
        }
        if (entry.in_domain === false) oodRepos.add(repo)
      }

      if (oodRepos.size === 0) {
        acc.missingOodDomainLabels = true
        continue
      }

      for (const repo of oodRepos) {
        acc.oodCounts.set(repo, (acc.oodCounts.get(repo) ?? 0) + 1)
      }
    }

    const out = new Map<string, boolean>()
    for (const [parentId, acc] of byParentId.entries()) {
      const parent = parentById.get(parentId)
      const expectedSubruns = parent?.cv_subrun_count ?? acc.subrunIds.size
      if (
        expectedSubruns < 2 ||
        acc.subrunIds.size !== expectedSubruns ||
        acc.missingOodDomainLabels ||
        acc.oodCounts.size === 0
      ) {
        out.set(parentId, false)
        continue
      }

      let allExactlyOnce = true
      for (const count of acc.oodCounts.values()) {
        if (count !== 1) {
          allExactlyOnce = false
          break
        }
      }
      out.set(parentId, allExactlyOnce)
    }
    return out
  }, [runs])

  /**
   * Compute fallback U OOD values for CV parents that fail single-OOD coverage
   * specifically due to accidental OOD relabeling from training sparsity.
   *
   * The adjusted value treats duplicate OOD appearances as comparison noise and
   * averages expected holdout-repo OOD rates only.
   */
  const cvParentUOodAdjustment = useMemo(() => {
    type ParentAcc = {
      subrunIds: Set<string>
      oodCounts: Map<string, number>
      accidentalOodCounts: Map<string, number>
      intentionalOodRateSamples: Map<string, number[]>
      expectedHoldoutRepos: Set<string>
      missingOodDomainLabels: boolean
    }

    const parentById = new Map<string, RunListItem>()
    const byParentId = new Map<string, ParentAcc>()

    for (const row of runs) {
      if (row.is_cv_parent) parentById.set(row.run_id, row)

      const parentId = String(row.cv_parent_run_id || '').trim()
      if (!row.is_cv_subrun || !parentId) continue

      let acc = byParentId.get(parentId)
      if (!acc) {
        acc = {
          subrunIds: new Set<string>(),
          oodCounts: new Map<string, number>(),
          accidentalOodCounts: new Map<string, number>(),
          intentionalOodRateSamples: new Map<string, number[]>(),
          expectedHoldoutRepos: new Set<string>(),
          missingOodDomainLabels: false,
        }
        byParentId.set(parentId, acc)
      }

      acc.subrunIds.add(row.run_id)
      const holdoutSet = new Set((row.cv_holdout_repos || []).map((repo) => String(repo || '').trim()).filter(Boolean))
      for (const holdout of holdoutSet) acc.expectedHoldoutRepos.add(holdout)
      const oodReposInSubrun = new Set<string>()

      for (const entry of row.repo_scores || []) {
        const repo = String(entry.repo || '').trim()
        if (!repo) continue

        const domain = String(entry.domain || '').trim().toLowerCase()
        const isOod = domain === 'out_of_domain' || entry.in_domain === false
        if (!isOod) continue

        oodReposInSubrun.add(repo)
        const isExpectedHoldout = holdoutSet.has(repo)
        if (!isExpectedHoldout) {
          acc.accidentalOodCounts.set(repo, (acc.accidentalOodCounts.get(repo) ?? 0) + 1)
        }
        if (
          isExpectedHoldout &&
          typeof entry.exact_match_rate === 'number' &&
          Number.isFinite(entry.exact_match_rate)
        ) {
          const samples = acc.intentionalOodRateSamples.get(repo) ?? []
          samples.push(entry.exact_match_rate)
          acc.intentionalOodRateSamples.set(repo, samples)
        }
      }

      if (oodReposInSubrun.size === 0) {
        acc.missingOodDomainLabels = true
        continue
      }

      for (const repo of oodReposInSubrun) {
        acc.oodCounts.set(repo, (acc.oodCounts.get(repo) ?? 0) + 1)
      }
    }

    const out = new Map<string, CvParentUOodAdjustment>()
    for (const [parentId, acc] of byParentId.entries()) {
      const parent = parentById.get(parentId)
      const expectedSubruns = parent?.cv_subrun_count ?? acc.subrunIds.size
      if (
        expectedSubruns < 2 ||
        acc.subrunIds.size !== expectedSubruns ||
        acc.missingOodDomainLabels ||
        acc.oodCounts.size === 0
      ) {
        continue
      }

      // A duplicate here means the same repo became OOD in multiple folds.
      const duplicateOodRepos = Array.from(acc.oodCounts.entries())
        .filter(([, count]) => count > 1)
        .map(([repo]) => repo)
        .sort((a, b) => a.localeCompare(b))
      if (duplicateOodRepos.length === 0) continue
      const accidentalOodRepos = Array.from(acc.accidentalOodCounts.keys()).sort((a, b) => a.localeCompare(b))
      const duplicateAccidentalOodRepos = duplicateOodRepos.filter((repo) => (acc.accidentalOodCounts.get(repo) ?? 0) > 0)
      if (duplicateAccidentalOodRepos.length === 0) continue

      // "As if accidental OOD repos were ID": compute U OOD from expected holdouts only.
      const expectedHoldoutRepoMeans: number[] = []
      for (const repo of acc.expectedHoldoutRepos) {
        const samples = acc.intentionalOodRateSamples.get(repo) ?? []
        if (samples.length === 0) continue
        expectedHoldoutRepoMeans.push(samples.reduce((sum, value) => sum + value, 0) / samples.length)
      }
      if (expectedHoldoutRepoMeans.length === 0) continue

      const adjustedRate = expectedHoldoutRepoMeans.reduce((sum, value) => sum + value, 0) / expectedHoldoutRepoMeans.length
      out.set(parentId, {
        adjustedRate,
        duplicateOodRepos,
        accidentalOodRepos,
      })
    }
    return out
  }, [runs])

  // The red outline marks the cell to compare across runs: Unweighted Total for
  // non-CV runs, Unweighted OOD for CV parents whose folds cleanly cover each
  // repo as OOD exactly once. CV rows without clean coverage get no outline —
  // their totals mix ID and OOD scores.
  function shouldEmphasizeUTotal(row: DisplayRunRow): boolean {
    return row.eval_kind !== 'CV' && !row._cv_is_child
  }

  function shouldEmphasizeUOod(row: DisplayRunRow): boolean {
    return (
      !row._cv_is_child &&
      row.eval_kind === 'CV' &&
      row.is_cv_parent &&
      cvParentHasSingleOodCoverage.get(row.run_id) === true
    )
  }

  function emphasisExplainer(row: DisplayRunRow): string {
    if (row.eval_kind === 'CV') {
      return (
        'The outline marks this run’s comparison number: the mean of the folds’ ' +
        'out-of-domain averages — each repo is scored by the fold that held it out ' +
        'of training, so nothing is scored in-domain.'
      )
    }
    if (row.eval_kind === 'OOD') {
      return (
        'The outline marks this run’s comparison number: the unweighted average over ' +
        'its repos, all out-of-domain — the model was trained on none of them.'
      )
    }
    return (
      'The outline marks this run’s comparison number: the unweighted average over ' +
      'all its repos, each repo weighted equally.'
    )
  }

  useEffect(() => {
    if (modelFilter === 'all') return
    const stillAvailable = modelFilterOptions.some((option) => option.value === modelFilter)
    if (!stillAvailable) setModelFilter('all')
  }, [modelFilter, modelFilterOptions])

  // Sorting is done here rather than by antd: pinning and CV grouping are
  // structural invariants, the sort is a refinement inside them. Pinned rows
  // stay in a top section, fold rows stay attached under their parent; the
  // active sort orders top-level rows within each section (and folds among
  // their siblings).
  const runsForDisplay = useMemo(() => {
    const now = Date.now()
    const modelFiltered = modelFilter === 'all'
      ? runs
      : runs.filter((run) => runModelLabel(run) === modelFilter)
    const baseRows = ageFilter === 'all'
      ? modelFiltered
      : modelFiltered.filter((run) => runMatchesAgeFilter(run, ageFilter, now))

    const runIdSet = new Set(baseRows.map((r) => r.run_id))

    const childrenByParent = new Map<string, RunListItem[]>()
    const unsortedTopRows: RunListItem[] = []
    for (const row of baseRows) {
      const parentId = String(row.cv_parent_run_id || '').trim()
      const isSubrun = Boolean(row.is_cv_subrun || parentId)
      if (isSubrun && parentId && runIdSet.has(parentId) && !pins.has(row.run_id)) {
        const bucket = childrenByParent.get(parentId) ?? []
        bucket.push(row)
        childrenByParent.set(parentId, bucket)
      } else {
        unsortedTopRows.push(row)
      }
    }

    const activeComparator = runSort ? runComparatorForKey(runSort.key) : null
    const direction = runSort?.order === 'descend' ? -1 : 1
    const sortedTopRows = activeComparator
      ? [...unsortedTopRows].sort((a, b) => direction * activeComparator(a, b))
      : unsortedTopRows

    for (const [parentId, children] of childrenByParent.entries()) {
      children.sort((a, b) => {
        if (activeComparator) {
          const byMetric = direction * activeComparator(a, b)
          if (byMetric !== 0) return byMetric
        }
        const ai = a.cv_subrun_index ?? Number.MAX_SAFE_INTEGER
        const bi = b.cv_subrun_index ?? Number.MAX_SAFE_INTEGER
        if (ai !== bi) return ai - bi
        return a.run_id.localeCompare(b.run_id)
      })
      childrenByParent.set(parentId, children)
    }

    // Stable partition: pinned rows first, each section keeps its sorted order.
    const topRows = orderPinnedRuns(sortedTopRows, pins)

    const out: DisplayRunRow[] = []
    for (const row of topRows) {
      const children = childrenByParent.get(row.run_id) ?? []
      const expandable = children.length > 0
      const expanded = expandable && cvExpandedParents.has(row.run_id)
      out.push({
        ...row,
        _cv_is_child: false,
        _cv_expandable: expandable,
        _cv_is_expanded: expanded,
      })
      if (expanded) {
        for (const child of children) {
          out.push({
            ...child,
            _cv_is_child: true,
            _cv_expandable: false,
            _cv_is_expanded: false,
          })
        }
      }
    }
    return out
  }, [runs, pins, cvExpandedParents, modelFilter, ageFilter, runSort])

  const lastPinnedRunId = useLastPinnedRunId(runsForDisplay, pins)

  // Syntax Mode (and the Syntax results column) only makes sense for runs that
  // were evaluated with the Daml toolchain and have a syntax_pass_rate; all
  // shipped runs leave it null, so hide the dead end rather than showing an
  // always-empty mode.
  const detailRun = useMemo(
    () => runs.find((run) => run.run_id === detailRunId) ?? null,
    [runs, detailRunId]
  )
  const detailHasSyntax = detailRun?.syntax_pass_rate !== null && detailRun?.syntax_pass_rate !== undefined
  const detailModeOptions = useMemo(
    () =>
      detailHasSyntax
        ? [
            { value: 'match', label: 'Match Mode' },
            { value: 'syntax', label: 'Syntax Mode' },
          ]
        : [{ value: 'match', label: 'Match Mode' }],
    [detailHasSyntax]
  )

  useEffect(() => {
    if (!detailHasSyntax && detailMode === 'syntax') {
      setDetailMode('match')
      setDetailOutcomeFilter('all')
      setDetailPage(1)
    }
  }, [detailHasSyntax, detailMode])

  const outcomeOptions = detailMode === 'match' ? MATCH_OUTCOME_OPTIONS : SYNTAX_OUTCOME_OPTIONS
  const detailSampleTypeOptions = useMemo(
    () => [
      { value: 'all', label: 'All categories' },
      ...((detailData?.sample_types || []).map((s) => ({ value: s, label: s }))),
    ],
    [detailData?.sample_types]
  )

  const detailSourceOptions = useMemo(
    () => [
      { value: 'all', label: 'All sources' },
      ...((detailData?.sources || []).map((s) => ({ value: s, label: s }))),
    ],
    [detailData?.sources]
  )

  const detailOutcomeLabelByValue = useMemo(() => {
    const out = new Map<string, string>()
    for (const option of outcomeOptions) out.set(option.value, option.label)
    return out
  }, [outcomeOptions])

  const detailSampleTypeLabelByValue = useMemo(() => {
    const out = new Map<string, string>()
    for (const option of detailSampleTypeOptions) out.set(option.value, option.label)
    return out
  }, [detailSampleTypeOptions])

  const detailOutcomeMenuItems = useMemo(
    () => outcomeOptions.map((option) => ({ key: option.value, label: option.label })),
    [outcomeOptions]
  )

  const detailCategoryMenuItems = useMemo(
    () => detailSampleTypeOptions.map((option) => ({ key: option.value, label: option.label })),
    [detailSampleTypeOptions]
  )

  function startInlineNoteEdit(run: RunListItem): void {
    if (editingNoteSaving) return
    const initial = run.notes ?? ''
    setEditingNoteRunId(run.run_id)
    setEditingNoteDraft(initial)
    setEditingNoteOriginal(initial)
  }

  function cancelInlineNoteEdit(): void {
    if (editingNoteSaving) return
    setEditingNoteRunId(null)
    setEditingNoteDraft('')
    setEditingNoteOriginal('')
  }

  async function commitInlineNoteEdit(): Promise<void> {
    if (!editingNoteRunId || editingNoteSaving) return
    if (editingNoteDraft === editingNoteOriginal) {
      cancelInlineNoteEdit()
      return
    }
    setEditingNoteSaving(true)
    try {
      const updated = await saveRunNotes(editingNoteRunId, editingNoteDraft)
      setRuns((prev) => prev.map((r) => (r.run_id === updated.run_id ? updated : r)))
      message.success('Notes saved')
      setEditingNoteRunId(null)
      setEditingNoteDraft('')
      setEditingNoteOriginal('')
    } catch (err) {
      console.error(err)
      message.error('Failed to save notes')
    } finally {
      setEditingNoteSaving(false)
    }
  }

  async function onCopyRunId(runId: string): Promise<void> {
    const copied = await copyWithFeedback(runId)
    if (!copied) {
      message.error('Failed to copy Run ID')
    }
  }

  function openDetailOverlay(run: RunListItem): void {
    setDetailRunId(run.run_id)
    setDetailMode('match')
    setDetailOutcomeFilter('all')
    setDetailSampleType('all')
    setDetailSource('all')
    setDetailResponseFilter('all')
    setDetailQuery('')
    setDetailPage(1)
    setDetailPageSize(25)
    setDetailData(null)
    setSelectedResultId(null)
    setSelectedResultDetail(null)
    setSelectedResultError(null)
    setDetailOpen(true)
  }

  async function loadSelectedResultDetail(resultId: string): Promise<void> {
    if (!detailRunId) return
    setSelectedResultId(resultId)
    setSelectedResultLoading(true)
    setSelectedResultError(null)
    try {
      const detail = await fetchRunResultDetail(detailRunId, resultId)
      setSelectedResultDetail(detail)
    } catch (err) {
      console.error(err)
      setSelectedResultError(readErrorDetail(err, 'Failed to load selected result detail'))
      message.error('Failed to load selected result detail')
    } finally {
      setSelectedResultLoading(false)
    }
  }

  async function onDeleteSelected(): Promise<void> {
    if (selectedRunIds.length === 0) return
    const deleting = [...selectedRunIds]
    setBulkDeletePending(true)
    try {
      await deleteRuns(deleting)
      setSelectedRunIds([])
      setPins((prev) => {
        const next = new Set(prev)
        for (const id of deleting) next.delete(id)
        return next
      })
      setCvExpandedParents((prev) => {
        const next = new Set(prev)
        for (const id of deleting) next.delete(id)
        return next
      })
      message.success('Deleted selected runs')
      await loadRuns()
    } catch (err) {
      console.error(err)
      message.error(readErrorDetail(err, 'Failed to delete selected runs'))
    } finally {
      setBulkDeletePending(false)
    }
  }

  function onArchiveSelected(): void {
    if (selectedRunIds.length === 0) return
    setArchiveModalOpen(true)
  }

  async function submitArchive(): Promise<void> {
    setArchiveSaving(true)
    try {
      await archiveRuns(selectedRunIds, archiveReason)
      setSelectedRunIds([])
      setArchiveReason('')
      setArchiveModalOpen(false)
      message.success('Archived selected runs')
      await loadRuns()
    } catch (err) {
      console.error(err)
      message.error('Failed to archive runs')
    } finally {
      setArchiveSaving(false)
    }
  }

  async function onCompareSelected(): Promise<void> {
    if (selectedRunIds.length !== 2) return
    setCompareModalOpen(true)
    setCompareLoading(true)
    try {
      const data = await compareRuns([selectedRunIds[0], selectedRunIds[1]])
      setCompareData(data)
    } catch (err) {
      console.error(err)
      message.error('Failed to compare runs')
      setCompareModalOpen(false)
    } finally {
      setCompareLoading(false)
    }
  }

  async function onDeleteRow(runId: string): Promise<void> {
    try {
      await deleteRuns([runId])
      setSelectedRunIds((prev) => prev.filter((id) => id !== runId))
      setPins((prev) => {
        const next = new Set(prev)
        next.delete(runId)
        return next
      })
      setCvExpandedParents((prev) => {
        const next = new Set(prev)
        next.delete(runId)
        return next
      })
      message.success(`Deleted run ${runId}`)
      await loadRuns()
    } catch (err) {
      console.error(err)
      message.error(readErrorDetail(err, `Failed to delete run ${runId}`))
    }
  }

  function getTotalAvg(row: RunListItem): number | null {
    if (row.eval_kind === 'ID') return row.in_domain_repo_average ?? row.exact_match_rate
    if (row.eval_kind === 'OOD') return row.out_of_domain_repo_average ?? row.exact_match_rate
    if (row.total_repo_average !== null) return row.total_repo_average
    const repoRates = (row.repo_scores || [])
      .map((entry) => entry.exact_match_rate)
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    if (repoRates.length > 0) {
      return repoRates.reduce((acc, value) => acc + value, 0) / repoRates.length
    }
    const id = row.in_domain_repo_average
    const ood = row.out_of_domain_repo_average
    if (id !== null && ood !== null) return (id + ood) / 2
    return row.exact_match_rate
  }

  function getIdAvg(row: RunListItem): number | null {
    if (row.in_domain_repo_average !== null) return row.in_domain_repo_average
    return row.eval_kind === 'ID' ? row.exact_match_rate : null
  }

  function getOodAvg(row: RunListItem): number | null {
    if (row.out_of_domain_repo_average !== null) return row.out_of_domain_repo_average
    return row.eval_kind === 'OOD' ? row.exact_match_rate : null
  }

  function getRepoRate(row: RunListItem, repo: string): number | null {
    const direct = row.repo_rate_map?.[repo]
    if (typeof direct === 'number') return direct
    const fromScores = (row.repo_scores || []).find((entry) => entry.repo === repo)?.exact_match_rate
    return typeof fromScores === 'number' ? fromScores : null
  }

  function getTotalOvergen(row: RunListItem): number | null {
    if (row.eval_kind === 'ID') return getIdOvergen(row)
    if (row.eval_kind === 'OOD') return getOodOvergen(row)
    if (row.total_repo_prefix_average !== null) return row.total_repo_prefix_average
    const repoRates = Object.values(row.repo_prefix_rate_map || {})
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    if (repoRates.length > 0) {
      return repoRates.reduce((acc, value) => acc + value, 0) / repoRates.length
    }
    return null
  }

  function getIdOvergen(row: RunListItem): number | null {
    if (row.in_domain_repo_prefix_average !== null) return row.in_domain_repo_prefix_average
    return row.eval_kind === 'ID' ? row.prefix_match_rate : null
  }

  function getOodOvergen(row: RunListItem): number | null {
    if (row.out_of_domain_repo_prefix_average !== null) return row.out_of_domain_repo_prefix_average
    return row.eval_kind === 'OOD' ? row.prefix_match_rate : null
  }

  // Mean over repos whose training-data relationship is genuinely unknown —
  // the run declared none for them (e.g. closed models whose training corpus
  // is not disclosed). Repos with any domain label are excluded, including CV
  // parents' "mixed" repos: those are known (ID in one fold, OOD in another),
  // just not reducible to a single ID/OOD label.
  function undisclosedRepoMean(
    row: RunListItem,
    rates: Record<string, number> | undefined
  ): number | null {
    const domainByRepo = new Map(
      (row.repo_scores || []).map((entry) => [entry.repo, String(entry.domain || '').toLowerCase()])
    )
    const values = Object.entries(rates || {})
      .filter(([repo]) => {
        // repo_domain_map is the backend's full ID/OOD classification (domain
        // labels, train_repos, in_domain booleans) — anything it classifies is
        // not undisclosed, regardless of the raw domain string.
        if ((row.repo_domain_map || {})[repo]) return false
        const domain = domainByRepo.get(repo)
        return !domain || domain === 'unknown'
      })
      .map(([, rate]) => rate)
      .filter((value) => Number.isFinite(value))
    if (values.length === 0) return null
    return values.reduce((acc, value) => acc + value, 0) / values.length
  }

  function getUndisclosedAvg(row: RunListItem): number | null {
    return undisclosedRepoMean(row, row.repo_rate_map)
  }

  function getUndisclosedOvergen(row: RunListItem): number | null {
    return undisclosedRepoMean(row, row.repo_prefix_rate_map)
  }

  // The value the red outline points at in the runs table, with its name — or
  // null for CV runs without clean single-OOD fold coverage, which have no
  // honest single comparison number.
  function comparisonMetricFor(row: RunListItem): { value: number | null; label: string } | null {
    if (row.eval_kind === 'CV') {
      if (row.is_cv_parent && cvParentHasSingleOodCoverage.get(row.run_id) === true) {
        return { value: getOodAvg(row), label: 'Unweighted OOD' }
      }
      return null
    }
    return { value: getTotalAvg(row), label: 'Unweighted Total' }
  }

  function buildCompareRows(data: CompareResponse): CompareMetricRow[] {
    const { left, right } = data
    const rows: CompareMetricRow[] = []
    const formatDelta = (l: number | null, r: number | null): string => {
      if (l === null || r === null) return '—'
      const pp = (r - l) * 100
      return `${pp >= 0 ? '+' : ''}${pp.toFixed(1)}pp`
    }
    const push = (
      key: string,
      metric: string,
      l: number | null,
      r: number | null,
      emphasized = false
    ) => {
      if (l === null && r === null) return
      rows.push({
        key,
        metric,
        left: formatPercent(l),
        right: formatPercent(r),
        delta: formatDelta(l, r),
        emphasized,
      })
    }

    const leftCmp = comparisonMetricFor(left)
    const rightCmp = comparisonMetricFor(right)
    const cmpLabel =
      leftCmp && rightCmp && leftCmp.label !== rightCmp.label
        ? `Comparison number (left: ${leftCmp.label}, right: ${rightCmp.label})`
        : `Comparison number (${(leftCmp ?? rightCmp)?.label ?? '—'})`
    push('comparison', cmpLabel, leftCmp?.value ?? null, rightCmp?.value ?? null, true)

    push('weighted_total', 'Weighted Total', left.exact_match_rate, right.exact_match_rate)
    push('unweighted_total', 'Unweighted Total', getTotalAvg(left), getTotalAvg(right))
    push('unweighted_id', 'Unweighted ID', getIdAvg(left), getIdAvg(right))
    push('unweighted_ood', 'Unweighted OOD', getOodAvg(left), getOodAvg(right))
    push(
      'unweighted_undisclosed',
      'Unweighted Undisclosed',
      getUndisclosedAvg(left),
      getUndisclosedAvg(right)
    )
    push('overgen', 'Over-generation (weighted)', left.prefix_match_rate, right.prefix_match_rate)
    push('syntax', 'Syntax Pass', left.syntax_pass_rate, right.syntax_pass_rate)
    push('lint', 'Lint Pass', left.lint_pass_rate, right.lint_pass_rate)
    return rows
  }

  // Single source of the per-column sort logic; the table's sorters are set to
  // `true` and the actual ordering happens in runsForDisplay so pinning and CV
  // grouping survive an active sort.
  function runComparatorForKey(key: string): ((a: RunListItem, b: RunListItem) => number) | null {
    if (key.startsWith('repo:')) {
      const repo = key.slice('repo:'.length)
      return (a, b) => numericSort(getRepoRate(a, repo), getRepoRate(b, repo))
    }
    switch (key) {
      case 'run_name':
        return (a, b) => (a.run_name ?? '').localeCompare(b.run_name ?? '')
      case 'run_id':
        return (a, b) => a.run_id.localeCompare(b.run_id)
      case 'eval_kind':
        return (a, b) => a.eval_kind.localeCompare(b.eval_kind)
      case 'age':
        return (a, b) =>
          numericSort(parseRunTimestamp(a)?.getTime() ?? null, parseRunTimestamp(b)?.getTime() ?? null)
      case 'model':
        return (a, b) => runModelLabel(a).localeCompare(runModelLabel(b))
      case 'version':
        return (a, b) => (a.dataset_version ?? '').localeCompare(b.dataset_version ?? '')
      case 'examples':
        return (a, b) => numericSort(a.total_examples, b.total_examples)
      case 'weighted_total_avg':
        return (a, b) => numericSort(a.exact_match_rate, b.exact_match_rate)
      case 'total_avg':
        return (a, b) => numericSort(getTotalAvg(a), getTotalAvg(b))
      case 'id_avg':
        return (a, b) => numericSort(getIdAvg(a), getIdAvg(b))
      case 'ood_avg':
        return (a, b) => numericSort(getOodAvg(a), getOodAvg(b))
      case 'undisclosed_avg':
        return (a, b) => numericSort(getUndisclosedAvg(a), getUndisclosedAvg(b))
      case 'lint_pass_rate':
        return (a, b) => numericSort(a.lint_pass_rate, b.lint_pass_rate)
      case 'lint_id':
        return (a, b) => numericSort(a.lint_in_domain_pass_rate, b.lint_in_domain_pass_rate)
      case 'lint_ood':
        return (a, b) => numericSort(a.lint_out_of_domain_pass_rate, b.lint_out_of_domain_pass_rate)
      default:
        return null
    }
  }

  /**
   * Render U OOD with an optional second badge:
   * - `Obs`: observed fold-mean U OOD from summary.
   * - `Adj`: comparison-oriented fallback when missing holdout sampling caused
   *   duplicate OOD coverage (for easier apples-to-apples run comparisons).
   */
  function renderUOodCell(row: DisplayRunRow): ReactNode {
    const observedRate = getOodAvg(row)
    const observedOvergen = getOodOvergen(row)
    const adjustment = cvParentUOodAdjustment.get(row.run_id)
    // Show only on the CV parent rows that would otherwise be highlight candidates.
    const showAdjustment = Boolean(
      adjustment &&
      row.eval_kind === 'CV' &&
      row.is_cv_parent &&
      cvParentHasSingleOodCoverage.get(row.run_id) !== true
    )

    if (!adjustment || !showAdjustment) {
      const cell = scoreCell(observedRate, observedOvergen, 'ood', 1)
      if (!shouldEmphasizeUOod(row)) return cell
      return (
        <Tooltip title={emphasisExplainer(row)} mouseEnterDelay={0.2}>
          <div>{cell}</div>
        </Tooltip>
      )
    }

    const tooltipTitle = (
      <div className="score-adjust-tooltip">
        <div>
          Alternate CV OOD average is shown because this parent has duplicate OOD repo coverage: a repo in a
          fold&apos;s training split had no examples sampled into training, so that fold evaluated it as OOD.
        </div>
        <div>
          `Obs` = current fold-mean U OOD. `Adj` = adjusted U OOD using only expected holdout repos per fold
          (treating accidental OOD appearances as ID for comparison).
        </div>
        <div>Accidental OOD repos: {adjustment.accidentalOodRepos.join(', ') || '—'}</div>
        <div>Duplicate OOD repos: {adjustment.duplicateOodRepos.join(', ') || '—'}</div>
      </div>
    )

    return (
      <Tooltip title={tooltipTitle} mouseEnterDelay={0} mouseLeaveDelay={0}>
        <div className="score-cell-wrap score-cell-dual-badge">
          <div className="score-badge-row">
            <span className="score-badge-label">Obs</span>
            {scoreBadge(observedRate, 'ood', 1)}
          </div>
          <div className="score-badge-row">
            <span className="score-badge-label">Adj</span>
            {scoreBadge(adjustment.adjustedRate, 'ood', 1)}
          </div>
          {observedOvergen !== null && !Number.isNaN(observedOvergen) && (
            <span className="score-subtext">{`+${Math.round(observedOvergen * 100)}%`}</span>
          )}
        </div>
      </Tooltip>
    )
  }

  function getRepoOvergen(row: RunListItem, repo: string): number | null {
    const direct = row.repo_prefix_rate_map?.[repo]
    if (typeof direct === 'number') return direct
    const fromScores = (row.repo_scores || []).find((entry) => entry.repo === repo)?.prefix_match_rate
    return typeof fromScores === 'number' ? fromScores : null
  }

  function getRepoRimKind(row: RunListItem, repo: string): 'id' | 'ood' | undefined {
    if (row.eval_kind === 'ID') return 'id'
    if (row.eval_kind === 'OOD') return 'ood'
    const kind = row.repo_domain_map?.[repo]
    if (kind === 'id' || kind === 'ood') return kind
    return undefined
  }

  const visibleRunColumnKeySet = useMemo(() => {
    if (visibleRunColumns.length === 0) return null
    return new Set(visibleRunColumns)
  }, [visibleRunColumns])

  const flexRunColumnWidths = useMemo(() => {
    const widths: Record<FlexRunColumnKey, number> = {
      run_name: FLEX_RUN_COLUMN_SPECS.run_name.base,
      notes: FLEX_RUN_COLUMN_SPECS.notes.base,
      run_id: FLEX_RUN_COLUMN_SPECS.run_id.base,
      model: FLEX_RUN_COLUMN_SPECS.model.base,
    }

    const isVisible = (key: string): boolean => !visibleRunColumnKeySet || visibleRunColumnKeySet.has(key)
    const activeFlexKeys = (Object.keys(FLEX_RUN_COLUMN_SPECS) as FlexRunColumnKey[]).filter((key) => isVisible(key))
    if (activeFlexKeys.length === 0) return widths

    let nonFlexVisibleWidth = 0
    const addVisibleWidth = (key: string, width: number) => {
      if (isVisible(key)) nonFlexVisibleWidth += width
    }

    addVisibleWidth('pin', STATIC_RUN_COLUMN_WIDTHS.pin)
    addVisibleWidth('age', STATIC_RUN_COLUMN_WIDTHS.age)
    addVisibleWidth('eval_kind', STATIC_RUN_COLUMN_WIDTHS.eval_kind)
    addVisibleWidth('version', STATIC_RUN_COLUMN_WIDTHS.version)
    addVisibleWidth('examples', STATIC_RUN_COLUMN_WIDTHS.examples)
    addVisibleWidth('weighted_total_avg', STATIC_RUN_COLUMN_WIDTHS.weighted_total_avg)
    addVisibleWidth('total_avg', STATIC_RUN_COLUMN_WIDTHS.total_avg)
    addVisibleWidth('id_avg', STATIC_RUN_COLUMN_WIDTHS.id_avg)
    addVisibleWidth('ood_avg', STATIC_RUN_COLUMN_WIDTHS.ood_avg)
    addVisibleWidth('undisclosed_avg', STATIC_RUN_COLUMN_WIDTHS.undisclosed_avg)
    addVisibleWidth('actions', STATIC_RUN_COLUMN_WIDTHS.actions)

    if (showRepoDetails) {
      for (const repo of repoColumns) {
        if (isVisible(`repo:${repo}`)) nonFlexVisibleWidth += STATIC_RUN_COLUMN_WIDTHS.repo
      }
    }

    if (lintColumnsActive) {
      addVisibleWidth('lint_pass_rate', STATIC_RUN_COLUMN_WIDTHS.lint_pass_rate)
      addVisibleWidth('lint_id', STATIC_RUN_COLUMN_WIDTHS.lint_id)
      addVisibleWidth('lint_ood', STATIC_RUN_COLUMN_WIDTHS.lint_ood)
    }

    const baseFlexWidth = activeFlexKeys.reduce((sum, key) => sum + FLEX_RUN_COLUMN_SPECS[key].base, 0)
    const availableFlexWidth = runTableContainerWidth > 0
      ? Math.max(runTableContainerWidth - RUN_TABLE_SELECTION_WIDTH - nonFlexVisibleWidth, 0)
      : baseFlexWidth

    if (availableFlexWidth >= baseFlexWidth) {
      const expandCapacity = activeFlexKeys.reduce(
        (sum, key) => sum + (FLEX_RUN_COLUMN_SPECS[key].max - FLEX_RUN_COLUMN_SPECS[key].base),
        0
      )
      const expandRatio = expandCapacity > 0
        ? Math.min(1, (availableFlexWidth - baseFlexWidth) / expandCapacity)
        : 0
      for (const key of activeFlexKeys) {
        const spec = FLEX_RUN_COLUMN_SPECS[key]
        widths[key] = Math.round(spec.base + (spec.max - spec.base) * expandRatio)
      }
      return widths
    }

    const shrinkCapacity = activeFlexKeys.reduce(
      (sum, key) => sum + (FLEX_RUN_COLUMN_SPECS[key].base - FLEX_RUN_COLUMN_SPECS[key].min),
      0
    )
    const shrinkRatio = shrinkCapacity > 0
      ? Math.min(1, (baseFlexWidth - availableFlexWidth) / shrinkCapacity)
      : 0
    for (const key of activeFlexKeys) {
      const spec = FLEX_RUN_COLUMN_SPECS[key]
      widths[key] = Math.round(spec.base - (spec.base - spec.min) * shrinkRatio)
    }

    return widths
  }, [repoColumns, runTableContainerWidth, lintColumnsActive, showRepoDetails, visibleRunColumnKeySet])

  const columns: ColumnsType<DisplayRunRow> = [
    {
      title: 'Pin',
      key: 'pin',
      dataIndex: 'pin',
      width: STATIC_RUN_COLUMN_WIDTHS.pin,
      fixed: 'left',
      render: (_, row) => (
        <Button
          size="small"
          onClick={() => togglePin(row.run_id)}
          aria-label={isPinned(row.run_id) ? 'Unpin run' : 'Pin run'}
        >
          {isPinned(row.run_id) ? '★' : '☆'}
        </Button>
      ),
    },
    {
      title: 'Run Name',
      key: 'run_name',
      dataIndex: 'run_name',
      width: flexRunColumnWidths.run_name,
      fixed: 'left',
      render: (_, row) => {
        const runName = row.run_name || '—'
        const leading = row._cv_expandable ? (
          <Button
            size="small"
            onClick={() =>
              setCvExpandedParents((prev) => {
                const next = new Set(prev)
                if (next.has(row.run_id)) next.delete(row.run_id)
                else next.add(row.run_id)
                return next
              })
            }
            title={row._cv_is_expanded ? 'Collapse CV folds' : 'Expand CV folds'}
          >
            {row._cv_is_expanded ? '▾' : '▸'}
          </Button>
        ) : (
          <span className="cv-toggle-placeholder">{row._cv_is_child ? '↳' : ''}</span>
        )
        return (
          <RunNameCell
            runId={row.run_id}
            runName={runName}
            copiedRunId={copiedRunId}
            onCopyRunId={onCopyRunId}
            leading={leading}
            runNameClassName={row._cv_is_child ? 'cv-child-run-id' : undefined}
            tags={[
              ...(row.archived
                ? [{ key: 'archived', label: 'Archived', color: 'gold' }]
                : []),
              ...(row.in_progress
                ? [{ key: 'in_progress', label: 'In Progress', color: 'processing' }]
                : []),
              ...(row._cv_is_child && row.cv_subrun_index !== null
                ? [{ key: 'fold', label: `Fold ${row.cv_subrun_index}` }]
                : []),
            ]}
          />
        )
      },
    },
    {
      title: 'Notes',
      key: 'notes',
      dataIndex: 'notes',
      width: flexRunColumnWidths.notes,
      render: (_, row) => {
        const isEditing = editingNoteRunId === row.run_id
        if (isEditing) {
          return (
            <Input
              className="notes-inline-input"
              autoFocus
              disabled={editingNoteSaving}
              value={editingNoteDraft}
              onChange={(e) => setEditingNoteDraft(e.target.value)}
              onBlur={() => {
                void commitInlineNoteEdit()
              }}
              onPressEnter={() => {
                void commitInlineNoteEdit()
              }}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  e.preventDefault()
                  cancelInlineNoteEdit()
                }
              }}
              placeholder="Run notes"
            />
          )
        }
        const rawNote = (row.notes || '').trim()
        const noteText = rawNote || '—'
        return (
          <Tooltip
            title={rawNote ? <span className="notes-tooltip-text">{rawNote}</span> : null}
            mouseEnterDelay={0}
            mouseLeaveDelay={0}
            destroyTooltipOnHide
          >
            <span
              className="notes-cell notes-cell-clickable"
              onClick={() => startInlineNoteEdit(row)}
            >
              {noteText}
            </span>
          </Tooltip>
        )
      },
    },
    {
      title: 'Run ID',
      key: 'run_id',
      dataIndex: 'run_id',
      width: flexRunColumnWidths.run_id,
      fixed: 'left',
      render: (value, row) => (
        <Text code className={row._cv_is_child ? 'cv-child-run-id' : undefined}>{value || '—'}</Text>
      ),
    },
    {
      title: (
        <Tooltip
          mouseEnterDelay={0.2}
          title={
            <div>
              How the evaluated repos relate to the model’s training data:
              <ul style={{ margin: '4px 0 0', paddingLeft: 16 }}>
                <li><b>ID</b> — all repos are in-domain (part of the model’s training data).</li>
                <li><b>OOD</b> — all repos are out-of-domain (the model was not trained on them).</li>
                <li><b>Multi</b> — a mix of declared in-domain and out-of-domain repos.</li>
                <li><b>Undisclosed</b> — no known training-data relationship, e.g. closed models whose training corpus is not disclosed.</li>
                <li><b>CV</b> — cross-validated: folds are aggregated so each repo is scored by the fold that held it out.</li>
              </ul>
            </div>
          }
        >
          Eval
        </Tooltip>
      ),
      key: 'eval_kind',
      dataIndex: 'eval_kind',
      width: STATIC_RUN_COLUMN_WIDTHS.eval_kind,
      fixed: 'left',
      render: (value) => {
        const evalTagColor =
          value === 'OOD'
            ? 'red'
            : value === 'CV'
              ? 'geekblue'
              : value === 'Multi'
                ? 'default'
                : value === 'Undisclosed'
                  ? 'gold'
                  : 'blue'
        return <Tag color={evalTagColor} style={{ marginInlineEnd: 0 }}>{value}</Tag>
      },
    },
    {
      title: (
        <Dropdown
          trigger={['click']}
          overlayClassName="detail-filter-menu"
          menu={{
            items: AGE_FILTER_OPTIONS.map((opt) => ({ key: opt.value, label: opt.label })),
            selectable: true,
            selectedKeys: [ageFilter],
            onClick: ({ key, domEvent }) => {
              // Menu renders in a portal but React events still bubble up the
              // component tree to the th's injected sort handler; stop it so
              // filtering doesn't also toggle the column sort.
              domEvent.stopPropagation()
              setAgeFilter(key as AgeFilter)
            },
          }}
        >
          <button
            type="button"
            className="detail-filter-header-btn"
            title="Filter by run age"
            onClick={(e) => e.stopPropagation()}
          >
            When
            {ageFilter !== 'all' && (
              <span className="detail-filter-header-value">
                {AGE_FILTER_OPTIONS.find((opt) => opt.value === ageFilter)?.label}
              </span>
            )}
          </button>
        </Dropdown>
      ),
      key: 'age',
      dataIndex: 'timestamp',
      width: STATIC_RUN_COLUMN_WIDTHS.age,
      render: (_, row) => {
        const date = parseRunTimestamp(row)
        if (!date) return '—'
        return (
          <Tooltip title={date.toLocaleString('en-GB')} mouseEnterDelay={0}>
            <span>{formatRelativeAge(date, Date.now())}</span>
          </Tooltip>
        )
      },
    },
    {
      title: (
        <Dropdown
          trigger={['click']}
          overlayClassName="detail-filter-menu"
          menu={{
            items: modelFilterMenuItems,
            selectable: true,
            selectedKeys: [modelFilter],
            onClick: ({ key, domEvent }) => {
              // Menu renders in a portal but React events still bubble up the
              // component tree to the th's injected sort handler; stop it so
              // filtering doesn't also toggle the column sort.
              domEvent.stopPropagation()
              setModelFilter(String(key))
            },
          }}
        >
          <button
            type="button"
            className="detail-filter-header-btn"
            title="Filter by model"
            onClick={(e) => e.stopPropagation()}
          >
            Model
            {modelFilter !== 'all' && (
              <span className="detail-filter-header-value">
                {modelFilterLabelByValue.get(modelFilter) || modelFilter}
              </span>
            )}
          </button>
        </Dropdown>
      ),
      key: 'model',
      dataIndex: 'base_model',
      width: flexRunColumnWidths.model,
      render: (_, row) => {
        const label = runModelLabel(row)
        const style = pillStyleFromHue(pillHueByKey.get(modelPillKey(row)) ?? 0)
        return (
          <span className="model-pill" style={style} title={row.model ?? ''}>
            {label}
          </span>
        )
      },
    },
    {
      title: 'Eval Dataset',
      key: 'version',
      dataIndex: 'dataset_version',
      width: STATIC_RUN_COLUMN_WIDTHS.version,
      render: (value, row) => {
        if (!value) return '—'
        return (
          <span
            className="dataset-version-pill"
            style={{
              background: row.dataset_version_bg ?? '#ecfeff',
              color: row.dataset_version_fg ?? '#155e75',
              borderColor: row.dataset_version_border ?? '#67e8f9',
            }}
          >
            {value}
          </span>
        )
      },
    },
    {
      title: 'Examples',
      key: 'examples',
      dataIndex: 'total_examples',
      width: STATIC_RUN_COLUMN_WIDTHS.examples,
      render: (value) => value ?? '—',
    },
    {
      title: scoreHeader(
        'Weighted Total',
        '+over-generation',
        'Exact-match rate over all examples pooled together, so repos with more examples contribute proportionally more.'
      ),
      key: 'weighted_total_avg',
      width: STATIC_RUN_COLUMN_WIDTHS.weighted_total_avg,
      render: (_, row) => scoreCell(row.exact_match_rate, row.prefix_match_rate, undefined, 1),
    },
    {
      title: scoreHeader(
        'Unweighted Total',
        '+over-generation',
        'Average of per-repo exact-match rates: every repo counts equally, regardless of how many examples it has.'
      ),
      key: 'total_avg',
      width: STATIC_RUN_COLUMN_WIDTHS.total_avg,
      onCell: (row) => ({
        className: shouldEmphasizeUTotal(row) ? 'score-cell-emphasis-border' : '',
      }),
      render: (_, row) => {
        const cell = scoreCell(getTotalAvg(row), getTotalOvergen(row), undefined, 1)
        if (!shouldEmphasizeUTotal(row)) return cell
        return (
          <Tooltip title={emphasisExplainer(row)} mouseEnterDelay={0.2}>
            <div>{cell}</div>
          </Tooltip>
        )
      },
    },
    {
      title: scoreHeader(
        'Unweighted ID',
        '+over-generation',
        'Unweighted average over in-domain (ID) repos: repos that were part of the model’s training data.'
      ),
      key: 'id_avg',
      width: STATIC_RUN_COLUMN_WIDTHS.id_avg,
      render: (_, row) => scoreCell(getIdAvg(row), getIdOvergen(row), 'id', 1),
    },
    {
      title: scoreHeader(
        'Unweighted OOD',
        '+over-generation',
        'Unweighted average over out-of-domain (OOD) repos: repos the model was not trained on.'
      ),
      key: 'ood_avg',
      width: STATIC_RUN_COLUMN_WIDTHS.ood_avg,
      onCell: (row) => ({
        className: shouldEmphasizeUOod(row) ? 'score-cell-emphasis-border' : '',
      }),
      render: (_, row) => renderUOodCell(row),
    },
    {
      title: scoreHeader(
        'Unweighted Undisclosed',
        '+over-generation',
        'Unweighted average over repos whose training-data membership can’t be determined, e.g. closed models whose training corpus is not disclosed.'
      ),
      key: 'undisclosed_avg',
      width: STATIC_RUN_COLUMN_WIDTHS.undisclosed_avg,
      render: (_, row) => scoreCell(getUndisclosedAvg(row), getUndisclosedOvergen(row), undefined, 1),
    },
  ]

  if (showRepoDetails) {
    for (const [idx, repo] of repoColumns.entries()) {
      const isFirst = idx === 0
      const isLast = idx === repoColumns.length - 1
      const repoCellClass = [
        'repo-detail-cell',
        isFirst ? 'repo-detail-cell-start' : '',
        isLast ? 'repo-detail-cell-end' : '',
      ]
        .filter(Boolean)
        .join(' ')
      const repoHeaderClass = [
        'repo-detail-header',
        isFirst ? 'repo-detail-header-start' : '',
        isLast ? 'repo-detail-header-end' : '',
      ]
        .filter(Boolean)
        .join(' ')

      columns.push({
        title: scoreHeader(repo, '+over-generation'),
        key: `repo:${repo}`,
        width: STATIC_RUN_COLUMN_WIDTHS.repo,
        className: repoCellClass,
        onHeaderCell: () => ({ className: repoHeaderClass }),
        render: (_, row) => scoreCell(getRepoRate(row, repo), getRepoOvergen(row, repo), getRepoRimKind(row, repo), 1),
      })
    }
  }

  if (lintColumnsActive) {
    columns.push(
      {
        title: scoreHeader('Lint Total', 'parseable'),
        dataIndex: 'lint_pass_rate',
        width: STATIC_RUN_COLUMN_WIDTHS.lint_pass_rate,
        render: (value) => scoreBadge(value, undefined, 1),
      },
      {
        title: scoreHeader('Lint ID', 'parseable'),
        key: 'lint_id',
        width: STATIC_RUN_COLUMN_WIDTHS.lint_id,
        render: (_, row) => scoreBadge(row.lint_in_domain_pass_rate, 'id', 1),
      },
      {
        title: scoreHeader('Lint OOD', 'parseable'),
        key: 'lint_ood',
        width: STATIC_RUN_COLUMN_WIDTHS.lint_ood,
        render: (_, row) => scoreBadge(row.lint_out_of_domain_pass_rate, 'ood', 1),
      }
    )
  }

  columns.push(
    {
      title: 'Actions',
      key: 'actions',
      width: STATIC_RUN_COLUMN_WIDTHS.actions,
      fixed: 'right',
      render: (_, row) => (
        <Space direction="vertical" size={4}>
          <Button size="small" type="primary" block onClick={() => openDetailOverlay(row)}>
            Details
          </Button>
          <Popconfirm
            title={`Delete ${row.run_id}?`}
            description="This removes the run from active and archived directories."
            okText="Delete"
            okButtonProps={{ danger: true }}
            onConfirm={() => void onDeleteRow(row.run_id)}
          >
            <Button size="small" danger block>
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    }
  )

  const runColumnOptions = useMemo(() => {
    const seen = new Set<string>()
    const out: Array<{ key: string; label: string }> = []
    const labelByKey: Record<string, string> = {
      pin: 'Pin',
      age: 'When',
      run_id: 'Run ID',
      eval_kind: 'Eval Kind',
      run_name: 'Run Name',
      model: 'Model',
      version: 'Eval Dataset',
      weighted_total_avg: 'Weighted Total Avg',
      total_avg: 'Unweighted Total Avg',
      id_avg: 'Unweighted ID Avg',
      ood_avg: 'Unweighted OOD Avg',
      undisclosed_avg: 'Unweighted Undisclosed Avg',
      lint_pass_rate: 'Lint Total',
      lint_id: 'Lint ID',
      lint_ood: 'Lint OOD',
      examples: 'Examples',
      notes: 'Notes',
      actions: 'Actions',
    }
    for (const col of columns) {
      const key = getColumnKey(col)
      if (!key || seen.has(key)) continue
      seen.add(key)
      const label =
        labelByKey[key] ??
        (key.startsWith('repo:') ? `Repo: ${key.slice(5)}` : key.replaceAll('_', ' '))
      out.push({ key, label })
    }
    return out
  }, [columns])

  useEffect(() => {
    const available = runColumnOptions.map((opt) => opt.key)
    if (available.length === 0) return
    const known = knownRunColumnsRef.current
    const hadKnown = known.size > 0
    // Columns available now but never offered before are newly introduced.
    const newlyAvailable = hadKnown
      ? available.filter((key) => !known.has(key) && !DEFAULT_HIDDEN_RUN_COLUMNS.has(key))
      : []
    let knownChanged = false
    for (const key of available) {
      if (!known.has(key)) {
        known.add(key)
        knownChanged = true
      }
    }
    if (knownChanged && typeof window !== 'undefined') {
      window.localStorage.setItem(RUN_COLUMN_KNOWN_STORAGE_KEY, JSON.stringify([...known]))
    }
    setVisibleRunColumns((prev) => {
      const prevValid = prev.filter((key) => available.includes(key))
      if (prevValid.length === 0) {
        const defaultVisible = available.filter((key) => !DEFAULT_HIDDEN_RUN_COLUMNS.has(key))
        return defaultVisible.length > 0 ? defaultVisible : available
      }
      // Surface newly-introduced columns, preserving their natural column order.
      const merged = newlyAvailable.length > 0
        ? available.filter((key) => prevValid.includes(key) || newlyAvailable.includes(key))
        : prevValid
      if (merged.length === prev.length && merged.every((value, idx) => value === prev[idx])) {
        return prev
      }
      return merged
    })
  }, [runColumnOptions])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(RUN_COLUMN_STORAGE_KEY, JSON.stringify(visibleRunColumns))
  }, [visibleRunColumns])

  const visibleColumns = useMemo(
    () =>
      filterVisibleColumns(columns, visibleRunColumns).map((col) => {
        const key = getColumnKey(col)
        if (!key || !runComparatorForKey(key)) return col
        // `sorter: true` keeps the clickable header + arrows; the actual
        // ordering is applied in runsForDisplay from runSort.
        return { ...col, sorter: true, sortOrder: runSort?.key === key ? runSort.order : null }
      }),
    [columns, visibleRunColumns, runSort]
  )

  // A sort must not outlive its column: hiding the sorted column (Columns
  // picker, repo-detail/lint toggles) would otherwise leave the table ordered
  // by an invisible column with no indicator anywhere.
  useEffect(() => {
    if (!runSort) return
    const stillVisible = visibleColumns.some((col) => getColumnKey(col) === runSort.key)
    if (!stillVisible) setRunSort(null)
  }, [runSort, visibleColumns])

  const runTableScrollX = useMemo(() => {
    const widthFromVisibleColumns = visibleColumns.reduce((sum, col) => {
      const colWidth = typeof col.width === 'number' && Number.isFinite(col.width) ? col.width : 0
      return sum + colWidth
    }, RUN_TABLE_SELECTION_WIDTH)

    if (runTableContainerWidth > 0) {
      return Math.max(widthFromVisibleColumns, runTableContainerWidth)
    }
    return widthFromVisibleColumns
  }, [runTableContainerWidth, visibleColumns])

  function detailOutcomeEmoji(mode: DetailMode, outcome: string): string {
    if (mode === 'syntax') {
      if (outcome === 'pass') return '✅'
      if (outcome === 'fail') return '❌'
      return '⚪'
    }
    if (outcome === 'match') return '✅'
    if (outcome === 'prefix') return '➕'
    return '❌'
  }

  function detailOutcomeColor(mode: DetailMode, outcome: string): string {
    if (mode === 'syntax') {
      if (outcome === 'pass') return 'green'
      if (outcome === 'fail') return 'red'
      return 'default'
    }
    if (outcome === 'match') return 'green'
    if (outcome === 'prefix') return 'geekblue'
    return 'red'
  }

  function detailOutcomeLabel(mode: DetailMode, outcome: string): string {
    if (mode === 'syntax') {
      if (outcome === 'pass') return 'Pass'
      if (outcome === 'fail') return 'Fail'
      return 'Inconclusive'
    }
    if (outcome === 'match') return 'Match'
    if (outcome === 'prefix') return 'Over-gen'
    return 'Mismatch'
  }

  const detailColumns: ColumnsType<ResultListItem> = [
    {
      title: (
        <Dropdown
          trigger={['click']}
          overlayClassName="detail-filter-menu"
          menu={{
            items: detailOutcomeMenuItems,
            selectable: true,
            selectedKeys: [detailOutcomeFilter],
            onClick: ({ key }) => {
              const value = String(key)
              setDetailOutcomeFilter(value)
              setDetailPage(1)
            },
          }}
        >
          <button
            type="button"
            className="detail-filter-header-btn"
            title="Filter by outcome"
          >
            Outcome
            {detailOutcomeFilter !== 'all' && (
              <span className="detail-filter-header-value">
                {detailOutcomeLabelByValue.get(detailOutcomeFilter) || detailOutcomeFilter}
              </span>
            )}
          </button>
        </Dropdown>
      ),
      dataIndex: 'outcome',
      width: 78,
      render: (value) => (
        <Tag color={detailOutcomeColor(detailMode, value)} title={detailOutcomeLabel(detailMode, value)}>
          {detailOutcomeEmoji(detailMode, value)}
        </Tag>
      ),
    },
    {
      title: (
        <Dropdown
          trigger={['click']}
          overlayClassName="detail-filter-menu"
          menu={{
            items: detailCategoryMenuItems,
            selectable: true,
            selectedKeys: [detailSampleType],
            onClick: ({ key }) => {
              const value = String(key)
              setDetailSampleType(value)
              setDetailPage(1)
            },
          }}
        >
          <button
            type="button"
            className="detail-filter-header-btn"
            title="Filter by category"
          >
            Category
            {detailSampleType !== 'all' && (
              <span className="detail-filter-header-value">
                {detailSampleTypeLabelByValue.get(detailSampleType) || detailSampleType}
              </span>
            )}
          </button>
        </Dropdown>
      ),
      dataIndex: 'category',
      width: 120,
      render: (_, row) => row.category || row.dataset_source || '—',
    },
    {
      title: 'File',
      dataIndex: 'file_path',
      width: 360,
      render: (value) => <span title={value || ''}>{value || '—'}</span>,
    },
    {
      title: 'Target',
      dataIndex: 'target_preview',
      width: 280,
      render: (value) => <span title={value}>{value || '—'}</span>,
    },
    {
      title: 'Response',
      dataIndex: 'response_preview',
      width: 280,
      render: (value) => <span title={value}>{value || '—'}</span>,
    },
    {
      title: 'Exact',
      dataIndex: 'exact_match',
      width: 80,
      render: (value) => (
        <Tag color={value ? 'green' : 'red'}>{value ? 'Yes' : 'No'}</Tag>
      ),
    },
    {
      title: (
        <Tooltip title={OVERGEN_EXPLAINER}>
          Over-gen
        </Tooltip>
      ),
      dataIndex: 'prefix_match',
      width: 90,
      render: (value) => (
        // Neutral colors: "Yes" is a near-miss (not a success) and "No" isn't a
        // failure, so green/red (as used for Exact) would be misleading here.
        <Tag color={value ? 'blue' : 'default'}>{value ? 'Yes' : 'No'}</Tag>
      ),
    },
    ...(detailHasSyntax
      ? [
          {
            title: 'Syntax',
            dataIndex: 'predicted_syntax_passed',
            width: 80,
            render: (value: boolean | null) => {
              if (value === null) return <Tag>—</Tag>
              return <Tag color={value ? 'green' : 'red'}>{value ? 'Pass' : 'Fail'}</Tag>
            },
          },
        ]
      : []),
  ]

  return (
    <div className="app-shell">
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Card>
          <Space wrap size={12}>
            <Tag color={apiHealthy === null ? 'default' : apiHealthy ? 'green' : 'red'}>
              API {apiHealthy === null ? 'Checking' : apiHealthy ? 'Online' : 'Offline'}
            </Tag>
            <Button onClick={() => void loadRuns()} loading={loading}>Refresh</Button>
            <Button onClick={() => void onRestartBackend()} loading={restartPending}>
              Restart Backend
            </Button>
            <span className="label-wrap">
              Include archived <Switch checked={includeArchived} onChange={setIncludeArchived} />
            </span>
            <Select<EvalKind>
              value={evalFilter}
              style={{ width: 140 }}
              onChange={setEvalFilter}
              options={[
                { value: 'All', label: 'All' },
                { value: 'Multi', label: 'Multi' },
                { value: 'ID', label: 'ID' },
                { value: 'OOD', label: 'OOD' },
                { value: 'Undisclosed', label: 'Undisclosed' },
                { value: 'CV', label: 'CV' },
              ]}
            />
            <Input.Search
              allowClear
              placeholder="Search run id/name/model/notes"
              style={{ width: 320 }}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Dropdown
              trigger={['click']}
              dropdownRender={() => (
                <div className="column-picker-menu">
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    {runColumnOptions.map((opt) => {
                      const checked = visibleRunColumns.includes(opt.key)
                      const disableUncheck = checked && visibleRunColumns.length === 1
                      return (
                        <Checkbox
                          key={opt.key}
                          checked={checked}
                          disabled={disableUncheck}
                          onChange={(e) => {
                            const nextChecked = e.target.checked
                            setVisibleRunColumns((prev) => {
                              if (nextChecked) return prev.includes(opt.key) ? prev : [...prev, opt.key]
                              return prev.filter((key) => key !== opt.key)
                            })
                          }}
                        >
                          {opt.label}
                        </Checkbox>
                      )
                    })}
                  </Space>
                </div>
              )}
            >
              <Button>Columns</Button>
            </Dropdown>
            <Button
              onClick={() => {
                setShowRepoDetails((prev) => {
                  const next = !prev
                  const repoKeys = repoColumns.map((repo) => `repo:${repo}`)
                  setVisibleRunColumns((current) => {
                    if (next) {
                      const seen = new Set(current)
                      const merged = [...current]
                      for (const key of repoKeys) {
                        if (seen.has(key)) continue
                        seen.add(key)
                        merged.push(key)
                      }
                      return merged
                    }
                    return current.filter((key) => !key.startsWith('repo:'))
                  })
                  return next
                })
              }}
            >
              {showRepoDetails ? 'Hide Repo Detail Columns' : 'Show Repo Detail Columns'}
            </Button>
            {anyRunHasLintData && (
              <Button
                onClick={() => {
                  setShowLintColumns((prev) => {
                    const next = !prev
                    setVisibleRunColumns((current) => {
                      if (next) {
                        const seen = new Set(current)
                        const merged = [...current]
                        for (const key of LINT_COLUMN_KEYS) {
                          if (seen.has(key)) continue
                          seen.add(key)
                          merged.push(key)
                        }
                        return merged
                      }
                      return current.filter((key) => !LINT_COLUMN_KEY_SET.has(key))
                    })
                    return next
                  })
                }}
              >
                {showLintColumns ? 'Hide Lint Columns' : 'Show Lint Columns'}
              </Button>
            )}
            <Button type="primary" disabled={selectedRunIds.length !== 2} onClick={() => void onCompareSelected()}>
              Compare ({selectedRunIds.length})
            </Button>
            <Button disabled={selectedRunIds.length === 0} onClick={onArchiveSelected}>
              Archive ({selectedRunIds.length})
            </Button>
            <Popconfirm
              title={`Delete ${selectedRunIds.length} run(s)?`}
              description="This removes selected runs from active and archived directories."
              okText="Delete"
              okButtonProps={{ danger: true, loading: bulkDeletePending }}
              onConfirm={() => void onDeleteSelected()}
              disabled={selectedRunIds.length === 0}
            >
              <Button danger disabled={selectedRunIds.length === 0} loading={bulkDeletePending}>
                Delete ({selectedRunIds.length})
              </Button>
            </Popconfirm>
          </Space>
        </Card>

        <Card>
          {loadError && (
            <Alert
              type="error"
              showIcon
              style={{ marginBottom: 12 }}
              message="Unable To Load FIM Runs"
              description={
                <div>
                  <div>{loadError}</div>
                  <div style={{ marginTop: 6 }}>
                    Check backend health: <code>curl {window.location.origin}/health</code>
                  </div>
                </div>
              }
            />
          )}
          <div ref={runTableContainerRef}>
            <Table<DisplayRunRow>
              rowKey={(row) => row.run_id}
              loading={loading}
              columns={visibleColumns}
              dataSource={runsForDisplay}
              size="small"
              showSorterTooltip={{ target: 'sorter-icon' }}
              onChange={(_pagination, _filters, sorter) => {
                const active = Array.isArray(sorter) ? sorter[0] : sorter
                const key = active?.columnKey != null ? String(active.columnKey) : ''
                if (active?.order && key) setRunSort({ key, order: active.order })
                else setRunSort(null)
              }}
              rowSelection={{
                columnWidth: RUN_TABLE_SELECTION_WIDTH,
                selectedRowKeys: selectedRunIds,
                onChange: (keys) => setSelectedRunIds(keys.map(String)),
              }}
              pagination={{ pageSize: 50, showSizeChanger: true, pageSizeOptions: [25, 50, 100] }}
              rowClassName={(row) => {
                const classes: string[] = []
                if (row._cv_is_child) classes.push('cv-child-row')
                if (lastPinnedRunId && row.run_id === lastPinnedRunId) classes.push('pinned-boundary-row')
                return classes.join(' ')
              }}
              scroll={{ x: runTableScrollX }}
            />
          </div>
        </Card>
      </Space>

      <Modal
        title={`Archive ${selectedRunIds.length} run(s)`}
        open={archiveModalOpen}
        onCancel={() => setArchiveModalOpen(false)}
        onOk={() => void submitArchive()}
        okButtonProps={{ loading: archiveSaving }}
        okText="Archive"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text type="secondary">Optional archive reason</Text>
          <Input.TextArea
            rows={4}
            value={archiveReason}
            onChange={(e) => setArchiveReason(e.target.value)}
            placeholder="Reason"
          />
        </Space>
      </Modal>

      <Modal
        title="Compare Runs"
        open={compareModalOpen}
        onCancel={() => setCompareModalOpen(false)}
        footer={<Button onClick={() => setCompareModalOpen(false)}>Close</Button>}
      >
        {compareLoading && <Text>Loading comparison...</Text>}
        {!compareLoading && compareData && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text><b>Left:</b> {compareData.left.run_id}</Text>
            <Text><b>Right:</b> {compareData.right.run_id}</Text>
            <Table<CompareMetricRow>
              size="small"
              pagination={false}
              dataSource={buildCompareRows(compareData)}
              rowClassName={(row) => (row.emphasized ? 'compare-emphasis-row' : '')}
              columns={[
                { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                { title: 'Left', dataIndex: 'left', key: 'left' },
                { title: 'Right', dataIndex: 'right', key: 'right' },
                { title: 'Right − Left', dataIndex: 'delta', key: 'delta' },
              ]}
            />
            <Text type="secondary">
              The highlighted row is each run’s comparison number (the red-outlined cell in the runs
              table): Unweighted OOD for CV runs whose folds cover each repo as OOD exactly once,
              Unweighted Total for non-CV runs. CV runs without that clean coverage have no single
              comparison number and show “—”.
            </Text>
          </Space>
        )}
      </Modal>

      <Modal
        title={`Run Details: ${detailRunId}`}
        open={detailOpen}
        onCancel={() => setDetailOpen(false)}
        footer={<Button onClick={() => setDetailOpen(false)}>Close</Button>}
        width="96vw"
        style={{ top: 24 }}
      >
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <Card size="small">
            <Space wrap size={10}>
              {detailHasSyntax && (
                <Select<DetailMode>
                  value={detailMode}
                  style={{ width: 130 }}
                  onChange={(v) => {
                    setDetailMode(v)
                    setDetailOutcomeFilter('all')
                    setDetailPage(1)
                  }}
                  options={detailModeOptions}
                />
              )}
              <Select
                value={detailOutcomeFilter}
                style={{ width: 170 }}
                onChange={(v) => {
                  setDetailOutcomeFilter(v)
                  setDetailPage(1)
                }}
                options={outcomeOptions}
              />
              <Select
                value={detailSampleType}
                style={{ width: 180 }}
                onChange={(v) => {
                  setDetailSampleType(v)
                  setDetailPage(1)
                }}
                options={detailSampleTypeOptions}
              />
              <Select
                value={detailSource}
                style={{ width: 150 }}
                onChange={(v) => {
                  setDetailSource(v)
                  setDetailPage(1)
                }}
                options={detailSourceOptions}
              />
              <Select<ResponseFilter>
                value={detailResponseFilter}
                style={{ width: 180 }}
                disabled={detailMode === 'syntax'}
                onChange={(v) => {
                  setDetailResponseFilter(v)
                  setDetailPage(1)
                }}
                options={RESPONSE_FILTER_OPTIONS}
              />
              <Input.Search
                allowClear
                placeholder="Search id/file/repo/response"
                style={{ width: 300 }}
                value={detailQuery}
                onChange={(e) => {
                  setDetailQuery(e.target.value)
                  setDetailPage(1)
                }}
              />
              <Button onClick={() => void loadDetailResults()} loading={detailLoading}>Refresh Results</Button>
            </Space>
          </Card>

          {detailError && (
            <Alert type="error" showIcon message="Failed to load results" description={detailError} />
          )}

          <Table<ResultListItem>
            rowKey={(row) => row.id}
            loading={detailLoading}
            columns={detailColumns}
            dataSource={detailData?.items || []}
            size="small"
            scroll={{ x: 1500, y: 340 }}
            pagination={{
              current: detailPage,
              pageSize: detailPageSize,
              total: detailData?.total || 0,
              showSizeChanger: true,
              pageSizeOptions: [10, 25, 50, 100],
              onChange: (page, pageSize) => {
                setDetailPage(page)
                setDetailPageSize(pageSize)
              },
            }}
            onRow={(row) => ({
              tabIndex: 0,
              onClick: () => {
                void loadSelectedResultDetail(row.id)
              },
              onKeyDown: (e: KeyboardEvent) => {
                if (e.key === 'Enter') {
                  void loadSelectedResultDetail(row.id)
                }
              },
            })}
            rowClassName={(row) => (row.id === selectedResultId ? 'result-row-selected' : '')}
          />

          <Card size="small" title={selectedResultId ? `Selected Result: ${selectedResultId}` : 'Result Detail'}>
            {selectedResultLoading && <Text>Loading selected result detail...</Text>}
            {selectedResultError && <Alert type="error" showIcon message={selectedResultError} />}
            {!selectedResultLoading && !selectedResultError && !selectedResultDetail && (
              <Text type="secondary">Click a result row above to inspect full details.</Text>
            )}
            {!selectedResultLoading && !selectedResultError && selectedResultDetail && (
              <Space direction="vertical" style={{ width: '100%' }} size={10}>
                <Space wrap>
                  <Tag>Match: {selectedResultDetail.outcome_match}</Tag>
                  <Tag>Syntax: {selectedResultDetail.outcome_syntax}</Tag>
                </Space>
                <div>
                  <Text strong>File</Text>
                  <div><code>{asText(selectedResultDetail.row.file_path)}</code></div>
                </div>
                <div>
                  <Text strong>Prefix</Text>
                  {renderHighlightedCode(selectedResultDetail.row.prefix, 540)}
                </div>
                <div className="detail-code-pair">
                  <div className="detail-code-col">
                    <Text strong>Target</Text>
                    {renderHighlightedCode(selectedResultDetail.row.target, 260)}
                  </div>
                  <div className="detail-code-col">
                    <Text strong>Predicted</Text>
                    {renderHighlightedCode(selectedResultDetail.row.predicted, 260)}
                  </div>
                </div>
                <div>
                  <Text strong>Suffix</Text>
                  {renderHighlightedCode(selectedResultDetail.row.suffix, 540)}
                </div>
              </Space>
            )}
          </Card>
        </Space>
      </Modal>
    </div>
  )
}
