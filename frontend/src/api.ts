/**
 * Every "ask the server" call for the web UI lives here, in one file.
 *
 * A component just calls a function like `fetchRuns()` or `deleteRuns()` and gets data
 * back — it never needs to know the server URL, how to phrase the request, or the shape
 * of the reply; all of that is kept in here. Why it's worth its own file:
 *   1. One place to change things — if an endpoint's URL or params change, you fix it
 *      here once instead of in every page that calls it.
 *   2. The pages stay simple — they ask for data by function name, not by wrangling HTTP.
 *   3. Mistakes get caught — each function declares what it sends and what comes back
 *      (typed against `types.ts`), so the editor flags incorrect use.
 *
 * Requests use a relative URL (`/`), so the Vite dev/preview proxy (vite.config.ts,
 * `VITE_API_TARGET`) — or whatever serves the built UI — forwards them to the backend.
 */
import axios from 'axios'
import type {
  CompareResponse,
  DetailMode,
  EvalKind,
  ResponseFilter,
  ResultDetailResponse,
  ResultListResponse,
  RunListItem,
  RunsResponse,
} from './types'

const client = axios.create({
  baseURL: '/',
  headers: { 'Content-Type': 'application/json' },
})

export type HealthResponse = {
  status: string
  pid?: number
  started_at_utc?: string
}

/** List evaluation runs, optionally including archived ones, filtered by eval kind and a free-text query. */
export async function fetchRuns(params: {
  includeArchived: boolean
  evalFilter: EvalKind
  query: string
}): Promise<RunsResponse> {
  const res = await client.get<RunsResponse>('/api/fim/runs', {
    params: {
      include_archived: params.includeArchived,
      eval_filter: params.evalFilter,
      q: params.query,
    },
  })
  return res.data
}

/** Backend liveness probe (status + pid + start time). */
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await client.get<HealthResponse>('/health')
  return res.data
}

/** Ask the server to restart itself (dev convenience, e.g. after backend code changes). */
export async function restartBackend(): Promise<{ status: string }> {
  const res = await client.post<{ status: string }>('/api/system/backend/restart')
  return res.data
}

/** Update the free-text notes on a run; returns the updated run row. */
export async function saveRunNotes(runId: string, notes: string): Promise<RunListItem> {
  const res = await client.put<RunListItem>(`/api/fim/runs/${encodeURIComponent(runId)}/notes`, { notes })
  return res.data
}

/** Archive the given runs (soft-delete, with a reason), moving them out of the active list. */
export async function archiveRuns(runIds: string[], reason: string): Promise<void> {
  await client.post('/api/fim/runs/archive', { run_ids: runIds, reason })
}

/** Permanently delete the given runs. */
export async function deleteRuns(runIds: string[]): Promise<void> {
  await client.post('/api/fim/runs/delete', { run_ids: runIds })
}

/** Compare exactly two runs (side-by-side metrics/results). */
export async function compareRuns(runIds: [string, string]): Promise<CompareResponse> {
  const res = await client.post<CompareResponse>('/api/fim/runs/compare', { run_ids: runIds })
  return res.data
}

/** Fetch a paginated, filtered page of a run's per-sample results. */
export async function fetchRunResults(params: {
  runId: string
  mode: DetailMode
  outcomeFilter: string
  sampleType: string
  source: string
  responseFilter: ResponseFilter
  query: string
  page: number
  pageSize: number
}): Promise<ResultListResponse> {
  const res = await client.get<ResultListResponse>(`/api/fim/runs/${encodeURIComponent(params.runId)}/results`, {
    params: {
      mode: params.mode,
      outcome_filter: params.outcomeFilter,
      sample_type: params.sampleType,
      source: params.source,
      response_filter: params.responseFilter,
      q: params.query,
      page: params.page,
      page_size: params.pageSize,
    },
  })
  return res.data
}

/** Fetch the full detail (prompt, prediction, target, …) for a single result/sample. */
export async function fetchRunResultDetail(runId: string, resultId: string): Promise<ResultDetailResponse> {
  const res = await client.get<ResultDetailResponse>(
    `/api/fim/runs/${encodeURIComponent(runId)}/results/${encodeURIComponent(resultId)}`
  )
  return res.data
}
