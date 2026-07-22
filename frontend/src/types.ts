/**
 * TypeScript shapes of the backend API's JSON responses (and the few request
 * params/enums that go with them). These mirror what the FastAPI server returns;
 * api.ts annotates each call with the matching type here, so the editor flags any
 * field the UI reads that the server doesn't actually send.
 */
export type EvalKind = 'All' | 'Multi' | 'ID' | 'OOD' | 'Undisclosed' | 'CV'

export type RunListItem = {
  run_id: string
  run_name: string | null
  archived: boolean
  in_progress: boolean
  eval_kind: Exclude<EvalKind, 'All'>
  provider: string | null
  model: string | null
  base_model: string | null
  dataset: string | null
  dataset_version: string | null
  dataset_version_bg: string | null
  dataset_version_fg: string | null
  dataset_version_border: string | null
  total_examples: number | null
  exact_match_rate: number | null
  prefix_match_rate: number | null
  syntax_pass_rate: number | null
  lint_pass_rate: number | null
  total_repo_average: number | null
  in_domain_repo_average: number | null
  out_of_domain_repo_average: number | null
  total_repo_prefix_average: number | null
  in_domain_repo_prefix_average: number | null
  out_of_domain_repo_prefix_average: number | null
  lint_in_domain_pass_rate: number | null
  lint_out_of_domain_pass_rate: number | null
  repo_scores: Array<{
    repo: string
    exact_match_rate: number | null
    prefix_match_rate: number | null
    total_examples: number | null
    domain: string | null
    in_domain: boolean | null
  }>
  repo_rate_map: Record<string, number>
  repo_prefix_rate_map: Record<string, number>
  repo_domain_map: Record<string, 'id' | 'ood'>
  repo_example_count_map: Record<string, number>
  train_repos: string[]
  duration_seconds: number | null
  train_duration_seconds: number | null
  timestamp: string | null
  notes: string
  cv_parent_run_id: string | null
  cv_subrun_index: number | null
  cv_subrun_count: number | null
  cv_subrun_run_ids: string[]
  cv_holdout_repos: string[]
  is_cv_parent: boolean
  is_cv_subrun: boolean
}

export type RunsResponse = {
  items: RunListItem[]
  total: number
}

export type CompareResponse = {
  left: RunListItem
  right: RunListItem
  deltas: Record<string, number | null>
}

export type DetailMode = 'match' | 'syntax'

export type OutcomeFilterMatch = 'all' | 'match' | 'prefix' | 'mismatch'
export type OutcomeFilterSyntax = 'all' | 'pass' | 'fail' | 'inconclusive'
export type ResponseFilter = 'all' | 'non_empty' | 'empty'

export type ResultListItem = {
  id: string
  file_path: string | null
  repo: string | null
  category: string | null
  dataset_source: string | null
  exact_match: boolean
  prefix_match: boolean
  predicted_syntax_passed: boolean | null
  original_syntax_passed: boolean | null
  input_tokens: number | null
  output_tokens: number | null
  response_preview: string
  target_preview: string
  outcome: string
  error_type: string | null
  error_message: string | null
}

export type ResultListResponse = {
  items: ResultListItem[]
  total: number
  page: number
  page_size: number
  sample_types: string[]
  sources: string[]
}

export type ResultDetailResponse = {
  run_id: string
  row: Record<string, unknown>
  outcome_match: string
  outcome_syntax: string
}

