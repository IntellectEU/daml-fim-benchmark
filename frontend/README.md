# FIM Benchmark Frontend (React + Vite)

Web UI for browsing FIM evaluation runs produced by the benchmark.

## Scope
- List FIM runs from `fim/logs` (+ `fim/logs/z_archive` when "Include archived" is on)
- Search and eval-kind filter (`All`, `Multi`, `ID`, `OOD`, `Undisclosed`, `CV`)
- Pin rows (persisted in browser localStorage)
- Edit notes (`notes.txt`)
- Bulk archive and delete
- Compare two selected runs (summary metrics)
- Overlay details modal for per-run results (mode/outcome/category/source filters + result inspection)

## Eval kinds
Each run is tagged with one `eval_kind`, shown as a colored tag and filterable via the dropdown:
- **ID** — every evaluated repo is in-domain (part of the model's training data).
- **OOD** — every evaluated repo is out-of-domain.
- **Multi** — a mix of declared in-domain and out-of-domain repos.
- **Undisclosed** — no declared training-data relationship at all, e.g. closed models like GitHub Copilot whose training corpus isn't disclosed.
- **CV** — cross-validated: fold subruns are aggregated so each repo is scored by the fold that held it out.

## Run table columns
Score columns come in weighted/unweighted pairs, and each has a hover tooltip explaining it:
- **Weighted Total** — exact-match rate pooled over every example in the run, so repos with more examples contribute proportionally more.
- **Unweighted Total** — average of per-repo exact-match rates; every repo counts equally regardless of example count.
- **Unweighted ID** — unweighted average over in-domain repos only.
- **Unweighted OOD** — unweighted average over out-of-domain repos only.
- **Unweighted Undisclosed** — unweighted average over repos with no declared training-data relationship for that run.

Each score cell also shows a `+x%` subtext under the percentage — over-generation: predictions that start with the correct target but add trailing text after it (exact matches are excluded from this count). Hovering the subtext, or the "+over-generation" label under a column header, shows the same explanation.

One column per run gets a red outline marking the number to use when comparing runs:
- **Unweighted Total**, for any non-CV run.
- **Unweighted OOD**, for a CV parent whose folds cover each repo as out-of-domain exactly once (so the number reflects only held-out scoring, never in-domain). CV parents without that clean coverage get no outline.

Hovering an outlined cell shows which of the two rules applied.

Toggle "Show Repo Detail Columns" for a per-repo score breakdown, and "Show Lint Columns" for the optional Daml syntax/lint pass-rate columns (Lint Total / Lint ID / Lint OOD) — that toggle only appears when at least one listed run has lint data. Use the "Columns" dropdown to show/hide any column.

## Cross-validation (CV) runs
A CV run is a parent row plus one fold subrun per model/holdout combination. If a parent has folds, its row gets an expand arrow (▸/▾) that reveals the fold subruns nested underneath, each tagged `Fold N`.

Opening **Details** on a CV parent (rather than a fold subrun) shows the held-out merge of its folds: for each repo, the rows come from whichever fold scored that repo out-of-domain, so nothing in the parent's Details view is ever shown as in-domain.

## Details modal
Clicking **Details** on a run opens a modal with:
- **Mode**: `Match` (exact/over-gen/mismatch) or `Syntax` (pass/fail/inconclusive) — changes which outcome filter options are shown.
- **Outcome** filter, options depend on mode.
- **Category** filter — the kind of masked span (e.g. `ensure`).
- **Source** filter — which test split the example came from (`syntax` or `random`).
- **Response** filter — all / non-empty / empty.
- Free-text search across id/file/repo/response.
- A paginated result table; clicking a row loads its full prefix/target/predicted/suffix into the panel below, with copy-to-clipboard on code blocks.

## Run
From the `frontend/` directory:

Recommended (single command, with log capture):
```bash
./run.sh
```

This starts the backend and the Vite dev server, writing logs to:
- `frontend/.logs/backend.log`
- `frontend/.logs/vite.log`

The backend resolves run artifacts against `DAML_FIM_BENCHMARK_ROOT` (defaults to
the benchmark repo root when launched via `run.sh`).

To use a non-default backend port, set both `API_PORT` (which port `run.sh` starts
uvicorn on) and `VITE_API_TARGET` (where Vite's dev-server proxy forwards `/api`
and `/health`) — they don't default off each other:
```bash
API_PORT=8020 VITE_API_TARGET=http://127.0.0.1:8020 ./run.sh
```

Manual mode:

1. Start the API backend from the benchmark repo root:
```bash
python -m uvicorn daml_fim_benchmark.server.app:app --host 127.0.0.1 --port 8010
```

2. In another terminal, start the frontend dev server:
```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`).

To restart just the backend (e.g. after editing server code) without restarting Vite:
```bash
./scripts/restart_backend.sh
```
It kills whatever is listening on `API_PORT` (default 8010) and relaunches uvicorn, logging to the same `frontend/.logs/backend.log`. The UI also has a "Restart Backend" button that calls this same script via `POST /api/system/backend/restart`.

## Quick troubleshooting
```bash
curl http://127.0.0.1:8010/health
tail -n 120 frontend/.logs/backend.log
tail -n 120 frontend/.logs/vite.log
```

## API endpoints
- `GET /health`
- `POST /api/system/backend/restart` — localhost-only; restarts the backend via `scripts/restart_backend.sh`.
- `GET /api/fim/runs` — list runs. Query params: `include_archived`, `eval_filter` (`All`/`Multi`/`ID`/`OOD`/`Undisclosed`/`CV`), `q` (free-text search).
- `GET /api/fim/runs/{run_id}/results` — paginated per-example results for one run. Query params: `mode` (`match`/`syntax`), `outcome_filter`, `sample_type`, `source`, `response_filter` (`all`/`non_empty`/`empty`), `q`, `page`, `page_size`.
- `GET /api/fim/runs/{run_id}/results/{result_id}` — full row detail for one example.
- `PUT /api/fim/runs/{run_id}/notes` — save a run's free-text notes.
- `POST /api/fim/runs/archive` — move runs to `fim/logs/z_archive`.
- `POST /api/fim/runs/delete` — delete runs (active and/or archived).
- `POST /api/fim/runs/compare` — diff summary metrics between exactly two runs.

## Note on CV OOD comparison numbers
For most CV parents, each fold's holdout repo(s) cover the OOD side exactly once, so the Unweighted OOD column is a clean comparison number (see above). In rare cases a repo in a fold's training split ends up with no examples actually sampled into training (small datasets), so that fold evaluates it as out-of-domain and the repo shows up as OOD in more than one fold; the UI then shows a fallback `Obs`/`Adj` pair in that cell instead of the outline, with a tooltip explaining which repos were affected. This is a research-workflow edge case, not something present in the shipped baseline data.
