# Uber Benchmark

WebArena-style benchmark for evaluating browser agents on a ride-sharing web application. 150 tasks across three categories, evaluated via SQL postcondition checks and fuzzy string matching.

## Task Breakdown

| Category | Count | Description |
|---|---|---|
| information_retrieval | 47 | Query travel times, compare prices, look up ride history |
| state_mutation | 45 | Book rides, cancel rides, update account settings |
| multi_step_reasoning | 58 | Multi-page workflows combining navigation, data extraction, and actions |

| Difficulty | Count |
|---|---|
| easy | 45 |
| medium | 61 |
| hard | 44 |

## Infrastructure Requirements

Three services must be running:

### 1. Uber Web App (frontend)

A ride-sharing web application serving these routes:

- `/` — home / ride booking
- `/account` — user account settings
- `/history` — ride history

Must have a seeded test user (default: `testuser1` / `password123`).

### 2. Benchmark API (backend)

An API server exposing two endpoints for evaluation:

**`POST /api/_benchmark/reset`**
- Request: `{"confirm": true}`
- Response: `{"users_count": N, "drivers_count": N, "rides_count": N}`
- Resets the database to its seed state between tasks.

**`GET /api/_benchmark/verify?query=<SQL>`**
- Response: `{"rows": [...], "count": N}`
- Executes a read-only SQL query for evaluation. The database must contain these tables:

| Table | Purpose |
|---|---|
| `users` | User accounts |
| `rides` | Ride records (bookings, completions, cancellations) |
| `_seed_rides` | Seed data for reset |
| `zones` | Geographic zones (pickup/dropoff areas) |
| `zone_travel_times` | Estimated travel times between zones |
| `surge_factors` | Dynamic pricing multipliers |

### 3. LLM Server (OpenAI-compatible)

Any OpenAI-compatible API server — **vLLM**, **SGLang**, or others. Must expose a `/models` endpoint.

Both vLLM and SGLang expose the same OpenAI-compatible chat completions API (`/v1/chat/completions`) and model listing (`/v1/models`). The benchmark runner uses `ChatOpenAI` which sends only standard OpenAI parameters (`temperature`, `frequency_penalty`, `max_completion_tokens`), all of which are supported by both backends.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_BASE_URL` | Yes | — | LLM server base URL (e.g. `http://localhost:8000/v1`). Falls back to `VLLM_URL` for backward compatibility. |
| `MODEL_NAME` | Yes | — | Model identifier (e.g. `meta-llama/Llama-3-70b`) |
| `LLM_API_KEY` | No | `EMPTY` | API key for the LLM server (most local servers don't require one) |
| `UBER_BASE_URL` | No | `http://158.130.4.153:3002` | Uber web app URL |
| `UBER_API_URL` | No | `http://158.130.4.153:8000` | Benchmark API URL |
| `TEST_USERNAME` | No | `testuser1` | Login username |
| `TEST_PASSWORD` | No | `password123` | Login password |

## Usage

### Setup environment

```bash
# With vLLM
export LLM_BASE_URL=http://localhost:8000/v1
export MODEL_NAME=meta-llama/Llama-3-70b

# With SGLang
export LLM_BASE_URL=http://localhost:30000/v1
export MODEL_NAME=meta-llama/Llama-3-70b

# Or use legacy env var (backward compatible)
export VLLM_URL=http://localhost:8000/v1
```

### Run the benchmark

```bash
# Run all 150 tasks
python benchmark/uber/run_uber.py --yes

# Run specific tasks by ID
python benchmark/uber/run_uber.py --task-ids 0,1,2 --yes

# Resume from task index N
python benchmark/uber/run_uber.py --start-from 50 --yes

# Skip DB reset between tasks (faster, less isolated)
python benchmark/uber/run_uber.py --no-reset --yes

# Use a custom tasks file
python benchmark/uber/run_uber.py --tasks-file path/to/tasks.json --yes
```

### Post-hoc validation

Re-evaluate a completed run against the database (requires the API to still be running):

```bash
python benchmark/uber/validate_uber.py benchmark_results/run_<timestamp>/
```

The validator uses a three-tier evaluation strategy:
1. **DB evaluation** — re-runs SQL postcondition checks from task specs
2. **Inline evaluation** — falls back to the eval results saved during the run
3. **String similarity** — last resort, compares agent output to manually provided expected answers (0.8 similarity threshold)

## Output

Results are saved incrementally to `benchmark_results/run_<timestamp>/`:

| File | Contents |
|---|---|
| `results_summary.json` | Full per-task results (agent output, eval, timing, errors) |
| `eval_report.json` | Condensed eval pass/fail per task |
| `expected_answers.json` | Template for manual review — fill in `expected_answer` for tasks that need it |

After a validation run, an additional `validation_report.json` is written with accuracy metrics.

## Evaluation Methods

Defined per-task in `tasks_uber.json` under the `eval` key:

### `db_state_check`
Runs a SQL `postcondition_query` and checks the result:
- `exists` — passes if >= 1 row returned
- `count` — passes if row count equals `expected_result`

### `string_match`
Compares agent text output against reference values:
- `must_include` — all listed strings must appear (case-insensitive)
- `reference_query` — runs a SQL query, then fuzzy-matches the result against agent output within a configurable `tolerance` (default 0.3 = 30% relative error)

Tasks with both eval types must pass all checks.

## Files

```
benchmark/uber/
  run_uber.py          # Benchmark runner
  validate_uber.py     # Post-hoc validation script
  tasks_uber.json      # 150 WebArena-format task definitions
  README.md            # This file
```
