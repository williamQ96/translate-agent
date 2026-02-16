# Development Log

This log summarizes project evolution from reconstructed `v0.1`, current `v0.2`, and planned `v0.3`.
Reconstruction is based on repository artifacts, README status, pipeline logs, and prior implementation history.

## v0.1 (Reconstructed Baseline)

### Scope
Initial end-to-end OCR translation pipeline, focused on producing a translated book output.

### Typical flow
1. OCR outputs aggregated into one combined markdown book.
2. Combined book split into chunks for translation.
3. Chunk translation + basic polish.
4. Assembly into one translated markdown.
5. Audit/rewrite existed but was not robustly chunk-anchored and not consistently loop-driven.

### Known limitations
1. Audit quality inconsistencies (false hallucination judgments on OCR-noisy chunks).
2. Audit/reference mismatch risk when chunk files were regenerated.
3. Rewrite loop instability (timeouts on late chunks, weak convergence).
4. Duplicate/redundant rewrite outputs in some runs.
5. Stage continuity gaps (some runs ended after translation without full audit+rewrite continuity).

## v0.2 (Current)

### Version goal
Stabilize pipeline execution and make quality loop chunk-anchored, resumable, and auditable.

### Current pipeline shape
1. Input ingestion (PDF/MD/TXT/OCR directory).
2. Source chunk organization first (`data/output/source_chunks`).
3. Glossary extraction/reuse (`data/glossary.json`).
4. Dense RAG indexing (ChromaDB).
5. Chunk translation (style-neutral first pass + optional style polish).
6. Assembly of first translated markdown.
7. Chunk-level audit against original source chunks.
8. Iterative rewrite-audit loop with score locking and final rewritten assembly.

### Major upgrades landed in v0.2
1. Chunk-to-chunk audit reference model (source chunk vs translated chunk).
2. Rewrite loop with lock mechanism (high-score chunks stop being rewritten).
3. Seed-audit freshness checks and auto-seed-audit fallback when report is stale/missing.
4. Resume-oriented run directory behavior (`--run-dir`) and loop state tracking.
5. Rewrite evidence retention and per-loop artifacts for traceability.
6. Audit guardrails to reduce unsupported hallucination penalties on noisy fragments.
7. RAG + glossary context injection used in translation/audit/rewrite paths.
8. Style injection control for polish stage (first-pass translation remains style-light).

### Observed status in current artifacts
1. Loop can progress across many rounds with chunk locking.
2. Hard-tail chunks still show timeout/latency spikes and slower convergence.
3. Remaining bottlenecks: retrieval quality and model-routing efficiency.

## v0.3 (Planned)

Planned from `dev_plan.md`:

### 1) Internet-search augmentation (controlled)
1. Triggered only on unresolved entities, repeated low-score chunks, or disputed audit claims.
2. Domain allowlist + context length caps + citation logging.
3. Search-result caching for determinism and speed.

### 2) Hybrid retrieval (dense + lexical)
1. Keep dense Chroma retrieval.
2. Add lexical retrieval (BM25).
3. Fuse with RRF and reuse across translation/audit/rewrite.

### 3) Model router (dual-model)
1. Default fast path: `qwen3:8b`.
2. Escalation path for hard chunks: `qwen3:30b`.
3. Strict timeout, capped escalation attempts, and automatic fallback to `8b`.

### 4) Additional planned quality controls
1. Optional reranking and self-correction retrieval.
2. Stronger audit evidence checks before harsh hallucination penalties.
3. Better convergence telemetry (retrieval confidence, escalation outcomes, timeout counters).

## v0.3 (In Progress - Implementation Started)

### Landed in current patch set
1. `src/rag/store.py` upgraded to hybrid retrieval foundation:
   - dense + lexical retrieval
   - RRF fusion
   - optional rerank (heuristic / local cross-encoder fallback)
   - retrieval self-correction (single-step query rewrite retry)
   - guarded web fallback providers (wikipedia/tavily/bing with cache)
2. `src/agents/writer.py` upgraded with dual-model routing:
   - default fast model path
   - escalation model path for difficult chunks
   - timeout fallback to default model
3. `src/audit.py` hardened for weak-evidence hallucination claims:
   - unsupported hallucination penalties are downgraded
   - weak-evidence cases are routed to human attention
4. `src/rewrite_audit_loop.py` observability upgrades:
   - duplicate loop-progress log suppression
   - per-loop start timestamp + elapsed output
   - terminal output tee to master/loop logs
5. New metric utility:
   - `scripts/compute_success_metrics.py` for success-metric snapshots.

## Version summary
1. `v0.1`: functional baseline, but fragile in audit/rewrite consistency.
2. `v0.2`: robust chunk-level quality loop with resumability and evidence tracking.
3. `v0.3` (planned): retrieval and routing optimization for better speed, quality, and convergence.

## 2026-02-15 Maintenance + Stabilization Checkpoint

### Scope completed
1. Permission validation flow verified with executable checks.
2. Generated artifact hygiene tightened in `.gitignore`.
3. Runtime output path normalization:
   - permission reports moved to `data/output/logs/permission_reports/`
4. Pipeline/runtime bug sweep with smoke tests and targeted fixes.

### Bugs fixed
1. `rewrite_audit_loop` crash: missing `re` import (`NameError` during loop phase print).
2. `rewrite_audit_loop` crash: `failed` item type mismatch (`dict` vs `int`) when building `failed_set`.
3. `WriterAgent.rewrite` compatibility issues:
   - added backward-compatible aliases for `current_translation`
   - added backward-compatible aliases for `issues`/`score`
   - added backward-compatible alias for `issue_tags`
4. `smoke_test_4chunks.ps1` robustness:
   - auto-resolve python interpreter (`VIRTUAL_ENV` > local venv > PATH)
   - fixed single-chunk `Count` handling in chunk-id parser
5. Windows console encoding issues:
   - added console UTF-8 setup in `src/pipeline.py`
   - added console UTF-8 setup in `scripts/audit_json_to_cn_report.py`
   - removed problematic non-ASCII help example in pipeline CLI help text
6. `check_permissions.ps1` stability:
   - fixed null/array recommendation handling
   - improved Defender fallback behavior
   - updated default report output location under `data/output/logs`

### Smoke-test evidence
1. Single-chunk smoke (`chunk_001`) completed end-to-end.
2. Single-chunk rewrite-loop run (`target=10`) executed rewrite path without parameter-signature crashes.
3. Remaining performance issue is model latency on hard chunks, not process crashes.

### Context checkpoint (for compacting)
1. Primary risk now: long rewrite latency on hard chunks.
2. Primary stability status: core pipeline is runnable and no longer failing on prior runtime type/signature errors.
3. Next tuning focus: model routing thresholds + timeout budgets + rewrite acceptance strategy.

## 2026-02-15 Beads Research + 3-Chunk Loop Test

### Beads research decision
1. `beads` is feasible as an optional memory/trace layer.
2. Best use in this project:
   - store chunk rewrite decisions and audit outcomes
   - store human-attention resolutions and glossary choices
3. Not recommended as a replacement for current retrieval stack (hybrid RAG remains primary).
4. Integration should be fail-open (pipeline continues when beads is unavailable).

### 3-chunk stress test (`chunk 1,2,4`, `max_loops=5`)
Run:
`data/output/_smoke_tests/smoke_20260215_170931/rewrites/rewrite_loop_run_20260215_171005`

Loop averages:
1. Loop 0: `7.3`
2. Loop 1: `8.3`
3. Loop 2: `8.0`
4. Loop 3: `9.0` (all chunks locked, loop stops)

Chunk trajectories:
1. Chunk 1: `5 -> 5 -> 8 -> 9`
2. Chunk 2: `9 -> 10`
3. Chunk 4: `8 -> 10`

Conclusion:
1. Trend is improving overall, but not strictly monotonic (Loop 1 -> 2 dipped).
2. Convergence achieved by Loop 3 for this 3-chunk case.
