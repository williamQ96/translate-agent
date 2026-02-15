# Dev Plan: Internet-Search-Augmented Translation + Advanced RAG Evaluation

## 1) Goal

Improve chunk-level translation quality and convergence speed in `translate -> audit -> rewrite` by:

1. Adding controlled internet search when local evidence is insufficient.
2. Upgrading retrieval from dense-only to stronger RAG options.
3. Keeping latency, cost, and hallucination risk bounded.

This document is evaluation-first. No implementation changes are applied in this step.

## 2) Current Baseline (from repo)

1. Dense RAG exists via Chroma (`src/rag/store.py`) and is used in translation/audit/rewrite flows.
2. Glossary extraction exists (`src/knowledge/extractor.py`, `src/knowledge/glossary.py`).
3. Pipeline already runs chunk-level quality loop by default (`src/pipeline.py` stage 7/8).
4. Rewrite loop artifacts show non-converging chunks and frequent human-attention flags.

Implication: retrieval quality and audit signal quality are current bottlenecks, not missing pipeline stages.

## 3) Feasibility + Worthwhile Assessment

### A. Internet search calls (on-demand)

- Feasibility: High
- Expected value: High for proper nouns, historical terms, OCR-noisy ambiguous phrases.
- Risk: Medium (external noise, source quality variance, latency/cost).
- Decision: **Worth doing, but with strict trigger/guardrails**.

Recommended usage points:

1. Glossary build/refresh: unresolved terms only.
2. Translation time: only when chunk has low-confidence entities/terms.
3. Rewrite time: only for chunks repeatedly failing due to factual terminology mismatch.
4. Audit: optional evidence check for hallucination claims in disputed chunks.

Trigger policy (must meet at least one):

1. Unknown named entity not in glossary + low lexical overlap with source.
2. Repeated low score for same chunk across loops (`<= target-3` after 2 loops).
3. Auditor marks hallucination/omission but evidence weak or contradictory.

Guardrails:

1. Domain allowlist by default (Wikipedia, Britannica, Grove/IMSLP-like music domains, official sources).
2. Max external snippets per chunk (e.g., 2-4), hard char cap in prompt.
3. Cache search results by normalized query (`data/cache/web_context.json`) to prevent repeated calls.
4. Save citations in chunk metadata for audit traceability.

### B. Hybrid retrieval (dense + lexical)

- Feasibility: High
- Expected value: High (handles exact terms + semantic context together).
- Risk: Low-Medium (extra index path, parameter tuning).
- Decision: **Top priority upgrade**.

Preferred implementation for this repo:

1. Keep current Chroma dense retrieval.
2. Add lexical retriever (BM25) over source chunks/paragraphs.
3. Fuse results with Reciprocal Rank Fusion (RRF).
4. Reuse same fused retrieval in translation, audit, rewrite for consistency.

Notes:

1. Chroma now has sparse/hybrid capabilities in Cloud docs; local stack in this repo should not depend on Cloud-only Search API.
2. Implement repository-local hybrid logic to avoid vendor lock and keep offline operability.

### C. Reranking

- Feasibility: Medium-High
- Expected value: Medium-High (better top-k precision before prompting).
- Risk: Medium (latency).
- Decision: **Worth doing after hybrid retrieval**.

Options:

1. Local reranker (`sentence-transformers` cross-encoder): best for privacy/offline; slower on CPU.
2. API reranker (e.g., Cohere): faster to integrate, external dependency/cost.

Recommendation:

1. Start local optional reranker on top-N candidates (`N=10 -> k=3`), enable per-stage toggle.
2. Fall back to no-rerank when timeout budget is exceeded.

### D. Graph RAG

- Feasibility: Medium-Low for near-term.
- Expected value: Low-Medium for this task (book translation chunk alignment), higher for corpus-level analytical QA.
- Risk: High (indexing complexity, cost/time, ops burden).
- Decision: **Not first-line for current translation pipeline**.

Use only if later needed for:

1. Cross-book entity disambiguation at scale.
2. Thematic/global question answering beyond chunk translation.

### E. Self-correction retrieval (query correction + retrieval quality checks)

- Feasibility: High
- Expected value: High for currently failing tail chunks.
- Risk: Medium (loop complexity).
- Decision: **Worth doing (phase 2)**.

Practical variant for this repo:

1. Retrieval critic scores evidence sufficiency before LLM generation.
2. If low confidence: rewrite retrieval query + retry retrieval once.
3. If still low: optionally call internet search fallback, then continue.
4. Record retrieval confidence in audit metadata.

### F. Dual-model strategy (qwen3:8b + qwen3:30b)

- Feasibility: High
- Expected value: High for stability + quality balance.
- Risk: Low-Medium (routing complexity).
- Decision: **Adopt as default operating mode**.

Rationale from current runs:

1. Large-model-only runs show long-tail latency and timeout spikes on late chunks.
2. Pipeline quality depends on loop completion/convergence, not only single-pass model strength.
3. Smaller model can keep throughput stable; larger model should focus on hard cases.

Recommended routing policy:

1. Default model: `qwen3:8b` for translation, first-pass rewrite, and most audits.
2. Escalation model: `qwen3:30b` only when chunk is persistently difficult.
3. De-escalate back to `8b` after one escalation attempt to avoid runaway slow loops.

Escalation triggers (any one):

1. Chunk score remains `< target_score` after 2 loops.
2. Chunk marked `needs_human_attention=true`.
3. Hallucination/omission flags persist for 2 consecutive loops.
4. Retrieval confidence remains low after one self-correction attempt.

Latency guardrails:

1. Per-call timeout budget for `30b` (strict, shorter than current failure tail).
2. Max escalation attempts per chunk per run (e.g., 1-2).
3. Automatic fallback to `8b` on timeout/error.

## 4) Recommended Rollout (Phased)

### Phase 0: Instrumentation (quick win)

1. Add retrieval diagnostics per chunk: source ids, scores, confidence, latency.
2. Add `evidence_used` and citation fields to audit/rewrite artifacts.
3. Add counters in `loop_state.json`: retrieval_failures, web_fallback_calls, rerank_timeouts.
4. Time every loop and print to terminal:
   - loop start timestamp
   - per-loop elapsed seconds
   - cumulative runtime snapshot
5. Persist all terminal-equivalent output into logs with layered scope:
   - master-level pipeline log (entire run)
   - loop-level logs (one file per rewrite loop)
   - optional chunk-level append sections for slow/error chunks

### Phase 1: Hybrid retrieval + internet search guardrails

1. Implement `HybridRetriever` (dense + BM25 + RRF).
2. Wire it into translation/audit/rewrite uniformly.
3. Add `WebContextProvider` with:
   - provider abstraction (`tavily`/`bing`/`off`)
   - allowlist/excludelist domains
   - cache + timeout + retry budget.
4. Keep default `web_search.enabled=false`; enable by config/CLI.

### Phase 2: Reranker + retrieval self-correction

1. Add optional reranker interface (`local_cross_encoder` first).
2. Add single-step retrieval self-correction loop:
   - initial retrieval
   - evidence quality check
   - one reformulated retrieval
   - optional web fallback.
3. Add per-stage latency budgets to avoid freeze on late chunks.

### Phase 2.5: Dual-model routing

1. Add model router (`8b` default, `30b` escalated).
2. Track escalation reason and outcomes in loop artifacts.
3. Add chunk-level model usage stats for post-run analysis.

### Phase 3: Audit robustness hardening

1. Add stricter audit evidence requirement before hallucination penalties.
2. If auditor claim is unsupported by source evidence: mark `needs_human_attention=true` and avoid harsh score drop.
3. Add disagreement detector (translator output vs auditor rationale mismatch).

### Phase 4 (Optional): GraphRAG spike

1. 1-week prototype on subset only.
2. Continue only if measurable gain on difficult chunks without major runtime increase.

## 5) Success Metrics (must improve vs baseline)

1. Rewrite loop convergence:
   - reduce active chunks after loop 3 by >= 30%.
   - increase locked chunks at target>=9 by >= 20%.
2. Throughput:
   - p95 chunk processing time bounded (no late-chunk freeze growth trend).
3. Quality:
   - lower false hallucination flags in audited sample set.
   - higher audit score monotonicity across loops (fewer regressions).
4. Robustness:
   - resume reliability after interruption.
   - deterministic behavior with cached retrieval/web context.
5. Model efficiency:
   - >= 70% chunks handled by `8b` path.
   - `30b` usage concentrated on unresolved hard chunks.
   - lower timeout count vs large-model-only baseline.
6. Observability:
   - 100% loops have start/end timestamps and elapsed time in terminal + log.
   - 100% runs produce master log + per-loop logs.
   - no silent stage transitions (all stage boundaries logged).

## 6) Proposed Config Additions

Add (proposed) sections to `config.yaml`:

1. `retrieval.mode: dense|hybrid`
2. `retrieval.rrf_k`, `retrieval.dense_k`, `retrieval.lexical_k`
3. `rerank.enabled`, `rerank.provider`, `rerank.top_n`, `rerank.timeout`
4. `web_search.enabled`, `web_search.provider`, `web_search.timeout`, `web_search.max_results`, `web_search.allowed_domains`, `web_search.cache_path`
5. `self_correction.enabled`, `self_correction.max_retrieval_retries`
6. `model_router.default_model`, `model_router.escalation_model`, `model_router.escalation_after_loops`, `model_router.max_escalations_per_chunk`
7. `model_router.escalate_on_human_attention`, `model_router.escalate_on_persistent_critical_flags`
8. `model_router.fallback_on_timeout`, `model_router.escalation_timeout`
9. `timing.print_loop_start`, `timing.print_loop_elapsed`, `timing.print_cumulative_runtime`
10. `logging.master_log_path`, `logging.loop_log_dir`, `logging.chunk_log_dir`, `logging.tee_terminal_output`

## 6.1) Logging and Timing Design Notes

1. Master-level log captures all stage prints for full-run replay/debug.
2. Loop-level logs isolate each rewrite/audit loop for faster diagnosis.
3. Terminal output should be "tee"-written to both console and corresponding log.
4. Each loop should print and log:
   - `LOOP N START: <ISO timestamp>`
   - `LOOP N END: <ISO timestamp> | elapsed=<seconds>`
   - `RUNTIME TOTAL: <seconds>`
5. Slow-chunk threshold logging:
   - if chunk time exceeds threshold, add explicit warning line in loop log and master log.
6. File layout (proposed):
   - master: `data/output/logs/pipeline_<run_id>.log`
   - loop: `data/output/rewrites/<run_id>/logs/loop_<NN>.log`
   - optional chunk slow/error: `.../logs/chunk_events.log`

## 7) Implementation Priority (recommended)

1. Hybrid retrieval (highest ROI).
2. Internet search with strict gating + caching.
3. Local reranker (optional, stage-specific).
4. Retrieval self-correction.
5. Dual-model routing (`8b` default + selective `30b`).
6. GraphRAG only as later experiment.

## 8) References (primary sources)

1. Chroma keyword search (FTS/BM25-related): https://cookbook.chromadb.dev/strategies/keyword-search/
2. Chroma sparse vector + hybrid search docs: https://docs.trychroma.com/cloud/schema/sparse-vector-search
3. Chroma hybrid search with RRF: https://docs.trychroma.com/cloud/search-api/hybrid-search
4. Chroma BM25 embedding function: https://docs.trychroma.com/integrations/embedding-models/chroma-bm25
5. Elastic RRF reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
6. LangChain EnsembleRetriever: https://api.python.langchain.com/en/latest/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
7. LangChain BM25Retriever: https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.bm25.BM25Retriever.html
8. SentenceTransformers CrossEncoder docs: https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
9. Cohere rerank docs: https://docs.cohere.com/docs/rerank
10. Self-RAG paper: https://arxiv.org/abs/2310.11511
11. CRAG paper: https://arxiv.org/abs/2401.15884
12. GraphRAG paper: https://arxiv.org/abs/2404.16130
13. Microsoft GraphRAG repo: https://github.com/microsoft/graphrag
14. Tavily Search API: https://docs.tavily.com/documentation/api-reference/endpoint/search
