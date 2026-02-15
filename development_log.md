# Development Log

## 2026-02-10 — Pipeline Optimization

### Problem
First translation run took ~99 minutes per chunk (50 chunks, projecting to ~82 hours total).
Each chunk required 4 serial LLM calls: translator → critic → polisher → checker.
All 1048 glossary terms were injected into every prompt regardless of relevance.

### Changes Made

| File | Change | Impact |
|---|---|---|
| `src/agents/prompts.py` | Merged `CRITIC_PROMPT` + `POLISHER_PROMPT` → `REVIEWER_PROMPT` | −1 LLM call/chunk (~25min) |
| `src/agents/workflow.py` | Reduced graph: translator → reviewer (2 steps) | −1 LLM call/chunk |
| `src/agents/state.py` | Removed `critique` field from state | cleanup |
| `src/knowledge/glossary.py` | Added `get_relevant_glossary_text(source_text)` | Smaller prompts → faster inference |
| `src/pipeline.py` | Per-chunk .md output, `--skip N` flag, async checker, trimmed glossary | All optimizations combined |

### Design Decisions
- **Merged critic+polisher**: The critic's numbered suggestions were only consumed by the polisher. Combining them into one "review and correct" step produces equivalent output with one fewer round-trip.
- **Async checker**: Quality checking runs in a ThreadPoolExecutor. While Ollama serializes GPU inference, the checker runs on the previous chunk's output while the current chunk's translator is executing — effectively free when the GPU is otherwise idle between steps.
- **Glossary trimming**: `get_relevant_glossary_text()` does a case-insensitive substring match of each term against the chunk source. This reduces prompt tokens from ~1048 terms to ~20-50 relevant terms per chunk.
- **Per-chunk output**: Each chunk is written to `data/output/chunks/chunk_001.md` immediately. This allows inspection during a long run.

### Expected Performance
~99min/chunk → ~40-50min/chunk (55% reduction)

### Command to Run
```bash
translateagent\Scripts\python -m src.pipeline --source "data/input/MinerU_processed" --skip 3
```

---

## 2026-02-10 10:56 — Glossary Bypass

### Problem
`stage_glossary` was re-scanning all 30 sections every run, even though `data/glossary.json` already had 1048 terms. The LLM also produced invalid JSON on some sections, causing parsing errors.

### Fix
Modified `stage_glossary` to check if the glossary file already has terms. If so, it reuses it immediately (0 LLM calls, instant). Extraction only runs on a truly empty glossary.

---

## 2026-02-10 11:00 — Checker Agent Removed

### Rationale
The checker added 1 extra LLM call per chunk (~25min) but only produced 3 scores (完整性/流畅度/准确性) that were logged but never acted upon — the pipeline didn't retry on low scores. Pure cost with no benefit.

### Changes
- Removed `CHECKER_PROMPT`, `run_checker()`, `ThreadPoolExecutor`, and all async checker logic from `pipeline.py`
- Per-chunk flow is now: **translator → reviewer** only (2 LLM calls total)

---

## 2026-02-10 11:00 — Glossary Cleanup

### Audit Results
| Category | Count |
|---|---|
| Total | 1175 |
| Valid translations | 182 |
| Identical EN=CN (useless) | 866 |
| Needs review (junk) | 62 |
| Empty | 65 |

### Actions
- Cleaned glossary: **1175 → 182 entries**
- Backup saved to `data/glossary_backup.json`
- Disabled `update_from_translation()` in `glossary.py` (was the source of `[NEEDS REVIEW]` junk)
- `get_relevant_glossary_text()` now operates on clean data only
