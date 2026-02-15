# Translate Agent

Chunk-first local translation pipeline for long documents (English -> Chinese), with integrated quality loop.

## Language Docs

- English: `README.md`
- Chinese: `README.zh-CN.md`

## Project Summary

Translate Agent is built for OCR-heavy books and long-form manuscripts where consistency and iterative quality control matter.

Core design:

1. Organize source into stable chunk files first.
2. Translate each chunk with glossary + RAG support.
3. Keep first-pass translation style-neutral.
4. Apply optional style only at polish/review stage.
5. Run chunk-level audit and iterative rewrite loop.
6. Re-assemble chunks into book-level markdown outputs.

## Current Pipeline (End-to-End)

`src.pipeline` now runs all stages by default:

1. OCR / input ingestion (`.pdf`, `.md`, `.txt`, or OCR directory)
2. Source chunk organization (`data/output/source_chunks`)
3. Glossary extraction/reuse (`data/glossary.json`)
4. RAG indexing (ChromaDB)
5. Chunk translation (`translator -> reviewer/polisher`)
6. Assembly (`data/output/*_translated.md`)
7. Chunk-level audit (`src.audit`)
8. Iterative rewrite loop (`src.rewrite_audit_loop`)

## Status Snapshot (From Latest Artifacts)

This snapshot is based on current workspace artifacts:

- Pipeline log: `data/output/MinerU_processed_pipeline.log`
- Active rewrite run: `data/output/rewrites/rewrite_loop_run_20260213_174529`
- Loop state: `data/output/rewrites/rewrite_loop_run_20260213_174529/loop_state.json`

Observed status:

1. Rewrite loop progressed to `next_loop=11`.
2. Locked chunks: `22/52` at target score `9`.
3. Last completed loop in history: `loop 10`.
4. Last loop had `16` human-attention chunks and `7` rejected rewrite candidates (guarded acceptance).

## Key Quality Controls

1. Chunk-level source/translation alignment (`source_chunks` + `chunks`).
2. Audit against original source chunks, not only assembled whole-book text.
3. Human-attention flag for low-confidence/ambiguous audit cases.
4. Guarded rewrite acceptance to reduce regression across loops.
5. Per-loop evidence folders (`loop1`, `loop2`, ...) with:
   - loop audit report copy
   - accepted rewrite chunks
   - rejected rewrite candidates

## Main Commands

Run full pipeline from OCR directory (recommended):

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --style "Readable, preserve original tone, localized Chinese"
```

Run full pipeline with neutral polish style:

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --no-style-prompt
```

Disable post-translation quality loop:

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --no-quality-loop
```

Tune rewrite-loop behavior from pipeline:

```bash
python -m src.pipeline \
  --source "data/input/MinerU_processed" \
  --loop-target-score 9 \
  --loop-max-loops 30 \
  --loop-acceptance-min-delta 1
```

Allow auto-rewrite even for human-attention chunks:

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --loop-rewrite-human-attention
```

Run audit/rewrite manually (if needed):

```bash
python -m src.audit --source-chunks-dir "data/output/source_chunks" --chunks-dir "data/output/chunks"
python -m src.rewrite_audit_loop --source-chunks-dir "data/output/source_chunks" --chunks-dir "data/output/chunks"
```

## Important Output Paths

1. Source chunks: `data/output/source_chunks/chunk_XXX.md`
2. First translated chunks: `data/output/chunks/chunk_XXX.md`
3. First assembled book: `data/output/*_translated.md`
4. Rewrite runs: `data/output/rewrites/rewrite_loop_run_YYYYMMDD_HHMMSS`
5. Final rewritten assembled markdown (per run): `.../rewritten_translated.md`
6. Pipeline log: `data/output/*_pipeline.log`

## Directory Layout

Keep runtime artifacts under `data/` only.

1. `data/input/`: source files (pdf/md/txt/OCR folders)
2. `data/output/`: all generated runtime artifacts (chunks/audits/rewrites/logs/smoke tests)
3. `data/output/logs/permission_reports/`: permission diagnostics reports
4. `src/`: application code
5. `scripts/`: operational helpers (smoke tests, converters, exports, diagnostics)

## Dependencies

1. Python dependencies:

```bash
pip install -r requirements.txt
```

2. Ollama model endpoint configured in `config.yaml` (default `qwen3:8b`, with `qwen3:30b` escalation).
