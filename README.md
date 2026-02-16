# Translate Agent

Chunk-first, local-first pipeline for long-form book translation (English -> Chinese), with iterative quality control.

## What We Ship Today

1. End-to-end chunk pipeline: translate -> audit -> rewrite -> re-audit -> assemble.
2. Dual-model routing:
   - default fast path: `qwen3:8b`
   - hard-chunk escalation: `qwen3:30b`
3. Hybrid retrieval stack:
   - dense retrieval + lexical retrieval + fusion
   - glossary-aware prompting
4. Chunk-level audit and locking:
   - lock chunks that meet target score
   - skip low-value rewrite candidates
5. Operational scripts:
   - smoke tests
   - artifact export
   - audit JSON -> readable report/CSV
   - output archiving to legacy folders
   - permission diagnostics

## Current Pipeline

`src.pipeline` runs these stages by default:

1. OCR/input ingestion (`.pdf`, `.md`, `.txt`, or OCR directory)
2. Source chunk organization (`data/output/source_chunks`)
3. Glossary extraction/reuse (`data/glossary.json`)
4. RAG indexing
5. Chunk translation + polish
6. Assembly (`data/output/*_translated.md`)
7. Chunk-level audit (`src.audit`)
8. Iterative rewrite loop (`src.rewrite_audit_loop`)

## Interactive GUI (Planned)

Interactive GUI is planned (not shipped yet). Proposed scope:

1. Start/stop/resume pipeline runs
2. Live loop progress (per-loop scores, locked chunks, time)
3. Human-attention chunk review panel
4. One-click export of delivery bundle (full markdown + latest chunks)

## Beads Integration Plan (Planned)

`beads` is being evaluated as an optional memory/trace layer, not as a replacement for RAG.

Planned usage:

1. Persist per-chunk rewrite decisions and audit outcomes
2. Persist human-review resolutions and term decisions
3. Reuse memory on recurrent hard chunks

Design constraint:

1. Fail-open integration (pipeline still runs when beads is unavailable)

## Domain Positioning (Evidence-Based)

Short answer: this project is strong and differentiated in workflow design, but we should not claim global uniqueness yet.

Why:

1. Document-level AI translation products already exist (DeepL document translation, Google Docs translation, Kindle/Amazon translation features).
2. The domain is active and competitive.
3. Our differentiator is the engineering stack:
   - chunk-to-chunk source auditing
   - iterative lock/rewrite loop
   - local-first dual-model routing
   - artifact traceability and reproducible runs

Recommended claim wording:

1. "A robust open-source workflow for publication-oriented long-book translation with iterative quality control."
2. Avoid claiming "world-first" or "industry-leading" without benchmark studies.

## Directory Layout (Publish-Friendly)

Keep root clean; keep runtime artifacts under `data/`.

1. `src/` - core application code
2. `scripts/` - operational scripts and debug tools
3. `config/` - runtime config files (for example `config/magic-pdf.json`)
4. `data/input/` - source files
5. `data/output/` - generated artifacts (chunks, audits, rewrites, logs, smoke runs)
6. `data/output/legacy_data/` - archived historical runs

## Main Commands

Run full pipeline:

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --style "Readable, preserve original tone, localized Chinese"
```

Run smoke test (3-4 chunks):

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_4chunks.ps1 -ChunkIds "1,2,3,4" -MaxLoops 5
```

Archive current output artifacts to legacy:

```bash
python scripts/archive_output_artifacts.py --run-name "work_YYYYMMDD_HHMMSS" --include-ocr
```

Convert audit JSON to readable Chinese markdown + CSV:

```bash
python scripts/audit_json_to_cn_report.py --input "path/to/audit_loop_XX.json"
```

## Dependencies

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure Ollama models are available:

```bash
ollama pull qwen3:8b
ollama pull qwen3:30b
```

## References

1. Beads: https://github.com/steveyegge/beads
2. Beads docs: https://beads.ignition.dev/
3. DeepL document translation: https://support.deepl.com/hc/en-us/articles/360020698639-Translate-documents
4. Google Docs translation: https://support.google.com/docs/answer/187189
5. Amazon KDP translation feature: https://kdp.amazon.com/en_US/help/topic/GTH4C7FLRNCXSWJW
