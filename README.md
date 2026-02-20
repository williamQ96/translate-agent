# Translate Agent

Chunk-first, local-first pipeline for long-form book translation (English -> Chinese), with iterative quality control.
Chinese docs: `README.zh-CN.md` (alias: `readme_cn.md`).

![logo_icon_text](img/logo_icon_texted.png)

## What We Ship Today

1. End-to-end chunk pipeline: translate -> audit -> rewrite -> re-audit -> assemble.
2. Dual-model routing:
   - default fast path: `qwen3-8b`
   - hard-chunk escalation: `qwen3-32b`
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

![pipeline](img/pipeline%20figure.png)

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

Build a delivery package (book + latest chunks + readable audit report):

```bash
python scripts/build_delivery_package.py --run-dir "data/output/rewrites/rewrite_loop_run_YYYYMMDD_HHMMSS"
```

The delivery script creates:

1. `rewritten_translated.md` at delivery root
2. `original_chunks/` (source English chunks)
3. `translated_chunks/` (effective latest chunks: base + rewrite overlay)
4. latest full audit JSON + readable Chinese report (`.md`) + table (`.csv`)
5. glossary JSON + manifest

## Repository Hygiene

1. Generated runtime data under `data/` is local-only and gitignored.
2. Keep root focused on source code and docs (`src/`, `scripts/`, `config/`, `img/`, README files).
3. Use delivery bundles for handoff instead of committing run artifacts.

## Runtime Backend (v0.32: LM Studio)

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Start LM Studio server (GUI or headless), OpenAI-compatible endpoint:

- default expected endpoint: `http://127.0.0.1:1234/v1`
- keep hardware strategy as `Priority Order` (5090 first, 4070Ti second)
- recommended:
  - `Limit Model Offload to Dedicated GPU Memory: ON`
  - `Offload KV Cache to GPU Memory: ON`
  - Guardrails: `Relaxed` (or `Balanced` if you prefer more safety)

3. Check model IDs exposed by LM Studio:

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\check_lmstudio_models.ps1
```

4. Ensure `config.yaml` model IDs match `/v1/models` output:

- `model.name` (default path)
- `model_router.default_model` (8B)
- `model_router.escalation_model` (32B)

## References

1. Beads: https://github.com/steveyegge/beads
2. Beads docs: https://beads.ignition.dev/
3. DeepL document translation: https://support.deepl.com/hc/en-us/articles/360020698639-Translate-documents
4. Google Docs translation: https://support.google.com/docs/answer/187189
5. Amazon KDP translation feature: https://kdp.amazon.com/en_US/help/topic/GTH4C7FLRNCXSWJW
