"""
pipeline.py - Full end-to-end translation pipeline.

Usage:
    python -m src.pipeline --source "path/to/file.pdf"
    python -m src.pipeline --source "path/to/file.md"   (skip OCR)
    python -m src.pipeline --source "data/input/MinerU_processed"  (directory)
    python -m src.pipeline --source "..." --skip 3      (skip first N chunks)

Pipeline stages:
    1. OCR (PDF -> Markdown) - skipped if input is already .md/.txt or OCR directory
    2. Source chunk organize - build stable source chunk files
    3. Glossary extraction   - extract/reuse terms for consistency
    4. RAG indexing          - index source corpus for cross-reference
    5. Translation           - Translator -> Reviewer per chunk
    6. Assembly              - stitch chunks into final output
    7. Audit                 - chunk-level audit over translated chunks
    8. Rewrite loop          - iterative rewrite+audit until convergence/limit
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from src.agents.workflow import build_graph
from src.knowledge.extractor import TermExtractor
from src.knowledge.glossary import GlossaryManager
from src.ocr.pdf_processor import PDFProcessor
from src.rag.store import RAGStore
from src.utils.config_loader import load_config


def _setup_console_encoding() -> None:
    """Best effort UTF-8 console output on Windows terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks by approximate token count."""
    words = text.split()
    words_per_chunk = int(chunk_size * 1.3)
    words_overlap = int(overlap * 1.3)

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunks.append(" ".join(words[start:end]))
        start = end - words_overlap

    return chunks


def log_progress(log_path: str, stage: str, detail: str, status: str = "OK") -> None:
    """Append a progress entry to the pipeline log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {status} {stage}: {detail}\n"
    print(f"  {status} {stage}: {detail}")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(entry)


def _run_subprocess_stage(log_path: str, stage: str, cmd: list[str]) -> None:
    """Run a subprocess stage and stream output to console + pipeline log."""
    cmd_line = subprocess.list2cmdline(cmd)
    log_progress(log_path, stage, f"Running: {cmd_line}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if process.stdout is not None:
        with open(log_path, "a", encoding="utf-8") as log_file:
            for line in process.stdout:
                text = line.rstrip("\n")
                if not text:
                    continue
                print(f"    {text}")
                log_file.write(f"    {text}\n")

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{stage} failed with exit code {return_code}")

    log_progress(log_path, stage, "Done")


def stage_post_quality_loop(
    source_chunks_dir: str,
    chunks_dir: str,
    output_root: str,
    target_score: int,
    max_loops: int,
    acceptance_min_delta: int,
    rewrite_human_attention: bool,
    log_path: str,
) -> str:
    """Stage 7-8: run chunk-level audit then iterative rewrite loop."""
    os.makedirs(output_root, exist_ok=True)

    audit_cmd = [
        sys.executable,
        "-m",
        "src.audit",
        "--source-chunks-dir",
        source_chunks_dir,
        "--chunks-dir",
        chunks_dir,
    ]
    _run_subprocess_stage(log_path, "STAGE 7 - AUDIT", audit_cmd)

    loop_cmd = [
        sys.executable,
        "-m",
        "src.rewrite_audit_loop",
        "--source-chunks-dir",
        source_chunks_dir,
        "--chunks-dir",
        chunks_dir,
        "--output-root",
        output_root,
        "--target-score",
        str(target_score),
        "--max-loops",
        str(max_loops),
        "--acceptance-min-delta",
        str(acceptance_min_delta),
    ]
    if rewrite_human_attention:
        loop_cmd.append("--rewrite-human-attention")

    _run_subprocess_stage(log_path, "STAGE 8 - REWRITE LOOP", loop_cmd)

    candidates = [path for path in glob.glob(os.path.join(output_root, "rewrite_loop_run_*")) if os.path.isdir(path)]
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def stage_ocr(source_path: str, log_path: str) -> str:
    """Stage 1: Convert PDF to Markdown via MinerU, or collate a processed directory."""
    log_progress(log_path, "STAGE 1 - OCR", f"Processing {os.path.basename(source_path)}...")

    processor = PDFProcessor()
    md_path = processor.process_pdf(source_path)

    with open(md_path, "r", encoding="utf-8") as file:
        content = file.read()

    log_progress(log_path, "STAGE 1 - OCR", f"Done. {len(content)} chars extracted -> {md_path}")
    return md_path


def _parse_page_number(name: str) -> int:
    match = re.match(r"^(\d+)", name)
    if match:
        return int(match.group(1))
    return 999999


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _split_units(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if parts:
        return parts
    cleaned = text.strip()
    return [cleaned] if cleaned else []


def _collect_units_from_source(source_path: str) -> tuple[list[str], str]:
    """
    Collect source units without creating a full aggregated markdown artifact.
    Returns (units, corpus_text_for_rag_glossary).
    """
    if os.path.isdir(source_path):
        subdirs = [name for name in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, name))]
        subdirs.sort(key=_parse_page_number)
        units: list[str] = []
        page_texts: list[str] = []
        for subdir in subdirs:
            full_md = os.path.join(source_path, subdir, "full.md")
            if not os.path.exists(full_md):
                continue
            page_text = _read_text(full_md).strip()
            if not page_text:
                continue
            page_texts.append(page_text)
            units.extend(_split_units(page_text))
        return units, "\n\n".join(page_texts)

    text = _read_text(source_path)
    return _split_units(text), text


def _split_unit_to_limit(unit: str, max_words: int) -> list[str]:
    words = unit.split()
    if len(words) <= max_words:
        return [unit]
    parts = []
    for start in range(0, len(words), max_words):
        segment = " ".join(words[start : start + max_words]).strip()
        if segment:
            parts.append(segment)
    return parts


def _build_chunks_from_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    words_per_chunk = int(chunk_size * 1.3)
    words_overlap = int(overlap * 1.3)
    if words_per_chunk <= 0:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue
        for part in _split_unit_to_limit(unit, max_words=words_per_chunk):
            part_words = len(part.split())
            projected = current_words + part_words
            if current_parts and projected > words_per_chunk:
                chunk_text = "\n\n".join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                if words_overlap > 0 and chunk_text:
                    overlap_words = chunk_text.split()[-words_overlap:]
                    overlap_text = " ".join(overlap_words).strip()
                    current_parts = [overlap_text] if overlap_text else []
                    current_words = len(overlap_words)
                else:
                    current_parts = []
                    current_words = 0

            current_parts.append(part)
            current_words += part_words

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def stage_organize_chunks(source_path: str, config: dict, log_path: str) -> tuple[list[str], str, str]:
    """
    Stage 2: Organize source content into reusable source chunks.
    Returns (source_chunks, source_chunks_dir, source_corpus_text).
    """
    log_progress(log_path, "STAGE 2 - CHUNK ORGANIZE", "Organizing source into chunk files...")

    units, corpus_text = _collect_units_from_source(source_path)
    if not units:
        raise RuntimeError(f"No source units found for chunking: {source_path}")

    chunk_config = config.get("chunking", {})
    chunks = _build_chunks_from_units(
        units,
        chunk_size=chunk_config.get("chunk_size", 2000),
        overlap=chunk_config.get("overlap", 200),
    )
    if not chunks:
        raise RuntimeError("Chunk organizer produced zero chunks")

    output_dir = config["directories"]["output"]
    source_chunks_dir = os.path.join(output_dir, "source_chunks")
    os.makedirs(source_chunks_dir, exist_ok=True)

    for name in os.listdir(source_chunks_dir):
        if name.startswith("chunk_") and name.endswith(".md"):
            os.remove(os.path.join(source_chunks_dir, name))

    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = os.path.join(source_chunks_dir, f"chunk_{idx:03d}.md")
        with open(chunk_path, "w", encoding="utf-8") as file:
            file.write(f"<!-- Source Chunk {idx}/{len(chunks)} -->\n\n")
            file.write(chunk)

    manifest_path = os.path.join(source_chunks_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "source": source_path,
                "total_units": len(units),
                "total_chunks": len(chunks),
                "chunk_size": chunk_config.get("chunk_size", 2000),
                "overlap": chunk_config.get("overlap", 200),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    log_progress(
        log_path,
        "STAGE 2 - CHUNK ORGANIZE",
        f"Done. {len(units)} units -> {len(chunks)} source chunks at {source_chunks_dir}",
    )
    return chunks, source_chunks_dir, corpus_text


def stage_glossary(source_corpus_text: str, config: dict, log_path: str) -> GlossaryManager:
    """Stage 3: Extract terminology from source corpus unless glossary already exists."""
    glossary_path = config["directories"].get("glossary", "data/glossary.json")
    glossary_manager = GlossaryManager(glossary_path)

    existing_count = glossary_manager.get_term_count()
    if existing_count > 0:
        log_progress(
            log_path,
            "STAGE 3 - GLOSSARY",
            f"Reusing existing glossary: {existing_count} terms from {glossary_path}",
        )
        return glossary_manager

    log_progress(log_path, "STAGE 3 - GLOSSARY", "Scanning source corpus for terminology...")

    extractor = TermExtractor()

    try:
        terms = extractor.extract_from_full_document(source_corpus_text)
        if terms:
            added = glossary_manager.add_terms(terms)
            log_progress(
                log_path,
                "STAGE 3 - GLOSSARY",
                f"Found {len(terms)} terms, {added} new -> {glossary_path}",
            )
        else:
            log_progress(
                log_path,
                "STAGE 3 - GLOSSARY",
                "No terms extracted (continuing without glossary)",
                "WARN",
            )
    except Exception as exc:  # noqa: BLE001
        log_progress(
            log_path,
            "STAGE 3 - GLOSSARY",
            f"Extraction failed: {exc}. Continuing without glossary.",
            "WARN",
        )

    log_progress(
        log_path,
        "STAGE 3 - GLOSSARY",
        f"Active glossary: {glossary_manager.get_term_count()} terms with translations",
    )
    return glossary_manager


def stage_rag_index(source_corpus_text: str, log_path: str) -> RAGStore:
    """Stage 4: Index source corpus into ChromaDB for cross-reference."""
    log_progress(log_path, "STAGE 4 - RAG INDEX", "Indexing source corpus into vector store...")

    rag_store = RAGStore()
    rag_store.index_document(source_corpus_text)

    log_progress(
        log_path,
        "STAGE 4 - RAG INDEX",
        f"Done. {rag_store.collection.count()} paragraphs indexed in ChromaDB",
    )
    return rag_store


def _trim_for_prompt(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text[:max_chars]


def _split_for_translation(text: str, max_chars: int = 3200) -> list[str]:
    """Split oversized chunks for safer translation calls."""

    def pack_units(units: list[str], sep: str) -> list[str]:
        packed: list[str] = []
        current: list[str] = []
        current_len = 0
        sep_len = len(sep)
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            unit_len = len(unit)
            projected = current_len + (sep_len if current else 0) + unit_len
            if current and projected > max_chars:
                packed.append(sep.join(current).strip())
                current = [unit]
                current_len = unit_len
            else:
                current.append(unit)
                current_len = projected if current_len else unit_len
        if current:
            packed.append(sep.join(current).strip())
        return [item for item in packed if item]

    def hard_split(unit: str) -> list[str]:
        step = max(800, max_chars)
        return [unit[idx : idx + step].strip() for idx in range(0, len(unit), step) if unit[idx : idx + step].strip()]

    raw = text.strip()
    if not raw:
        return [""]

    paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
    if not paragraphs:
        return hard_split(raw)

    normalized_units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            normalized_units.append(para)
        else:
            lines = [line.strip() for line in para.splitlines() if line.strip()]
            if len(lines) > 1:
                normalized_units.extend(pack_units(lines, "\n"))
            else:
                sentences = [s.strip() for s in re.split(r"(?<=[ã€‚ï¼ï¼Ÿ.!?;ï¼›:ï¼š])\s+", para) if s.strip()]
                if len(sentences) > 1:
                    normalized_units.extend(pack_units(sentences, " "))
                else:
                    normalized_units.extend(hard_split(para))

    segments = pack_units(normalized_units, "\n\n")
    if segments:
        return segments
    return hard_split(raw)


def _invoke_translation_with_retry(
    app,
    segment: str,
    chunk_id: int,
    glossary_text: str,
    rag_context: str,
    polish_style: str,
    max_attempts: int,
) -> tuple[str | None, str | None]:
    """Run translator->reviewer chain with bounded retries."""
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        state = {
            "chunk_id": chunk_id,
            "source_text": segment,
            "glossary": glossary_text if attempt == 1 else _trim_for_prompt(glossary_text, max_chars=800),
            "rag_context": rag_context if attempt == 1 else "",
            "polish_style": polish_style,
            "draft_translation": None,
            "final_translation": None,
            "iteration_count": 0,
        }
        try:
            result = app.invoke(state)
            return result["final_translation"], None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < max_attempts:
                print(f"retrying after error: {last_error}")
    return None, last_error


def stage_translate(
    source_chunks: list[str],
    base_name: str,
    glossary_manager: GlossaryManager,
    rag_store: RAGStore,
    config: dict,
    log_path: str,
    skip_chunks: int = 0,
    polish_style: str = "",
) -> list[str]:
    """Stage 5: Translate each source chunk with checkpointing."""
    log_progress(log_path, "STAGE 5 - TRANSLATION", "Loading source chunks and building pipeline...")

    chunks = source_chunks
    output_dir = config["directories"]["output"]
    progress_path = os.path.join(output_dir, f"{base_name}_progress.json")
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    results: list[str] = []
    start_chunk = 0

    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as file:
                saved_state = json.load(file)
            results = saved_state.get("results", [])
            if len(results) > len(chunks):
                log_progress(
                    log_path,
                    "RESUME",
                    f"Progress has {len(results)} chunks but current source has {len(chunks)}. Resetting progress.",
                    "WARN",
                )
                results = []
            start_chunk = len(results)
            if start_chunk > 0:
                log_progress(
                    log_path,
                    "RESUME",
                    f"Found {start_chunk} completed chunks in progress file",
                )
        except Exception as exc:  # noqa: BLE001
            log_progress(log_path, "RESUME", f"Failed to load progress: {exc}. Starting fresh.", "WARN")

    if start_chunk == 0:
        for name in os.listdir(chunks_dir):
            if name.startswith("chunk_") and name.endswith(".md"):
                os.remove(os.path.join(chunks_dir, name))

    if skip_chunks > start_chunk:
        log_progress(log_path, "SKIP", f"Skipping first {skip_chunks} chunks (--skip flag)")
        while len(results) < skip_chunks:
            results.append(f"[SKIPPED - chunk {len(results) + 1}]")
        start_chunk = skip_chunks

    log_progress(
        log_path,
        "STAGE 5 - TRANSLATION",
        f"Split into {len(chunks)} chunks. Starting from chunk {start_chunk + 1}...",
    )

    app = build_graph()
    total_start = time.time()
    translation_cfg = config.get("translation", {})
    max_attempts = int(translation_cfg.get("max_attempts", 2))
    rag_k = int(translation_cfg.get("rag_k", 3))
    glossary_max_chars = int(translation_cfg.get("glossary_max_chars", 1600))
    rag_max_chars = int(translation_cfg.get("rag_max_chars", 2200))
    segment_threshold_chars = int(translation_cfg.get("segment_threshold_chars", 7000))
    segment_size_chars = int(translation_cfg.get("segment_size_chars", 3200))

    for index in range(start_chunk, len(chunks)):
        chunk = chunks[index]
        chunk_start = time.time()
        segment_mode = len(chunk) > segment_threshold_chars
        segments = _split_for_translation(chunk, max_chars=segment_size_chars) if segment_mode else [chunk]
        if segment_mode:
            log_progress(
                log_path,
                f"CHUNK {index + 1}/{len(chunks)}",
                f"Segment mode enabled ({len(chunk)} chars -> {len(segments)} segments)",
            )

        segment_outputs: list[str] = []
        final_error: str | None = None
        total_terms = 0

        for seg_idx, segment in enumerate(segments, start=1):
            glossary_text = _trim_for_prompt(glossary_manager.get_relevant_glossary_text(segment), glossary_max_chars)
            term_count = len(glossary_text.strip().split("\n")) if glossary_text.strip() else 0
            total_terms += term_count

            rag_context = _trim_for_prompt(rag_store.retrieve_context(segment, k=rag_k), rag_max_chars)
            rag_note = f"({len(rag_context)} chars of context)" if rag_context else "(no extra context)"
            seg_label = f" segment {seg_idx}/{len(segments)}" if len(segments) > 1 else ""

            log_progress(
                log_path,
                f"CHUNK {index + 1}/{len(chunks)}",
                f"Translating{seg_label} {rag_note}, {term_count} relevant glossary terms...",
            )

            translated, error = _invoke_translation_with_retry(
                app=app,
                segment=segment,
                chunk_id=index,
                glossary_text=glossary_text,
                rag_context=rag_context,
                polish_style=polish_style,
                max_attempts=max_attempts,
            )
            if translated is None:
                final_error = error or "unknown error"
                break
            segment_outputs.append(translated.strip())

        if final_error is not None:
            log_progress(
                log_path,
                f"CHUNK {index + 1}/{len(chunks)}",
                f"FAILED: {final_error}. Saving progress and continuing.",
                "ERR",
            )
            final = f"[TRANSLATION ERROR: {final_error}]"
        else:
            final = "\n\n".join(segment_outputs).strip()
            if not final:
                final = "[TRANSLATION ERROR: empty output]"

        elapsed = time.time() - chunk_start

        glossary_manager.update_from_translation(chunk, final)

        chunk_filename = f"chunk_{index + 1:03d}.md"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as file:
            file.write(
                f"<!-- Chunk {index + 1}/{len(chunks)} | {elapsed:.0f}s | {total_terms} glossary terms -->\n\n"
            )
            file.write(final)

        log_progress(
            log_path,
            f"CHUNK {index + 1}/{len(chunks)}",
            f"Done in {elapsed:.0f}s -> {chunk_filename}",
        )

        results.append(final)

        try:
            with open(progress_path, "w", encoding="utf-8") as file:
                json.dump({"results": results}, file, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN Failed to save progress: {exc}")

    total_elapsed = time.time() - total_start
    log_progress(log_path, "STAGE 5 - TRANSLATION", f"All chunks done in {total_elapsed:.0f}s")

    return results


def stage_assemble(results: list[str], output_path: str, log_path: str) -> None:
    """Stage 6: Assemble translated chunks into final Markdown."""
    valid_results = [
        result
        for result in results
        if not result.startswith("[SKIPPED") and not result.startswith("[TRANSLATION ERROR")
    ]

    log_progress(log_path, "STAGE 6 - ASSEMBLY", f"Writing {len(valid_results)} chunks to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n\n".join(valid_results))

    file_size = os.path.getsize(output_path)
    log_progress(log_path, "STAGE 6 - ASSEMBLY", f"Done. Output: {output_path} ({file_size} bytes)")


def _resolve_polish_style(style: str, prompt_if_empty: bool = True) -> str:
    """Resolve polish style once at pipeline start."""
    if style and style.strip():
        return style.strip()

    if not prompt_if_empty:
        return ""

    print("\nPolish style (optional, applied after first-pass translation).")
    print("Example: èƒ½çœ‹æ‡‚ï¼Œä¿æŒåŽŸä½œé£Žæ ¼ï¼Œä¸­æ–‡æœ¬åœŸåŒ–")
    print("Press Enter to keep neutral/no extra style.")
    try:
        value = input("Style> ").strip()
        return value
    except EOFError:
        return ""


def run_pipeline(
    source_path: str,
    skip_chunks: int = 0,
    polish_style: str = "",
    prompt_style: bool = True,
    run_quality_loop: bool = True,
    loop_target_score: int = 9,
    loop_max_loops: int = 30,
    loop_acceptance_min_delta: int = 1,
    loop_rewrite_human_attention: bool = False,
) -> None:
    """Run the full translation pipeline."""
    config = load_config()
    output_dir = config["directories"]["output"]

    if os.path.isdir(source_path):
        base_name = os.path.basename(source_path.rstrip("/\\"))
    else:
        base_name = os.path.splitext(os.path.basename(source_path))[0]

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{base_name}_pipeline.log")

    resolved_style = _resolve_polish_style(polish_style, prompt_if_empty=prompt_style)
    style_label = resolved_style if resolved_style else "NONE (neutral)"

    with open(log_path, "w", encoding="utf-8") as file:
        file.write("=== Translate Agent Pipeline ===\n")
        file.write(f"Source: {source_path}\n")
        file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Model: {config['model']['name']}\n")
        file.write(f"Skip: {skip_chunks} chunks\n")
        file.write(f"Polish Style: {style_label}\n")
        file.write(f"{'=' * 40}\n\n")

    print(f"\n{'=' * 50}")
    print("  TRANSLATE AGENT PIPELINE")
    print(f"  Source: {source_path}")
    print(f"  Model:  {config['model']['name']}")
    print(f"  Skip:   {skip_chunks} chunks")
    print(f"  Style:  {style_label}")
    print(f"  Post Quality Loop: {'ON' if run_quality_loop else 'OFF'}")
    print(f"{'=' * 50}\n")

    pipeline_start = time.time()

    extension = os.path.splitext(source_path)[1].lower()
    if os.path.isdir(source_path) or extension == ".pdf":
        if os.path.isdir(source_path):
            source_ref = source_path
            log_progress(log_path, "STAGE 1 - OCR", "Skipped combine (using OCR directory directly)")
        else:
            source_ref = stage_ocr(source_path, log_path)
    elif extension in (".md", ".txt"):
        source_ref = source_path
        log_progress(log_path, "STAGE 1 - OCR", f"Skipped (input is already {extension})")
    else:
        raise ValueError(f"Unsupported file type: {extension}. Use .pdf, .md, .txt or directory")

    source_chunks, source_chunks_dir, source_corpus = stage_organize_chunks(source_ref, config, log_path)
    glossary_manager = stage_glossary(source_corpus, config, log_path)
    rag_store = stage_rag_index(source_corpus, log_path)
    results = stage_translate(
        source_chunks,
        base_name=base_name,
        glossary_manager=glossary_manager,
        rag_store=rag_store,
        config=config,
        log_path=log_path,
        skip_chunks=skip_chunks,
        polish_style=resolved_style,
    )

    output_path = os.path.join(output_dir, f"{base_name}_translated.md")
    stage_assemble(results, output_path, log_path)

    rewrite_run_dir = ""
    if run_quality_loop:
        rewrite_run_dir = stage_post_quality_loop(
            source_chunks_dir=source_chunks_dir,
            chunks_dir=os.path.join(output_dir, "chunks"),
            output_root=os.path.join(output_dir, "rewrites"),
            target_score=loop_target_score,
            max_loops=loop_max_loops,
            acceptance_min_delta=loop_acceptance_min_delta,
            rewrite_human_attention=loop_rewrite_human_attention,
            log_path=log_path,
        )

    rag_store.clear()

    total = time.time() - pipeline_start
    summary = f"Pipeline complete in {total:.0f}s. Output: {output_path}"
    if rewrite_run_dir:
        summary += f" | Rewrite loop: {rewrite_run_dir}"
    log_progress(log_path, "COMPLETE", summary)

    print(f"\n{'=' * 50}")
    print(f"  DONE in {total:.0f}s")
    print(f"  Output:  {output_path}")
    print(f"  Source:  {source_chunks_dir}")
    print("  Chunks:  data/output/chunks/")
    if rewrite_run_dir:
        print(f"  Rewrite: {rewrite_run_dir}")
    print(f"  Log:     {log_path}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    _setup_console_encoding()
    parser = argparse.ArgumentParser(description="Translate Agent - Full Pipeline")
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Path to source file (.pdf, .md, .txt) or source directory",
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip first N chunks")
    parser.add_argument(
        "--style",
        default="",
        help="Optional polish style injected after first-pass translation",
    )
    parser.add_argument(
        "--no-style-prompt",
        action="store_true",
        help="Do not prompt for style when --style is empty",
    )
    parser.add_argument(
        "--no-quality-loop",
        action="store_true",
        help="Disable post-translation chunk-level audit + rewrite loop",
    )
    parser.add_argument(
        "--loop-target-score",
        type=int,
        default=9,
        help="Target score for iterative rewrite loop",
    )
    parser.add_argument(
        "--loop-max-loops",
        type=int,
        default=30,
        help="Maximum rewrite loops",
    )
    parser.add_argument(
        "--loop-acceptance-min-delta",
        type=int,
        default=1,
        help="Minimum score improvement required to accept rewrite",
    )
    parser.add_argument(
        "--loop-rewrite-human-attention",
        action="store_true",
        help="Allow rewriting chunks marked as human-attention",
    )
    arguments = parser.parse_args()

    run_pipeline(
        arguments.source,
        skip_chunks=arguments.skip,
        polish_style=arguments.style,
        prompt_style=not arguments.no_style_prompt,
        run_quality_loop=not arguments.no_quality_loop,
        loop_target_score=arguments.loop_target_score,
        loop_max_loops=arguments.loop_max_loops,
        loop_acceptance_min_delta=arguments.loop_acceptance_min_delta,
        loop_rewrite_human_attention=arguments.loop_rewrite_human_attention,
    )
