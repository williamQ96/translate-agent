"""
rewrite_audit.py - Rewrite flagged chunks from latest audit and audit rewritten output.

Usage:
    python -m src.rewrite_audit --source "data/output/ocr/MinerU_processed_combined.md"
"""

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime

from src.agents.writer import WriterAgent
from src.audit import audit_chunk, get_llm, load_chunks, load_source_corpus
from src.knowledge.glossary import GlossaryManager
from src.rag.store import RAGStore
from src.utils.config_loader import load_config


def _latest_audit_report(audit_dir: str) -> str:
    reports = [
        os.path.join(audit_dir, name)
        for name in os.listdir(audit_dir)
        if name.endswith(".json") and os.path.isfile(os.path.join(audit_dir, name))
    ]
    if not reports:
        # Backward-compatible fallback used by earlier audit flows.
        fallback = os.path.join(os.path.dirname(audit_dir), "chunks", "audit_report.json")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(
            f"No audit report JSON files found in: {audit_dir}. "
            f"Run `python -m src.audit --source <source_md>` first or pass --report explicitly."
        )
    reports.sort(key=os.path.getmtime, reverse=True)
    return reports[0]


def _resolve_report_path(report: str, audit_dir: str) -> str:
    if not report:
        return _latest_audit_report(audit_dir)

    if os.path.exists(report):
        return report

    placeholder_tokens = ("YYYY", "MM", "DD", "HHMMSS")
    if any(token in report for token in placeholder_tokens):
        fallback = _latest_audit_report(audit_dir)
        print(f"WARN: Placeholder report path detected, using latest audit report: {fallback}")
        return fallback

    raise FileNotFoundError(
        f"Report file not found: {report}. "
        "Pass a real JSON path, or omit --report to auto-use latest in --audit-dir."
    )


def _strip_chunk_header(chunk_text: str) -> str:
    lines = chunk_text.splitlines()
    if lines and lines[0].startswith("<!--"):
        return "\n".join(lines[1:]).lstrip("\n")
    return chunk_text.strip()


def _is_rewritten_chunk(chunk_path: str) -> bool:
    """A rewritten chunk contains a header line with 'REWRITTEN from ...'."""
    if not os.path.exists(chunk_path):
        return False
    try:
        with open(chunk_path, "r", encoding="utf-8") as file:
            first_line = file.readline().strip()
        return first_line.startswith("<!--") and "REWRITTEN from" in first_line
    except Exception:  # noqa: BLE001
        return False


def _completed_rewritten_chunks(chunks_dir: str) -> set[int]:
    """Detect completed rewritten chunks by file header markers."""
    completed: set[int] = set()
    for name in os.listdir(chunks_dir):
        if not (name.startswith("chunk_") and name.endswith(".md")):
            continue
        try:
            chunk_num = int(name.split("_")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        path = os.path.join(chunks_dir, name)
        if _is_rewritten_chunk(path):
            completed.add(chunk_num)
    return completed


def _assemble_markdown(chunks_dir: str, output_md: str) -> None:
    chunk_files = sorted(
        [
            file_name
            for file_name in os.listdir(chunks_dir)
            if file_name.startswith("chunk_") and file_name.endswith(".md")
        ]
    )
    parts = []
    for file_name in chunk_files:
        path = os.path.join(chunks_dir, file_name)
        with open(path, "r", encoding="utf-8") as file:
            parts.append(_strip_chunk_header(file.read()))

    with open(output_md, "w", encoding="utf-8") as file:
        file.write("\n\n".join(parts))


def _split_for_rewrite(text: str, max_chars: int = 1800) -> list[str]:
    """
    Split long source text for safer rewrites.

    Primary mode preserves paragraph boundaries. If text is one huge paragraph
    (common in OCR bibliographies/indexes), fall back to line/sentence/hard splits.
    """

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
        step = max(400, max_chars)
        return [unit[idx : idx + step].strip() for idx in range(0, len(unit), step) if unit[idx : idx + step].strip()]

    def split_long_unit(unit: str) -> list[str]:
        if len(unit) <= max_chars:
            return [unit]

        lines = [line.strip() for line in unit.splitlines() if line.strip()]
        if len(lines) > 1:
            packed_lines = pack_units(lines, "\n")
            if packed_lines and max(len(item) for item in packed_lines) <= max_chars:
                return packed_lines

        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?;；:：])\s+", unit) if s.strip()]
        if len(sentences) > 1:
            packed_sentences = pack_units(sentences, " ")
            if packed_sentences and max(len(item) for item in packed_sentences) <= max_chars:
                return packed_sentences

        return hard_split(unit)

    raw = text.strip()
    if not raw:
        return [""]

    paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
    if not paragraphs:
        return hard_split(raw)

    normalized_units: list[str] = []
    for paragraph in paragraphs:
        normalized_units.extend(split_long_unit(paragraph))

    segments = pack_units(normalized_units, "\n\n")
    if segments:
        return segments

    return hard_split(raw)


def _trim_for_prompt(text: str, max_chars: int) -> str:
    """Trim optional context blocks to keep prompts bounded."""
    if not text:
        return ""
    return text[:max_chars]


def _high_risk_rewrite_mode(source_text: str, tags: list[str], score: int) -> bool:
    """High-risk chunks should prefer strict source-only rewrite to avoid hallucination."""
    # Omission alone is too broad/noisy in iterative loops; treat stronger failures as high-risk.
    risk_tags = {"HALLUCINATION", "MISTRANSLATION", "FORMAT", "HUMAN_ATTENTION"}
    if any(tag in risk_tags for tag in tags):
        return True
    if score <= 5:
        return True
    if len(source_text) > 2800:
        return True
    return False


def _rewrite_chunk_with_strategy(
    writer: WriterAgent,
    glossary_manager: GlossaryManager,
    rag_store: RAGStore | None,
    rag_k: int,
    source_text: str,
    current_translation: str,
    issues: list[str],
    tags: list[str],
    score: int,
    loop_index: int | None = None,
    stagnation_rounds: int = 0,
) -> str:
    """Rewrite chunk with adaptive strategy to reduce hallucination on long chunks."""
    high_risk = _high_risk_rewrite_mode(source_text, tags, score)
    glossary_text = _trim_for_prompt(glossary_manager.get_relevant_glossary_text(source_text), 1800)
    rag_context = ""
    if rag_store is not None and not high_risk:
        rag_context = _trim_for_prompt(rag_store.retrieve_context(source_text, k=rag_k), 3200)

    force_segmented = high_risk or len(source_text) > 3200
    if not force_segmented:
        return writer.rewrite(
            source_text=source_text,
            current_translation=current_translation,
            issues=issues,
            issue_tags=tags,
            audit_score=score,
            loop_index=loop_index,
            glossary_text=glossary_text,
            rag_context=rag_context,
            stagnation_rounds=stagnation_rounds,
        )

    segment_tags = list(dict.fromkeys(tags + ["FORCE_SOURCE_ONLY", "STRICT_LITERAL"]))
    segments = _split_for_rewrite(source_text, max_chars=900)
    outputs = []
    for segment in segments:
        seg_glossary = _trim_for_prompt(glossary_manager.get_relevant_glossary_text(segment), 900)
        seg_context = ""
        if rag_store is not None and not high_risk:
            seg_context = _trim_for_prompt(rag_store.retrieve_context(segment, k=max(1, rag_k - 1)), 1400)

        rewritten = writer.rewrite(
            source_text=segment,
            current_translation="[SEGMENT_MODE]",
            issues=issues,
            issue_tags=segment_tags,
            audit_score=min(score, 3),
            loop_index=loop_index,
            glossary_text=seg_glossary,
            rag_context=seg_context,
            stagnation_rounds=stagnation_rounds,
        )
        outputs.append(rewritten.strip())
    return "\n\n".join(outputs)


def _rewrite_with_retry(
    writer: WriterAgent,
    glossary_manager: GlossaryManager,
    rag_store: RAGStore | None,
    rag_k: int,
    source_text: str,
    current_translation: str,
    issues: list[str],
    tags: list[str],
    score: int,
    loop_index: int | None = None,
    stagnation_rounds: int = 0,
    max_attempts: int = 2,
    log_fn=None,
) -> tuple[str | None, str | None]:
    """Attempt rewrite with fallback strategy on timeout/errors.

    Both attempts keep chunk-level grounding with source + audit feedback.
    Retry path is stricter source-only and minimizes context contamination.
    """
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                rewritten = _rewrite_chunk_with_strategy(
                    writer=writer,
                    glossary_manager=glossary_manager,
                    rag_store=rag_store,
                    rag_k=rag_k,
                    source_text=source_text,
                    current_translation=current_translation,
                    issues=issues,
                    tags=tags,
                    score=score,
                    loop_index=loop_index,
                    stagnation_rounds=stagnation_rounds,
                )
                route = getattr(writer, "last_route", {}) or {}
                if route:
                    model_name = route.get("model", "unknown")
                    api_base = route.get("api_base", "unknown")
                    escalated = bool(route.get("escalated", False))
                    timeout = route.get("timeout", "n/a")
                    stagnation = route.get("stagnation_rounds", 0)
                    msg = (
                        f"[route attempt={attempt} model={model_name} base={api_base} "
                        f"escalated={escalated} timeout={timeout} stagnation={stagnation}]"
                    )
                    if callable(log_fn):
                        log_fn(msg)
                    else:
                        print(msg)
            else:
                # Fallback: force strict source-only rewrite with minimal context.
                fallback_tags = list(dict.fromkeys(tags + ["FORCE_SOURCE_ONLY", "STRICT_LITERAL"]))
                rewritten = _rewrite_chunk_with_strategy(
                    writer=writer,
                    glossary_manager=glossary_manager,
                    rag_store=rag_store,
                    rag_k=1,
                    source_text=source_text,
                    current_translation="[RETRY_MODE]",
                    issues=issues,
                    tags=fallback_tags,
                    score=min(score, 2),
                    loop_index=loop_index,
                    stagnation_rounds=stagnation_rounds,
                )
                route = getattr(writer, "last_route", {}) or {}
                if route:
                    model_name = route.get("model", "unknown")
                    api_base = route.get("api_base", "unknown")
                    escalated = bool(route.get("escalated", False))
                    timeout = route.get("timeout", "n/a")
                    stagnation = route.get("stagnation_rounds", 0)
                    msg = (
                        f"[route attempt={attempt} model={model_name} base={api_base} "
                        f"escalated={escalated} timeout={timeout} stagnation={stagnation}]"
                    )
                    if callable(log_fn):
                        log_fn(msg)
                    else:
                        print(msg)
            return rewritten, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < max_attempts:
                if callable(log_fn):
                    log_fn(f"retrying after error: {last_error}")
                else:
                    print(f"retrying after error: {last_error}")

    return None, last_error


def _compact_issues(issues: list[str], limit: int = 3, max_chars: int = 260) -> list[str]:
    """Keep audit feedback concise to avoid prompt overload/noise."""
    compact = []
    for issue in issues[:limit]:
        text = issue.strip()
        compact.append(text[:max_chars])
    return compact


def _load_flagged_from_report(report_path: str) -> tuple[list[int], dict[int, dict]]:
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    meta_by_chunk: dict[int, dict] = {}
    for item in report.get("results", []):
        if item.get("verdict") != "REWRITE":
            continue
        chunk_num = item.get("chunk_num")
        if isinstance(chunk_num, int):
            tags = []
            if item.get("hallucination"):
                tags.append("HALLUCINATION")
            if item.get("omission"):
                tags.append("OMISSION")
            if item.get("mistranslation"):
                tags.append("MISTRANSLATION")
            if item.get("format_ok") is False:
                tags.append("FORMAT")
            needs_human_attention = bool(item.get("needs_human_attention", False))
            if needs_human_attention:
                tags.append("HUMAN_ATTENTION")

            meta_by_chunk[chunk_num] = {
                "issues": _compact_issues(item.get("issues", [])),
                "tags": tags,
                "score": int(item.get("score", 0) or 0),
                "needs_human_attention": needs_human_attention,
                "human_attention_reason": item.get("human_attention_reason", ""),
            }

    chunk_numbers = sorted(meta_by_chunk.keys())
    return chunk_numbers, meta_by_chunk


def _audit_chunks(source_md: str, chunks_dir: str, output_audit_dir: str, source_chunks_dir: str) -> str:
    chunk_files = sorted(
        [
            file_name
            for file_name in os.listdir(chunks_dir)
            if file_name.startswith("chunk_") and file_name.endswith(".md")
        ]
    )
    chunk_numbers = [int(name.split("_")[1].split(".")[0]) for name in chunk_files]
    pairs, total_chunks = load_chunks(source_md, chunks_dir, chunk_numbers, source_chunks_dir=source_chunks_dir)

    llm = get_llm()
    config = load_config()
    audit_cfg = config.get("audit", {})
    use_rag = bool(audit_cfg.get("use_rag", True))
    audit_rag_k = int(audit_cfg.get("rag_k", 2))
    audit_rag_max_chars = int(audit_cfg.get("rag_max_chars", 1200))
    rag_store: RAGStore | None = None
    if use_rag:
        try:
            source_corpus = load_source_corpus(source_md, source_chunks_dir=source_chunks_dir)
            rag_store = RAGStore()
            rag_store.index_document(source_corpus)
            print(f"  Audit RAG: ON (k={audit_rag_k}, max_chars={audit_rag_max_chars})")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN Audit RAG init failed, fallback to non-RAG audit: {exc}")
            rag_store = None
    else:
        print("  Audit RAG: OFF")

    results = []

    try:
        for pair in pairs:
            chunk_num = pair["chunk_num"]
            start = time.time()
            print(f"  Auditing rewritten chunk {chunk_num}/{total_chunks}...", end=" ", flush=True)
            try:
                rag_context = ""
                if rag_store is not None:
                    rag_context = _trim_for_prompt(rag_store.retrieve_context(pair["source"], k=audit_rag_k), audit_rag_max_chars)
                audit = audit_chunk(llm, pair["source"], pair["translation"], rag_context=rag_context)
                elapsed = time.time() - start
                status = "PASS" if audit["verdict"] == "PASS" else "REWRITE"
                print(f"{status} (score: {audit['score']}/10, {elapsed:.0f}s)")
                audit["chunk_num"] = chunk_num
                audit["chunk_file"] = pair["chunk_file"]
                results.append(audit)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {exc}")
                results.append({"chunk_num": chunk_num, "error": str(exc), "verdict": "ERROR"})
    finally:
        if rag_store is not None:
            rag_store.clear()

    passed = sum(1 for item in results if item.get("verdict") == "PASS")
    flagged = sum(1 for item in results if item.get("verdict") == "REWRITE")
    errors = sum(1 for item in results if item.get("verdict") == "ERROR")
    avg_score = sum(item.get("score", 0) for item in results if "score" in item) / max(len(results), 1)

    print("\n" + "=" * 50)
    print("  REWRITE AUDIT SUMMARY")
    print(f"  Passed: {passed} | Flagged: {flagged} | Errors: {errors}")
    print(f"  Average score: {avg_score:.1f}/10")
    print("=" * 50 + "\n")

    os.makedirs(output_audit_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_audit_dir, f"audit_rewrite_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_audited": len(results),
                "passed": passed,
                "flagged": flagged,
                "avg_score": round(avg_score, 1),
                "results": [{k: v for k, v in item.items() if k != "raw"} for item in results],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Rewrite audit report saved: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite flagged chunks from latest audit report, then audit rewritten output."
    )
    parser.add_argument(
        "--source",
        "-s",
        default="",
        help="Path to source markdown used for chunking (e.g. data/output/ocr/MinerU_processed_combined.md)",
    )
    parser.add_argument(
        "--source-chunks-dir",
        default="data/output/source_chunks",
        help="Directory containing source chunk_XXX.md files (preferred)",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/output/chunks",
        help="Directory containing original chunk_XXX.md files",
    )
    parser.add_argument(
        "--audit-dir",
        default="data/output/audits",
        help="Directory containing prior audit JSON reports",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional explicit audit report JSON path (defaults to latest in --audit-dir)",
    )
    parser.add_argument(
        "--output-root",
        default="data/output/rewrites",
        help="Root directory for rewritten chunk outputs",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Existing rewrite run directory to resume (e.g. data/output/rewrites/rewrite_run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=1,
        help="Only rewrite chunks with chunk_num >= this value",
    )
    parser.add_argument(
        "--rewrite-human-attention",
        action="store_true",
        help="Rewrite chunks marked as needs_human_attention (default: skip and request manual review)",
    )
    args = parser.parse_args()

    report_path = _resolve_report_path(args.report, args.audit_dir)
    flagged_chunks, meta_by_chunk = _load_flagged_from_report(report_path)

    if not flagged_chunks:
        raise RuntimeError(f"No REWRITE chunks found in report: {report_path}")

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_root, f"rewrite_run_{run_id}")

    rewritten_chunks_dir = os.path.join(run_dir, "chunks")
    rewritten_audits_dir = os.path.join(run_dir, "audits")
    os.makedirs(rewritten_chunks_dir, exist_ok=True)

    # On fresh runs, seed run-dir with base chunks. On resume, preserve existing files.
    if not args.run_dir:
        chunk_files = [
            name
            for name in os.listdir(args.chunks_dir)
            if name.startswith("chunk_") and name.endswith(".md")
        ]
        for name in chunk_files:
            shutil.copy2(os.path.join(args.chunks_dir, name), os.path.join(rewritten_chunks_dir, name))

    print("\n" + "=" * 50)
    print("  REWRITE PHASE")
    source_label = args.source_chunks_dir if os.path.isdir(args.source_chunks_dir) else args.source
    print(f"  Source: {source_label}")
    print(f"  Base chunks: {args.chunks_dir}")
    print(f"  Audit report: {report_path}")
    print(f"  Rewritten output dir: {rewritten_chunks_dir}")
    print(f"  Flagged chunks: {len(flagged_chunks)}")
    print("=" * 50 + "\n")

    config = load_config()
    glossary_path = config.get("directories", {}).get("glossary", "data/glossary.json")
    rag_k = int(config.get("rag", {}).get("rewrite_k", 3))
    glossary_manager = GlossaryManager(glossary_path)

    rag_store: RAGStore | None = None
    try:
        source_text = load_source_corpus(args.source, source_chunks_dir=args.source_chunks_dir)
        rag_store = RAGStore()
        rag_store.index_document(source_text)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN RAG initialization failed, continuing without RAG context: {exc}")
        rag_store = None

    writer = WriterAgent()
    completed = _completed_rewritten_chunks(rewritten_chunks_dir)
    human_attention_chunks = [
        c for c in flagged_chunks if meta_by_chunk.get(c, {}).get("needs_human_attention", False)
    ]
    if human_attention_chunks and not args.rewrite_human_attention:
        preview = human_attention_chunks[:10]
        suffix = " ..." if len(human_attention_chunks) > len(preview) else ""
        print(f"  Human attention required for {len(human_attention_chunks)} chunks: {preview}{suffix}")
    target_chunks = [
        c
        for c in flagged_chunks
        if c >= args.start_chunk
        and c not in completed
        and (args.rewrite_human_attention or c not in set(human_attention_chunks))
    ]
    if completed:
        print(f"  Resume detected: {len(completed)} chunks already rewritten, skipping them.")
    if args.start_chunk > 1:
        print(f"  Start chunk filter active: chunk >= {args.start_chunk}")
    if not target_chunks:
        print("  No pending chunks to rewrite; proceeding to audit phase.")
    rewrite_pairs, _ = load_chunks(
        args.source,
        args.chunks_dir,
        target_chunks,
        source_chunks_dir=args.source_chunks_dir,
    )

    failed_chunks = []
    for pair in rewrite_pairs:
        chunk_num = pair["chunk_num"]
        meta = meta_by_chunk.get(chunk_num, {})
        issues = meta.get("issues", [])
        tags = meta.get("tags", [])
        score = meta.get("score", 0)
        print(f"  Rewriting chunk {chunk_num}...", end=" ", flush=True)
        start = time.time()

        rewritten, error = _rewrite_with_retry(
            writer=writer,
            glossary_manager=glossary_manager,
            rag_store=rag_store,
            rag_k=rag_k,
            source_text=pair["source"],
            current_translation=pair["translation"],
            issues=issues,
            tags=tags,
            score=score,
            loop_index=1,
            max_attempts=2,
        )
        elapsed = time.time() - start

        if rewritten is None:
            print(f"FAILED ({elapsed:.0f}s): {error}")
            failed_chunks.append({"chunk_num": chunk_num, "error": error})
            continue

        output_path = os.path.join(rewritten_chunks_dir, f"chunk_{chunk_num:03d}.md")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(
                f"<!-- Chunk {chunk_num} | REWRITTEN from {os.path.basename(report_path)} | {elapsed:.0f}s -->\n\n"
            )
            file.write(rewritten)
        print(f"done ({elapsed:.0f}s)")

    combined_output = os.path.join(run_dir, "rewritten_translated.md")
    _assemble_markdown(rewritten_chunks_dir, combined_output)
    print(f"\n  Rewritten combined markdown: {combined_output}")
    if failed_chunks:
        failed_path = os.path.join(run_dir, "rewrite_failures.json")
        with open(failed_path, "w", encoding="utf-8") as file:
            json.dump(failed_chunks, file, ensure_ascii=False, indent=2)
        print(f"  WARN rewrite failures saved: {failed_path}")

    print("\n" + "=" * 50)
    print("  AUDIT PHASE (REWRITTEN OUTPUT)")
    print("=" * 50 + "\n")
    new_report = _audit_chunks(
        args.source,
        rewritten_chunks_dir,
        rewritten_audits_dir,
        source_chunks_dir=args.source_chunks_dir,
    )

    if rag_store is not None:
        rag_store.clear()

    print("\n" + "=" * 50)
    print("  REWRITE + AUDIT COMPLETE")
    print(f"  Rewritten chunks: {rewritten_chunks_dir}")
    print(f"  Rewritten markdown: {combined_output}")
    print(f"  New audit report: {new_report}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
