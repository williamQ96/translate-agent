"""
rewrite_audit_loop.py - Iterative rewrite/audit loop with score-based chunk locking.

Workflow:
1. Rewrite unlocked chunks using latest audit feedback.
2. Audit rewritten chunks.
3. Lock chunks with score >= target_score (default 9) so they are not rewritten again.
4. Repeat until all chunks are locked or max_loops is reached.
"""

import argparse
import filecmp
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
from src.rewrite_audit import _latest_audit_report, _rewrite_with_retry
from src.utils.config_loader import load_config


def _print_safe(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode())


def _list_chunk_numbers(chunks_dir: str) -> list[int]:
    nums = []
    for name in os.listdir(chunks_dir):
        if name.startswith("chunk_") and name.endswith(".md"):
            try:
                nums.append(int(name.split("_")[1].split(".")[0]))
            except (ValueError, IndexError):
                continue
    return sorted(nums)


def _effective_chunk_path(base_chunks_dir: str, rewrite_chunks_dir: str, chunk_num: int) -> str:
    rewrite_path = os.path.join(rewrite_chunks_dir, f"chunk_{chunk_num:03d}.md")
    if os.path.isfile(rewrite_path):
        return rewrite_path
    return os.path.join(base_chunks_dir, f"chunk_{chunk_num:03d}.md")


def _strip_chunk_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("<!--"):
        return "\n".join(lines[1:]).lstrip("\n")
    return text.strip()


def _read_translation_chunk(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return _strip_chunk_header(file.read())


def _chunk_mtime_bounds(base_chunks_dir: str, rewrite_chunks_dir: str, chunk_numbers: list[int]) -> tuple[float, float]:
    mtimes: list[float] = []
    for num in chunk_numbers:
        path = _effective_chunk_path(base_chunks_dir, rewrite_chunks_dir, num)
        if os.path.isfile(path):
            mtimes.append(os.path.getmtime(path))
    if not mtimes:
        return 0.0, 0.0
    return min(mtimes), max(mtimes)


def _load_effective_pairs(
    source_md: str,
    base_chunks_dir: str,
    rewrite_chunks_dir: str,
    chunk_numbers: list[int],
    source_chunks_dir: str,
):
    """Load chunk pairs where translation comes from rewrite overlay when available."""
    pairs, total_chunks = load_chunks(
        source_md,
        base_chunks_dir,
        chunk_numbers,
        source_chunks_dir=source_chunks_dir,
    )
    for pair in pairs:
        chunk_num = pair["chunk_num"]
        effective_path = _effective_chunk_path(base_chunks_dir, rewrite_chunks_dir, chunk_num)
        if effective_path != pair["chunk_file"]:
            pair["translation"] = _read_translation_chunk(effective_path)
            pair["chunk_file"] = effective_path
    return pairs, total_chunks


def _assemble_effective_markdown(
    base_chunks_dir: str,
    rewrite_chunks_dir: str,
    chunk_numbers: list[int],
    output_md: str,
) -> None:
    parts: list[str] = []
    for chunk_num in chunk_numbers:
        path = _effective_chunk_path(base_chunks_dir, rewrite_chunks_dir, chunk_num)
        if not os.path.isfile(path):
            continue
        parts.append(_read_translation_chunk(path))
    with open(output_md, "w", encoding="utf-8") as file:
        file.write("\n\n".join(parts))


def _prune_identical_overlay_chunks(base_chunks_dir: str, rewrite_chunks_dir: str) -> int:
    """Remove overlay chunk files that are byte-identical to base chunks."""
    removed = 0
    if not os.path.isdir(rewrite_chunks_dir):
        return removed
    for name in os.listdir(rewrite_chunks_dir):
        if not (name.startswith("chunk_") and name.endswith(".md")):
            continue
        rewrite_path = os.path.join(rewrite_chunks_dir, name)
        base_path = os.path.join(base_chunks_dir, name)
        if not os.path.isfile(base_path):
            continue
        try:
            if filecmp.cmp(rewrite_path, base_path, shallow=False):
                os.remove(rewrite_path)
                removed += 1
        except OSError:
            continue
    return removed


def _loop_artifact_dir(run_dir: str, loop_id: int) -> str:
    return os.path.join(run_dir, f"loop{loop_id}")


def _copy_report_to_loop_dir(report_path: str, loop_dir: str) -> str:
    os.makedirs(loop_dir, exist_ok=True)
    target = os.path.join(loop_dir, os.path.basename(report_path))
    shutil.copy2(report_path, target)
    return target


def _compact_issues(issues: list[str], limit: int = 3, max_chars: int = 260) -> list[str]:
    compact = []
    for issue in issues[:limit]:
        compact.append(issue.strip()[:max_chars])
    return compact


def _tags_from_result(item: dict) -> list[str]:
    tags = []
    if item.get("hallucination"):
        tags.append("HALLUCINATION")
    if item.get("omission"):
        tags.append("OMISSION")
    if item.get("mistranslation"):
        tags.append("MISTRANSLATION")
    if item.get("format_ok") is False:
        tags.append("FORMAT")
    if item.get("needs_human_attention"):
        tags.append("HUMAN_ATTENTION")
    return tags


def _load_report_meta(report_path: str) -> dict[int, dict]:
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    meta: dict[int, dict] = {}
    for item in report.get("results", []):
        chunk_num = item.get("chunk_num")
        if not isinstance(chunk_num, int):
            continue
        meta[chunk_num] = {
            "score": int(item.get("score", 0) or 0),
            "issues": _compact_issues(item.get("issues", [])),
            "tags": _tags_from_result(item),
            "verdict": item.get("verdict", "REWRITE"),
            "error": item.get("error", ""),
            "needs_human_attention": bool(item.get("needs_human_attention", False)),
            "human_attention_reason": item.get("human_attention_reason", ""),
        }
    return meta


def _is_lockable_meta(meta: dict, target_score: int) -> bool:
    """Lock only if score target is met and no critical issue tags remain."""
    score = int(meta.get("score", 0) or 0)
    tags = meta.get("tags", []) or []
    error = meta.get("error", "")
    needs_human_attention = bool(meta.get("needs_human_attention", False))
    return score >= target_score and not tags and not error and not needs_human_attention


def _is_effective_pass(audit: dict, target_score: int) -> bool:
    """Derive pass/fail from score and critical flags instead of trusting model verdict text."""
    if audit.get("needs_human_attention"):
        return False
    score = int(audit.get("score", 0) or 0)
    has_critical_flag = any(
        [
            bool(audit.get("hallucination")),
            bool(audit.get("omission")),
            bool(audit.get("mistranslation")),
            audit.get("format_ok") is False,
        ]
    )
    return score >= target_score and not has_critical_flag


def _critical_tag_count(item: dict) -> int:
    count = 0
    if item.get("hallucination"):
        count += 1
    if item.get("omission"):
        count += 1
    if item.get("mistranslation"):
        count += 1
    if item.get("format_ok") is False:
        count += 1
    if item.get("needs_human_attention"):
        count += 1
    return count


def _accept_rewrite_candidate(
    prev_score: int,
    prev_tag_count: int,
    candidate_audit: dict,
    min_score_delta: int,
) -> tuple[bool, str]:
    """
    Keep score trend stable:
    - Accept if candidate score >= prev_score + min_score_delta
    - If score ties, accept only when critical-tag count does not worsen.
    """
    cand_score = int(candidate_audit.get("score", 0) or 0)
    cand_tag_count = _critical_tag_count(candidate_audit)

    if cand_score >= prev_score + min_score_delta:
        return True, f"accepted(score {prev_score}->{cand_score})"

    if cand_score == prev_score and cand_tag_count <= prev_tag_count:
        return True, f"accepted(tie score {cand_score}, tags {prev_tag_count}->{cand_tag_count})"

    return False, f"rejected(score {prev_score}->{cand_score}, tags {prev_tag_count}->{cand_tag_count})"


def _audit_selected_chunks(
    source_md: str,
    base_chunks_dir: str,
    rewrite_chunks_dir: str,
    chunk_numbers: list[int],
    llm,
    output_audit_dir: str,
    loop_id: int,
    target_score: int,
    source_chunks_dir: str,
    rag_store: RAGStore | None = None,
    audit_rag_k: int = 2,
    audit_rag_max_chars: int = 1200,
) -> tuple[str, dict[int, dict]]:
    pairs, total_chunks = _load_effective_pairs(
        source_md=source_md,
        base_chunks_dir=base_chunks_dir,
        rewrite_chunks_dir=rewrite_chunks_dir,
        chunk_numbers=chunk_numbers,
        source_chunks_dir=source_chunks_dir,
    )
    results = []

    for pair in pairs:
        chunk_num = pair["chunk_num"]
        start = time.time()
        print(f"  Auditing chunk {chunk_num}/{total_chunks}...", end=" ", flush=True)
        try:
            rag_context = ""
            if rag_store is not None:
                rag_context = pair["source"]
                rag_context = rag_store.retrieve_context(rag_context, k=audit_rag_k)[:audit_rag_max_chars]
            audit = audit_chunk(llm, pair["source"], pair["translation"], rag_context=rag_context)
            elapsed = time.time() - start
            status = "PASS" if _is_effective_pass(audit, target_score) else "REWRITE"
            audit["model_verdict"] = audit.get("verdict", "UNKNOWN")
            audit["verdict"] = status
            if status == "REWRITE" and not audit.get("issues"):
                score = int(audit.get("score", 0) or 0)
                if score < target_score:
                    audit["issues"] = [f"SCORE_BELOW_TARGET: {score}/{target_score}"]
            _print_safe(f"{status} (score: {audit['score']}/10, {elapsed:.0f}s)")
            audit["chunk_num"] = chunk_num
            audit["chunk_file"] = pair["chunk_file"]
            results.append(audit)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            results.append({"chunk_num": chunk_num, "error": str(exc), "verdict": "ERROR", "score": 0, "issues": []})

    passed = sum(1 for item in results if item.get("verdict") == "PASS")
    flagged = sum(1 for item in results if item.get("verdict") == "REWRITE")
    errors = sum(1 for item in results if item.get("verdict") == "ERROR")
    human_attention = sum(1 for item in results if item.get("needs_human_attention"))
    avg_score = sum(item.get("score", 0) for item in results if "score" in item) / max(len(results), 1)

    os.makedirs(output_audit_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_audit_dir, f"audit_loop_{loop_id:02d}_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "loop": loop_id,
                "total_audited": len(results),
                "passed": passed,
                "flagged": flagged,
                "errors": errors,
                "needs_human_attention": human_attention,
                "avg_score": round(avg_score, 1),
                "results": [{k: v for k, v in item.items() if k != "raw"} for item in results],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 50)
    print(f"  LOOP {loop_id} AUDIT SUMMARY")
    print(f"  Passed: {passed} | Flagged: {flagged} | Errors: {errors}")
    print(f"  Needs human attention: {human_attention}")
    print(f"  Average score: {avg_score:.1f}/10")
    print(f"  Report: {report_path}")
    print("=" * 50 + "\n")

    return report_path, _load_report_meta(report_path)


def _save_state(state_path: str, state: dict) -> None:
    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def _assess_seed_report(
    report_path: str,
    report_meta: dict[int, dict],
    base_chunks_dir: str,
    rewrite_chunks_dir: str,
    chunk_numbers: list[int],
    stale_grace_seconds: int = 5,
) -> tuple[bool, list[str]]:
    """
    Validate whether the seed audit report matches current working chunks.

    A report is considered stale/incompatible when:
    - it does not cover all current chunk numbers, or
    - it is older than the latest chunk file (with a small grace window), or
    - the report file does not exist.
    """
    reasons: list[str] = []
    all_set = set(chunk_numbers)
    report_set = set(report_meta.keys())

    missing = sorted(all_set - report_set)
    if missing:
        reasons.append(f"missing_chunk_scores={len(missing)}")

    if not os.path.exists(report_path):
        reasons.append("report_file_missing")
    else:
        _, latest_chunk_mtime = _chunk_mtime_bounds(base_chunks_dir, rewrite_chunks_dir, chunk_numbers)
        report_mtime = os.path.getmtime(report_path)
        if latest_chunk_mtime > 0 and (report_mtime + stale_grace_seconds) < latest_chunk_mtime:
            reasons.append("report_older_than_chunks")

    if not report_meta:
        reasons.append("report_has_no_results")

    return (len(reasons) > 0), reasons


def _resolve_report_path(report: str, audit_dir: str) -> str:
    if not report:
        return _latest_audit_report(audit_dir)

    if os.path.exists(report):
        return report

    placeholder_tokens = ("YYYY", "MM", "DD", "HHMMSS")
    if any(token in report for token in placeholder_tokens):
        fallback = _latest_audit_report(audit_dir)
        _print_safe(f"WARN: Placeholder report path detected, using latest audit report: {fallback}")
        return fallback

    raise FileNotFoundError(
        f"Report file not found: {report}. "
        "Pass a real JSON path, or omit --report to auto-use latest in --audit-dir."
    )


def _safe_latest_audit_report(audit_dir: str) -> str:
    try:
        return _latest_audit_report(audit_dir)
    except FileNotFoundError:
        return ""


def _bootstrap_report_path(explicit_report: str, audit_dir: str, run_dir: str) -> str:
    """
    Resolve initial audit report path without hard-failing when none exists.

    Priority:
    1) explicit --report
    2) report referenced by run-dir state (current_report / initial_report)
    3) latest report in run-dir/audits
    4) latest report in --audit-dir
    """
    if explicit_report:
        return _resolve_report_path(explicit_report, audit_dir)

    state_path = os.path.join(run_dir, "loop_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as file:
                state = json.load(file)
            for key in ("current_report", "initial_report"):
                candidate = state.get(key, "")
                if candidate and os.path.exists(candidate):
                    return candidate
                if candidate:
                    by_name = os.path.join(run_dir, "audits", os.path.basename(candidate))
                    if os.path.exists(by_name):
                        return by_name
        except Exception:  # noqa: BLE001
            pass

    run_audits_dir = os.path.join(run_dir, "audits")
    if os.path.isdir(run_audits_dir):
        candidate = _safe_latest_audit_report(run_audits_dir)
        if candidate:
            return candidate

    return _safe_latest_audit_report(audit_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative rewrite->audit loop with score-based chunk locking.")
    parser.add_argument("--source", "-s", default="", help="Source markdown path used for chunking (fallback)")
    parser.add_argument(
        "--source-chunks-dir",
        default="data/output/source_chunks",
        help="Directory containing source chunk_XXX.md files (preferred)",
    )
    parser.add_argument("--chunks-dir", default="data/output/chunks", help="Initial chunk directory")
    parser.add_argument("--audit-dir", default="data/output/audits", help="Directory containing audit reports")
    parser.add_argument("--report", default="", help="Initial audit report JSON (default: latest in --audit-dir)")
    parser.add_argument("--output-root", default="data/output/rewrites", help="Root output directory")
    parser.add_argument("--run-dir", default="", help="Existing loop run directory to resume")
    parser.add_argument("--target-score", type=int, default=9, help="Lock threshold (default: 9)")
    parser.add_argument("--max-loops", type=int, default=30, help="Maximum loop iterations (default: 30)")
    parser.add_argument(
        "--no-guarded-acceptance",
        action="store_true",
        help="Disable per-chunk acceptance guard (default: guard ON to avoid score regression)",
    )
    parser.add_argument(
        "--acceptance-min-delta",
        type=int,
        default=0,
        help="Minimum score delta required for automatic accept (default: 0)",
    )
    parser.add_argument(
        "--rewrite-human-attention",
        action="store_true",
        help="Rewrite chunks marked as needs_human_attention (default: skip and request manual review)",
    )
    parser.add_argument(
        "--no-auto-seed-audit",
        action="store_true",
        help="Disable automatic seed-audit refresh when report is stale/incompatible",
    )
    parser.add_argument(
        "--max-same-signal-retries",
        type=int,
        default=2,
        help=(
            "Max consecutive rejected rewrite attempts for a chunk before temporary rewrite backoff. "
            "Chunk still stays in audit until it locks or needs manual intervention."
        ),
    )
    args = parser.parse_args()

    if args.target_score < 1 or args.target_score > 10:
        raise ValueError("--target-score must be in [1, 10]")
    if args.acceptance_min_delta < 0:
        raise ValueError("--acceptance-min-delta must be >= 0")

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_root, f"rewrite_loop_run_{run_id}")

    report_path = _bootstrap_report_path(args.report, args.audit_dir, run_dir)
    if report_path:
        initial_meta = _load_report_meta(report_path)
    else:
        initial_meta = {}
        _print_safe("WARN: No active audit report found. A seed audit will run before rewrite.")
    current_report_path = report_path

    chunks_work_dir = os.path.join(run_dir, "chunks")
    audits_dir = os.path.join(run_dir, "audits")
    logging_cfg = load_config().get("logging", {})
    master_log_path = str(logging_cfg.get("master_log_path", "") or "").strip()
    loop_log_root = str(logging_cfg.get("loop_log_dir", "logs") or "logs").strip()
    loop_log_dir = loop_log_root if os.path.isabs(loop_log_root) else os.path.join(run_dir, loop_log_root)
    os.makedirs(loop_log_dir, exist_ok=True)
    loop_log_path = os.path.join(loop_log_dir, "rewrite_loop.log")
    _set_log_files(master_log_path=master_log_path, loop_log_path=loop_log_path)
    state_path = os.path.join(run_dir, "loop_state.json")
    os.makedirs(chunks_work_dir, exist_ok=True)
    os.makedirs(audits_dir, exist_ok=True)

    removed_overlay = _prune_identical_overlay_chunks(args.chunks_dir, chunks_work_dir)
    if removed_overlay:
        _print_safe(f"Pruned {removed_overlay} redundant overlay chunks identical to base.")

    all_chunks = _list_chunk_numbers(args.chunks_dir)
    if not all_chunks:
        raise RuntimeError(f"No chunk_XXX.md files found in {args.chunks_dir}")

    all_chunk_set = set(all_chunks)

    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as file:
            state = json.load(file)
        locked_chunks = set(int(x) for x in state.get("locked_chunks", []))
        latest_meta = {int(k): v for k, v in state.get("latest_meta", {}).items()}
        stale_locked = sorted(num for num in locked_chunks if num not in all_chunk_set)
        if stale_locked:
            locked_chunks = {num for num in locked_chunks if num in all_chunk_set}
            _print_safe(f"WARN: Dropped {len(stale_locked)} stale locked chunks not in current base chunk set.")
        loop_start = int(state.get("next_loop", 1))
        current_report_path = state.get("current_report", report_path) or report_path
        if not os.path.exists(current_report_path):
            _print_safe(
                f"WARN: Saved current report missing ({current_report_path}); "
                "will validate metadata and auto-refresh seed audit if needed."
            )
        _print_safe(f"Resuming loop run from {run_dir}, next loop: {loop_start}")
    else:
        latest_meta = initial_meta
        locked_chunks = {
            num for num, meta in latest_meta.items() if (num in all_chunk_set and _is_lockable_meta(meta, args.target_score))
        }
        loop_start = 1
        state = {
            "run_dir": run_dir,
            "source": args.source,
            "target_score": args.target_score,
            "max_loops": args.max_loops,
            "initial_report": report_path,
            "history": [],
            "next_loop": loop_start,
            "current_report": current_report_path,
            "locked_chunks": sorted(locked_chunks),
            "latest_meta": {str(k): v for k, v in latest_meta.items()},
        }
        _save_state(state_path, state)

    _print_safe("\n" + "=" * 50)
    _print_safe("  ITERATIVE REWRITE LOOP")
    _print_safe(f"  Run dir: {run_dir}")
    source_label = args.source_chunks_dir if os.path.isdir(args.source_chunks_dir) else args.source
    _print_safe(f"  Source: {source_label}")
    _print_safe(f"  Base chunks: {args.chunks_dir}")
    _print_safe(f"  Rewrite overlay: {chunks_work_dir}")
    _print_safe(f"  Target score: {args.target_score}")
    _print_safe(f"  Max loops: {args.max_loops}")
    _print_safe(f"  Total chunks: {len(all_chunks)}")
    _print_safe(f"  Initially locked: {len(locked_chunks)}")
    _print_safe("=" * 50 + "\n")

    config = load_config()
    glossary_path = config.get("directories", {}).get("glossary", "data/glossary.json")
    rag_k = int(config.get("rag", {}).get("rewrite_k", 3))
    audit_cfg = config.get("audit", {})
    audit_use_rag = bool(audit_cfg.get("use_rag", True))
    audit_rag_k = int(audit_cfg.get("rag_k", 2))
    audit_rag_max_chars = int(audit_cfg.get("rag_max_chars", 1200))
    glossary_manager = GlossaryManager(glossary_path)
    writer = WriterAgent()
    audit_llm = get_llm()

    source_text = load_source_corpus(args.source, source_chunks_dir=args.source_chunks_dir)

    rag_store: RAGStore | None = None
    try:
        rag_store = RAGStore()
        rag_store.index_document(source_text)
    except Exception as exc:  # noqa: BLE001
        _print_safe(f"WARN: RAG unavailable for loop run: {exc}")
        rag_store = None

    # Guard against stale audit seeds after chunk regeneration or directory cleanup.
    # This avoids rewriting with outdated feedback from a previous chunk set.
    history = state.get("history", [])
    needs_seed_refresh, seed_reasons = _assess_seed_report(
        report_path=current_report_path,
        report_meta=latest_meta,
        base_chunks_dir=args.chunks_dir,
        rewrite_chunks_dir=chunks_work_dir,
        chunk_numbers=all_chunks,
    )
    critical_seed_reasons = {"report_file_missing", "report_has_no_results"}
    has_critical_seed_gap = any(reason in critical_seed_reasons for reason in seed_reasons)
    should_seed_refresh = (
        (not args.no_auto_seed_audit)
        and needs_seed_refresh
        and ((loop_start <= 1 and not history) or has_critical_seed_gap)
    )
    if should_seed_refresh:
        _print_safe(
            "Seed audit is stale/incompatible for current chunks "
            f"({', '.join(seed_reasons)}). Generating fresh seed audit..."
        )
        seed_report, seed_meta = _audit_selected_chunks(
            source_md=args.source,
            base_chunks_dir=args.chunks_dir,
            rewrite_chunks_dir=chunks_work_dir,
            chunk_numbers=all_chunks,
            llm=audit_llm,
            output_audit_dir=audits_dir,
            loop_id=0,
            target_score=args.target_score,
            source_chunks_dir=args.source_chunks_dir,
            rag_store=rag_store if audit_use_rag else None,
            audit_rag_k=audit_rag_k,
            audit_rag_max_chars=audit_rag_max_chars,
        )
        seed_loop_dir = _loop_artifact_dir(run_dir, 0)
        _copy_report_to_loop_dir(seed_report, seed_loop_dir)
        latest_meta = seed_meta
        locked_chunks = {num for num, meta in latest_meta.items() if _is_lockable_meta(meta, args.target_score)}
        current_report_path = seed_report
        state["initial_report"] = seed_report
        state["current_report"] = seed_report
        state["locked_chunks"] = sorted(locked_chunks)
        state["latest_meta"] = {str(k): v for k, v in latest_meta.items()}
        _save_state(state_path, state)
    elif needs_seed_refresh:
        _print_safe(
            "WARN: Seed audit appears stale/incompatible "
            f"({', '.join(seed_reasons)}), but auto-refresh was skipped."
        )

    try:
        for loop_id in range(loop_start, args.max_loops + 1):
            active_chunks = [num for num in all_chunks if num not in locked_chunks]
            if not active_chunks:
                _print_safe("All chunks have reached target score. Stopping loop.")
                break

            loop_dir = _loop_artifact_dir(run_dir, loop_id)
            loop_chunks_dir = os.path.join(loop_dir, "chunks")
            loop_rejected_dir = os.path.join(loop_dir, "rejected")
            os.makedirs(loop_chunks_dir, exist_ok=True)
            os.makedirs(loop_rejected_dir, exist_ok=True)

            human_attention_chunks = [
                num for num in active_chunks if latest_meta.get(num, {}).get("needs_human_attention", False)
            ]
            if human_attention_chunks and not args.rewrite_human_attention:
                preview = human_attention_chunks[:10]
                suffix = " ..." if len(human_attention_chunks) > len(preview) else ""
                _print_safe(
                    f"Human attention required for {len(human_attention_chunks)} chunks: {preview}{suffix}"
                )

            _print_safe(f"\n--- LOOP {loop_id} REWRITE PHASE ---")
            rewriteable_chunks = (
                active_chunks
                if args.rewrite_human_attention
                else [num for num in active_chunks if num not in set(human_attention_chunks)]
            )
            rejection_streaks = state.setdefault("rejection_streaks", {})
            throttled_rejections: list[int] = []
            filtered_chunks: list[int] = []
            for num in rewriteable_chunks:
                reject_count = int(rejection_streaks.get(str(num), 0) or 0)
                if reject_count >= int(args.max_same_signal_retries):
                    throttled_rejections.append(num)
                    continue
                filtered_chunks.append(num)
            rewriteable_chunks = filtered_chunks

            if throttled_rejections:
                preview = sorted(throttled_rejections)[:10]
                suffix = " ..." if len(throttled_rejections) > 10 else ""
                _print_safe(
                    "Rewrite backoff (consecutive rejected rewrites) for "
                    f"{len(throttled_rejections)} chunks: {preview}{suffix}"
                )

            _print_safe(f"Active chunks to rewrite: {len(rewriteable_chunks)}")

            if not rewriteable_chunks:
                _print_safe("No auto-rewrite candidates remain. Stopping for manual intervention.")
                break

            rewrite_pairs, _ = _load_effective_pairs(
                source_md=args.source,
                base_chunks_dir=args.chunks_dir,
                rewrite_chunks_dir=chunks_work_dir,
                chunk_numbers=rewriteable_chunks,
                source_chunks_dir=args.source_chunks_dir,
            )
            failed = []
            rejected = []
            for pair in rewrite_pairs:
                chunk_num = pair["chunk_num"]
                meta = latest_meta.get(chunk_num, {})
                issues = meta.get("issues", [])
                tags = meta.get("tags", [])
                score = int(meta.get("score", 0) or 0)
                prev_tag_count = len([t for t in tags if t in {"HALLUCINATION", "OMISSION", "MISTRANSLATION", "FORMAT", "HUMAN_ATTENTION"}])
                audit_input = os.path.basename(current_report_path)

                print(
                    f"  Rewriting chunk {chunk_num}... (audit={audit_input}, prev_score={score})",
                    end=" ",
                    flush=True,
                )
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
                    max_attempts=2,
                )
                elapsed = time.time() - start

                if rewritten is None:
                    print(f"FAILED ({elapsed:.0f}s): {error}")
                    failed.append({"chunk_num": chunk_num, "error": error})
                    continue

                if not args.no_guarded_acceptance:
                    candidate_rag_context = ""
                    if rag_store is not None and audit_use_rag:
                        candidate_rag_context = pair["source"]
                        candidate_rag_context = rag_store.retrieve_context(candidate_rag_context, k=audit_rag_k)[
                            :audit_rag_max_chars
                        ]
                    candidate_audit = audit_chunk(
                        audit_llm,
                        pair["source"],
                        rewritten,
                        rag_context=candidate_rag_context,
                    )
                    accepted, decision = _accept_rewrite_candidate(
                        prev_score=score,
                        prev_tag_count=prev_tag_count,
                        candidate_audit=candidate_audit,
                        min_score_delta=args.acceptance_min_delta,
                    )
                    if not accepted:
                        rejected_path = os.path.join(loop_rejected_dir, f"chunk_{chunk_num:03d}.md")
                        with open(rejected_path, "w", encoding="utf-8") as file:
                            file.write(
                                f"<!-- Chunk {chunk_num} | LOOP {loop_id} | REJECTED | {decision} | "
                                f"INPUT {audit_input} -->\n\n"
                            )
                            file.write(rewritten)
                        print(f"skip ({decision})")
                        rejected.append(
                            {
                                "chunk_num": chunk_num,
                                "decision": decision,
                                "candidate_score": int(candidate_audit.get("score", 0) or 0),
                            }
                        )
                        continue

                out_path = os.path.join(chunks_work_dir, f"chunk_{chunk_num:03d}.md")
                with open(out_path, "w", encoding="utf-8") as file:
                    file.write(
                        f"<!-- Chunk {chunk_num} | LOOP {loop_id} | INPUT {audit_input} | "
                        f"PREV_SCORE {score} | {elapsed:.0f}s -->\n\n"
                    )
                    file.write(rewritten)
                loop_chunk_path = os.path.join(loop_chunks_dir, f"chunk_{chunk_num:03d}.md")
                shutil.copy2(out_path, loop_chunk_path)
                print(f"done ({elapsed:.0f}s)")

            _print_safe(f"\n--- LOOP {loop_id} AUDIT PHASE ---")
            active_after_rewrite = [num for num in all_chunks if num not in locked_chunks]
            new_report, loop_meta = _audit_selected_chunks(
                source_md=args.source,
                base_chunks_dir=args.chunks_dir,
                rewrite_chunks_dir=chunks_work_dir,
                chunk_numbers=active_after_rewrite,
                llm=audit_llm,
                output_audit_dir=audits_dir,
                loop_id=loop_id,
                target_score=args.target_score,
                source_chunks_dir=args.source_chunks_dir,
                rag_store=rag_store if audit_use_rag else None,
                audit_rag_k=audit_rag_k,
                audit_rag_max_chars=audit_rag_max_chars,
            )
            loop_report_copy = _copy_report_to_loop_dir(new_report, loop_dir)

            latest_meta.update(loop_meta)
            current_report_path = new_report
            newly_locked = {
                num
                for num in active_after_rewrite
                if _is_lockable_meta(latest_meta.get(num, {}), args.target_score)
            }
            before = len(locked_chunks)
            locked_chunks.update(newly_locked)
            gained = len(locked_chunks) - before

            summary = {
                "loop": loop_id,
                "report": new_report,
                "loop_dir": loop_dir,
                "loop_report": loop_report_copy,
                "loop_chunks_dir": loop_chunks_dir,
                "loop_rejected_dir": loop_rejected_dir,
                "active_count": len(active_after_rewrite),
                "locked_after_loop": len(locked_chunks),
                "newly_locked": sorted(newly_locked),
                "human_attention_chunks": sorted(
                    [num for num in active_after_rewrite if latest_meta.get(num, {}).get("needs_human_attention", False)]
                ),
                "failed_rewrites": failed,
                "rejected_rewrites": rejected,
            }
            state.setdefault("history", []).append(summary)
            state["next_loop"] = loop_id + 1
            state["current_report"] = current_report_path
            state["locked_chunks"] = sorted(locked_chunks)
            state["latest_meta"] = {str(k): v for k, v in latest_meta.items()}
            _save_state(state_path, state)

            rejection_streaks = state.setdefault("rejection_streaks", {})
            rejected_set = {int(item.get("chunk_num")) for item in rejected if item.get("chunk_num") is not None}
            failed_set = {
                num
                for num in (_as_chunk_num(item) for item in failed)
                if num is not None
            }
            accepted_set = set(rewriteable_chunks) - rejected_set - failed_set
            for num in rejected_set:
                key = str(num)
                rejection_streaks[key] = int(rejection_streaks.get(key, 0) or 0) + 1
            for num in accepted_set:
                rejection_streaks.pop(str(num), None)

            _print_safe(
                f"Loop {loop_id} progress: locked {len(locked_chunks)}/{len(all_chunks)} "
                f"(+{gained} this loop, rejected rewrites: {len(rejected)})"
            )

        final_output = os.path.join(run_dir, "rewritten_translated.md")
        _assemble_effective_markdown(args.chunks_dir, chunks_work_dir, all_chunks, final_output)
        _print_safe("\n" + "=" * 50)
        _print_safe("  LOOP COMPLETE")
        _print_safe(f"  Locked chunks: {len(locked_chunks)}/{len(all_chunks)}")
        _print_safe(f"  Final markdown: {final_output}")
        _print_safe(f"  Loop state: {state_path}")
        _print_safe("=" * 50 + "\n")
    finally:
        if rag_store is not None:
            rag_store.clear()
        _set_log_files("", "")


_ORIGINAL_PRINT_SAFE = _print_safe
_LAST_PRINT_LINE = {"value": ""}
_SEEN_LOOP_STATUS_LINES = set()
_LOG_FILE_HANDLES = {"master": None, "loop": None}
_LOOP_TIMING = {}
_LOOP_ELAPSED_PRINTED = set()


def _as_chunk_num(value):
    """Best-effort chunk number extraction from int/str/dict payloads."""
    if isinstance(value, dict):
        for key in ("chunk_num", "chunk", "id"):
            if key in value and value.get(key) is not None:
                try:
                    return int(value.get(key))
                except Exception:
                    pass
        return None
    try:
        return int(value)
    except Exception:
        return None


def _set_log_files(master_log_path: str = "", loop_log_path: str = "") -> None:
    for key in ("master", "loop"):
        handle = _LOG_FILE_HANDLES.get(key)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
            _LOG_FILE_HANDLES[key] = None

    if master_log_path:
        os.makedirs(os.path.dirname(master_log_path) or ".", exist_ok=True)
        _LOG_FILE_HANDLES["master"] = open(master_log_path, "a", encoding="utf-8")
    if loop_log_path:
        os.makedirs(os.path.dirname(loop_log_path) or ".", exist_ok=True)
        _LOG_FILE_HANDLES["loop"] = open(loop_log_path, "a", encoding="utf-8")


def _print_safe(message: str) -> None:
    """
    Runtime guard against accidental duplicated loop-progress prints.
    Suppresses repeated loop status lines, even when separator lines are interleaved.
    """
    text = str(message)
    if "LOOP" in text and "REWRITE PHASE ---" in text:
        _SEEN_LOOP_STATUS_LINES.clear()
        match = re.search(r"LOOP\s+(\d+)", text)
        if match:
            loop_id = int(match.group(1))
            _LOOP_TIMING[loop_id] = time.time()
            text = f"{text} start={datetime.now().isoformat(timespec='seconds')}"

    noisy = ("progress: locked" in text) or text.startswith("Human attention required for ")
    if noisy:
        if text in _SEEN_LOOP_STATUS_LINES:
            return
        _SEEN_LOOP_STATUS_LINES.add(text)
    elif _LAST_PRINT_LINE.get("value") == text:
        return
    _LAST_PRINT_LINE["value"] = text
    _ORIGINAL_PRINT_SAFE(text)
    for key in ("master", "loop"):
        handle = _LOG_FILE_HANDLES.get(key)
        if handle is None:
            continue
        try:
            handle.write(text + "\n")
            handle.flush()
        except Exception:
            pass

    # Emit per-loop elapsed once when progress line first appears.
    if "progress: locked" in text:
        match = re.search(r"Loop\s+(\d+)\s+progress", text)
        if match:
            loop_id = int(match.group(1))
            if loop_id in _LOOP_TIMING and loop_id not in _LOOP_ELAPSED_PRINTED:
                elapsed = max(0.0, time.time() - float(_LOOP_TIMING[loop_id]))
                elapsed_line = f"Loop {loop_id} elapsed: {elapsed:.1f}s"
                _LOOP_ELAPSED_PRINTED.add(loop_id)
                _ORIGINAL_PRINT_SAFE(elapsed_line)
                for key in ("master", "loop"):
                    handle = _LOG_FILE_HANDLES.get(key)
                    if handle is None:
                        continue
                    try:
                        handle.write(elapsed_line + "\n")
                        handle.flush()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
