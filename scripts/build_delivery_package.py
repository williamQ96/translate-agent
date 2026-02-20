#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a delivery package for handoff.

What it does:
1) Create a delivery folder.
2) Aggregate translated chunks into one book markdown in delivery root.
3) Copy original English chunks into `original_chunks/`.
4) Copy latest translated chunks into `translated_chunks/`.
5) Generate Chinese readable audit report (markdown + csv) from latest audit json.
6) Copy glossary json into delivery root.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _setup_console_encoding() -> None:
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


def _latest_rewrite_run(rewrites_root: Path) -> Path:
    runs = [p for p in rewrites_root.glob("rewrite_loop_run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No rewrite_loop_run_* found under: {rewrites_root}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _extract_entries(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "audits", "chunks", "items", "data"):
        val = payload.get(key)
        if isinstance(val, list) and (not val or isinstance(val[0], dict)):
            return [x for x in val if isinstance(x, dict)]
    return []


def _audit_entries_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(_extract_entries(payload))


def _best_audit_json(audits_dir: Path, expected_chunks: int) -> Path:
    if not audits_dir.is_dir():
        raise FileNotFoundError(f"Audit dir not found: {audits_dir}")
    candidates = [
        p
        for p in audits_dir.glob("*.json")
        if p.is_file() and not p.name.startswith("audit_rewrite_seed_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No audit json found in: {audits_dir}")

    scored: list[tuple[int, float, Path]] = []
    for path in candidates:
        count = _audit_entries_count(path)
        mtime = path.stat().st_mtime
        scored.append((count, mtime, path))

    # Prefer most complete report (max entry count), then latest by mtime.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_count = scored[0][0]
    best_group = [item for item in scored if item[0] == best_count]
    best_group.sort(key=lambda item: item[1], reverse=True)
    best = best_group[0][2]

    # If completeness is very low, fall back to latest file to avoid hard failure.
    if expected_chunks > 0 and best_count < max(1, int(expected_chunks * 0.3)):
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest
    return best


def _chunk_sort_key(path: Path) -> int:
    match = re.match(r"^chunk_(\d+)\.md$", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _list_chunks(chunks_dir: Path) -> list[Path]:
    items = [p for p in chunks_dir.glob("chunk_*.md") if p.is_file()]
    items.sort(key=_chunk_sort_key)
    return items


def _merge_effective_chunks(base_chunks_dir: Path, overlay_chunks_dir: Path) -> list[Path]:
    base = { _chunk_sort_key(p): p for p in _list_chunks(base_chunks_dir) }
    overlay = { _chunk_sort_key(p): p for p in _list_chunks(overlay_chunks_dir) }
    all_ids = sorted({k for k in base.keys() if k < 10**9} | {k for k in overlay.keys() if k < 10**9})
    merged: list[Path] = []
    for chunk_id in all_ids:
        if chunk_id in overlay:
            merged.append(overlay[chunk_id])
        elif chunk_id in base:
            merged.append(base[chunk_id])
    return merged


def _strip_chunk_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("<!--"):
        return "\n".join(lines[1:]).lstrip("\n")
    return text.strip()


def _aggregate_book(chunks: list[Path], out_path: Path) -> None:
    parts: list[str] = []
    for path in chunks:
        text = path.read_text(encoding="utf-8")
        body = _strip_chunk_header(text).strip()
        if body:
            parts.append(body)
    out_path.write_text("\n\n".join(parts) + ("\n" if parts else ""), encoding="utf-8")


def _copy_chunks(chunks: list[Path], target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in chunks:
        shutil.copy2(path, target_dir / path.name)
    return len(chunks)


def _run_readable_report(report_script: Path, input_json: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(report_script),
        "--input",
        str(input_json),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def _run_full_audit(source_chunks_dir: Path, translated_chunks_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.audit",
        "--source-chunks-dir",
        str(source_chunks_dir),
        "--chunks-dir",
        str(translated_chunks_dir),
    ]
    subprocess.run(cmd, check=True)


def _write_manifest(
    manifest_path: Path,
    run_dir: Path,
    audit_json: Path,
    original_count: int,
    translated_count: int,
    book_name: str,
    glossary_name: str,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Delivery Manifest",
        "",
        f"- generated_at: {now}",
        f"- source_run_dir: `{run_dir.as_posix()}`",
        f"- latest_audit_json: `{audit_json.name}`",
        f"- original_chunk_count: {original_count}",
        f"- translated_chunk_count: {translated_count}",
        "",
        "## Package Content",
        "",
        f"1. `{book_name}`: aggregated translated book markdown.",
        "2. `original_chunks/`: original English chunks.",
        "3. `translated_chunks/`: latest translated chunks.",
        "4. `audit_*.json` + readable report files generated from latest audit.",
        f"5. `{glossary_name}`: glossary json.",
        "",
    ]
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _setup_console_encoding()
    parser = argparse.ArgumentParser(description="Build delivery package for translated book handoff.")
    parser.add_argument("--run-dir", default="", help="Rewrite run dir. If empty, auto-pick latest run.")
    parser.add_argument(
        "--rewrites-root",
        default="data/output/rewrites",
        help="Root dir for rewrite_loop_run_* (used when --run-dir is empty).",
    )
    parser.add_argument("--source-chunks-dir", default="data/output/source_chunks", help="Original chunks dir.")
    parser.add_argument(
        "--base-chunks-dir",
        default="data/output/chunks",
        help="Base translated chunks dir. Effective translated chunks = base + overlay(run/chunks).",
    )
    parser.add_argument(
        "--translated-chunks-dir",
        default="",
        help="Translated chunks dir. If empty, use <run-dir>/chunks.",
    )
    parser.add_argument(
        "--audits-dir",
        default="",
        help="Audit dir. If empty, use <run-dir>/audits.",
    )
    parser.add_argument("--glossary", default="data/glossary.json", help="Glossary file path.")
    parser.add_argument("--delivery-root", default="data/output/delivery", help="Delivery root dir.")
    parser.add_argument("--book-name", default="rewritten_translated.md", help="Aggregated book file name.")
    parser.add_argument(
        "--no-refresh-audit",
        action="store_true",
        help="Skip fresh full audit on delivery translated chunks.",
    )
    parser.add_argument(
        "--report-script",
        default="scripts/audit_json_to_cn_report.py",
        help="Script path for readable Chinese audit report generation.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_rewrite_run(Path(args.rewrites_root))
    source_chunks_dir = Path(args.source_chunks_dir)
    base_chunks_dir = Path(args.base_chunks_dir)
    translated_chunks_dir = Path(args.translated_chunks_dir) if args.translated_chunks_dir else (run_dir / "chunks")
    audits_dir = Path(args.audits_dir) if args.audits_dir else (run_dir / "audits")
    glossary_path = Path(args.glossary)
    report_script = Path(args.report_script)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not source_chunks_dir.is_dir():
        raise FileNotFoundError(f"Source chunks dir not found: {source_chunks_dir}")
    if not base_chunks_dir.is_dir():
        raise FileNotFoundError(f"Base chunks dir not found: {base_chunks_dir}")
    if not translated_chunks_dir.is_dir():
        raise FileNotFoundError(f"Translated chunks dir not found: {translated_chunks_dir}")
    if not report_script.is_file():
        raise FileNotFoundError(f"Readable report script not found: {report_script}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    delivery_dir = Path(args.delivery_root) / f"delivery_{ts}"
    delivery_dir.mkdir(parents=True, exist_ok=True)

    original_dst = delivery_dir / "original_chunks"
    translated_dst = delivery_dir / "translated_chunks"

    original_chunks = _list_chunks(source_chunks_dir)
    translated_chunks = _merge_effective_chunks(base_chunks_dir=base_chunks_dir, overlay_chunks_dir=translated_chunks_dir)
    if not translated_chunks:
        raise RuntimeError(f"No chunk_*.md found in translated chunks dir: {translated_chunks_dir}")

    original_count = _copy_chunks(original_chunks, original_dst)
    translated_count = _copy_chunks(translated_chunks, translated_dst)

    if args.no_refresh_audit:
        audit_json = _best_audit_json(audits_dir, expected_chunks=len(translated_chunks))
    else:
        _run_full_audit(source_chunks_dir=source_chunks_dir, translated_chunks_dir=translated_dst)
        fresh_audits_dir = delivery_dir / "audits"
        audit_json = _best_audit_json(fresh_audits_dir, expected_chunks=len(translated_chunks))

    book_path = delivery_dir / args.book_name
    _aggregate_book(translated_chunks, book_path)

    shutil.copy2(audit_json, delivery_dir / audit_json.name)
    _run_readable_report(report_script=report_script, input_json=audit_json, out_dir=delivery_dir)

    glossary_name = glossary_path.name
    if glossary_path.is_file():
        shutil.copy2(glossary_path, delivery_dir / glossary_name)
    else:
        glossary_name = f"{glossary_path.name} (missing)"

    _write_manifest(
        manifest_path=delivery_dir / "DELIVERY_MANIFEST.md",
        run_dir=run_dir,
        audit_json=audit_json,
        original_count=original_count,
        translated_count=translated_count,
        book_name=args.book_name,
        glossary_name=glossary_name,
    )

    print("=" * 60)
    print("DELIVERY PACKAGE READY")
    print(f"run_dir: {run_dir}")
    print(f"delivery_dir: {delivery_dir}")
    print(f"book: {book_path.name}")
    print(f"latest_audit: {audit_json.name}")
    print(f"original_chunks: {original_count}")
    print(f"translated_chunks: {translated_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
