#!/usr/bin/env python
"""
Archive generated artifacts from data/output into legacy_data per run.

Default behavior:
- Archive common runtime artifacts:
  dirs: audits, chunks, logs, rewrites, source_chunks, _smoke_tests
  files: *_pipeline.log, *_progress.json, *_translated.md
- Exclude `legacy_data` itself.
- Keep `ocr/` unless --include-ocr is provided.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MoveItem:
    src: Path
    dst: Path
    kind: str  # file|dir


DEFAULT_DIRS = ["audits", "chunks", "logs", "rewrites", "source_chunks", "_smoke_tests"]
DEFAULT_FILE_PATTERNS = ["*_pipeline.log", "*_progress.json", "*_translated.md"]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 1000):
        candidate = parent / f"{stem}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot allocate destination name for: {path}")


def collect_items(output_root: Path, archive_root: Path, include_ocr: bool) -> list[MoveItem]:
    items: list[MoveItem] = []

    # Directories
    candidate_dirs = list(DEFAULT_DIRS)
    if include_ocr:
        candidate_dirs.append("ocr")

    for name in candidate_dirs:
        src = output_root / name
        if not src.exists():
            continue
        dst = archive_root / name
        dst = _unique_destination(dst)
        items.append(MoveItem(src=src, dst=dst, kind="dir"))

    # Files by pattern
    seen = {str(item.src.resolve()) for item in items}
    for pattern in DEFAULT_FILE_PATTERNS:
        for path_str in glob.glob(str(output_root / pattern)):
            src = Path(path_str)
            if not src.is_file():
                continue
            resolved = str(src.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            dst = _unique_destination(archive_root / src.name)
            items.append(MoveItem(src=src, dst=dst, kind="file"))

    return items


def run_archive(
    output_root: Path,
    legacy_root: Path,
    run_name: str,
    include_ocr: bool,
    dry_run: bool,
    copy_mode: bool,
) -> dict:
    archive_root = legacy_root / run_name
    items = collect_items(output_root=output_root, archive_root=archive_root, include_ocr=include_ocr)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "legacy_root": str(legacy_root),
        "archive_root": str(archive_root),
        "include_ocr": include_ocr,
        "dry_run": dry_run,
        "copy_mode": copy_mode,
        "count": len(items),
        "items": [
            {"src": str(item.src), "dst": str(item.dst), "kind": item.kind}
            for item in items
        ],
    }

    if dry_run:
        return summary

    archive_root.mkdir(parents=True, exist_ok=True)
    for item in items:
        item.dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode:
            if item.kind == "dir":
                shutil.copytree(item.src, item.dst)
            else:
                shutil.copy2(item.src, item.dst)
        else:
            shutil.move(str(item.src), str(item.dst))

    manifest = archive_root / "manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive generated artifacts from data/output to legacy_data")
    parser.add_argument("--output-root", default="data/output", help="Output root path")
    parser.add_argument("--legacy-root", default="", help="Legacy root path (default: <output-root>/legacy_data)")
    parser.add_argument("--run-name", default="", help="Archive folder name (default: work_YYYYMMDD_HHMMSS)")
    parser.add_argument("--include-ocr", action="store_true", help="Also archive output/ocr directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not move/copy")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(f"output root not found: {output_root}")

    legacy_root = Path(args.legacy_root) if args.legacy_root else (output_root / "legacy_data")
    legacy_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip() or f"work_{_timestamp()}"
    summary = run_archive(
        output_root=output_root,
        legacy_root=legacy_root,
        run_name=run_name,
        include_ocr=args.include_ocr,
        dry_run=args.dry_run,
        copy_mode=args.copy,
    )

    print("Archive plan/result:")
    print(f"- output_root: {summary['output_root']}")
    print(f"- archive_root: {summary['archive_root']}")
    print(f"- mode: {'copy' if summary['copy_mode'] else 'move'}")
    print(f"- dry_run: {summary['dry_run']}")
    print(f"- include_ocr: {summary['include_ocr']}")
    print(f"- item_count: {summary['count']}")
    for item in summary["items"]:
        print(f"  - [{item['kind']}] {item['src']} -> {item['dst']}")

    if not args.dry_run:
        print(f"Manifest written: {Path(summary['archive_root']) / 'manifest.json'}")


if __name__ == "__main__":
    main()
