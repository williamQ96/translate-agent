#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出交付包：
1) 整书聚合文件 rewritten_translated.md
2) chunk 级 latest_chunks（完整分块，便于人工校验）

默认假设：
- base chunks: data/output/chunks
- rewrite run: data/output/rewrites/rewrite_loop_run_xxx
- rewrite overlay: <run-dir>/chunks（通常只包含被重写过的子集）

会将 overlay 覆盖到 base，得到“最新有效分块”，并输出到 delivery 目录。
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


CHUNK_RE = re.compile(r"^chunk_(\d+)\.md$", re.IGNORECASE)


def _chunk_num(path: Path) -> int | None:
    match = CHUNK_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _collect_chunks(directory: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not directory.is_dir():
        return out
    for item in directory.iterdir():
        if not item.is_file():
            continue
        num = _chunk_num(item)
        if num is None:
            continue
        out[num] = item
    return out


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_latest_chunks(base_dir: Path, overlay_dir: Path, latest_dir: Path) -> list[Path]:
    base_chunks = _collect_chunks(base_dir)
    overlay_chunks = _collect_chunks(overlay_dir)
    all_nums = sorted(set(base_chunks.keys()) | set(overlay_chunks.keys()))

    if not all_nums:
        raise FileNotFoundError("未找到 chunk 文件，请检查 --base-chunks-dir / --overlay-chunks-dir")

    _ensure_dir(latest_dir)
    written: list[Path] = []
    for num in all_nums:
        src = overlay_chunks.get(num) or base_chunks.get(num)
        if src is None:
            continue
        dst = latest_dir / f"chunk_{num:03d}.md"
        shutil.copy2(src, dst)
        written.append(dst)
    return written


def _assemble_book(chunks: list[Path], output_md: Path) -> None:
    parts: list[str] = []
    for chunk in sorted(chunks, key=lambda p: _chunk_num(p) or 0):
        text = chunk.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts.append(text)
    _ensure_dir(output_md.parent)
    output_md.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出交付包（整书 + 最新 chunk 目录）")
    parser.add_argument("--run-dir", required=True, help="rewrite run 目录，例如 data/output/rewrites/rewrite_loop_run_xxx")
    parser.add_argument("--base-chunks-dir", default="data/output/chunks", help="基础 chunks 目录")
    parser.add_argument("--overlay-chunks-dir", default="", help="覆盖层 chunks 目录（默认 <run-dir>/chunks）")
    parser.add_argument("--out-dir", default="", help="交付输出目录（默认 <run-dir>/delivery）")
    parser.add_argument("--book-name", default="rewritten_translated.md", help="整书输出文件名")
    parser.add_argument("--latest-chunks-name", default="latest_chunks", help="chunk 输出目录名")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"找不到 run 目录：{run_dir}")

    base_dir = Path(args.base_chunks_dir)
    overlay_dir = Path(args.overlay_chunks_dir) if args.overlay_chunks_dir else (run_dir / "chunks")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "delivery")
    latest_dir = out_dir / args.latest_chunks_name
    book_path = out_dir / args.book_name

    written_chunks = _copy_latest_chunks(base_dir=base_dir, overlay_dir=overlay_dir, latest_dir=latest_dir)
    _assemble_book(written_chunks, book_path)

    print("交付包导出完成：")
    print(f"- 整书文件: {book_path}")
    print(f"- 最新分块: {latest_dir}")
    print(f"- chunk 数量: {len(written_chunks)}")
    print(f"- base chunks: {base_dir}")
    print(f"- overlay chunks: {overlay_dir}")


if __name__ == "__main__":
    main()
