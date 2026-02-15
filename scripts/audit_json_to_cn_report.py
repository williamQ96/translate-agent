#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将审计 JSON 转成中文可读报告（Markdown）和表格（CSV）。

用法示例：
python scripts/audit_json_to_cn_report.py ^
  --input "data/output/rewrites/rewrite_loop_run_20260213_174529/loop1/audit_loop_01_20260213_210025.json"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ENTRY_KEY_PRIORITY = ["results", "audits", "chunks", "items", "data"]
CHUNK_KEYS = ["chunk_num", "chunk", "chunk_id", "id"]
REASON_KEYS = [
    "reason",
    "rationale",
    "feedback",
    "comment",
    "comments",
    "notes",
    "summary",
    "analysis",
    "explanation",
]


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


def _first_non_empty(mapping: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _normalize_issues(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "；".join(parts)
    text = str(value).strip()
    return text


def _extract_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ENTRY_KEY_PRIORITY:
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value

    # 宽松兜底：扫描所有顶层字段，找看起来像审计条目的列表
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if any("score" in item or "issues" in item for item in value if isinstance(item, dict)):
                return [item for item in value if isinstance(item, dict)]

    return []


def _get_chunk_id(entry: dict[str, Any], fallback_index: int) -> str:
    for key in CHUNK_KEYS:
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return str(fallback_index)


def _to_rows(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, entry in enumerate(entries, start=1):
        chunk_id = _get_chunk_id(entry, index)
        score = str(entry.get("score", "")).strip()
        verdict = str(entry.get("verdict", "")).strip()
        issues_text = _normalize_issues(entry.get("issues"))
        reason_text = _first_non_empty(entry, REASON_KEYS, default="")
        if not reason_text:
            reason_text = issues_text or "（未提供）"

        human_attention = "是" if bool(entry.get("needs_human_attention")) else "否"

        rows.append(
            {
                "Chunk": chunk_id,
                "分数": score,
                "判定": verdict or "（未提供）",
                "原因": reason_text,
                "问题": issues_text or "（未提供）",
                "人工关注": human_attention,
            }
        )
    return rows


def _avg_score(rows: list[dict[str, str]]) -> float:
    nums: list[float] = []
    for row in rows:
        try:
            nums.append(float(row["分数"]))
        except (ValueError, TypeError):
            continue
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Chunk", "分数", "判定", "人工关注", "原因", "问题"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_markdown(rows: list[dict[str, str]], source_path: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg = _avg_score(rows)
    total = len(rows)

    lines: list[str] = []
    lines.append("# 审计报告（中文可读版）")
    lines.append("")
    lines.append(f"- 来源文件：`{source_path.as_posix()}`")
    lines.append(f"- 生成时间：`{timestamp}`")
    lines.append(f"- 总条目：`{total}`")
    lines.append(f"- 平均分：`{avg:.2f}/10`")
    lines.append("")
    lines.append("## 总览表")
    lines.append("")
    lines.append("| Chunk | 分数 | 判定 | 人工关注 |")
    lines.append("|---|---:|---|---|")
    for row in rows:
        lines.append(
            f"| {row['Chunk']} | {row['分数']} | {row['判定']} | {row['人工关注']} |"
        )
    lines.append("")
    lines.append("## 逐条详情")
    lines.append("")
    for row in rows:
        lines.append(f"### Chunk {row['Chunk']}")
        lines.append(f"- 分数：{row['分数']}/10")
        lines.append(f"- 判定：{row['判定']}")
        lines.append(f"- 人工关注：{row['人工关注']}")
        lines.append(f"- 原因：{row['原因']}")
        lines.append(f"- 问题：{row['问题']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _convert_one(input_path: Path, out_dir: Path | None = None, md_name: str = "", csv_name: str = "") -> tuple[Path, Path, int, float]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    entries = _extract_entries(payload)
    if not entries:
        raise ValueError(f"未在 JSON 中找到可识别的审计条目列表：{input_path}")

    rows = _to_rows(entries)
    target_dir = out_dir if out_dir else input_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    final_md_name = md_name or f"{stem}_可读报告.md"
    final_csv_name = csv_name or f"{stem}_表格.csv"
    md_path = target_dir / final_md_name
    csv_path = target_dir / final_csv_name

    _write_markdown(rows, input_path, md_path)
    _write_csv(rows, csv_path)
    return md_path, csv_path, len(rows), _avg_score(rows)


def main() -> None:
    _setup_console_encoding()
    parser = argparse.ArgumentParser(description="审计 JSON -> 中文可读 Markdown + CSV")
    parser.add_argument("--input", default="", help="单个审计 JSON 文件路径")
    parser.add_argument(
        "--run-dir",
        default="",
        help="批量模式：运行目录（例如 data/output/rewrites/rewrite_loop_run_xxx），会扫描 loop*/audit_loop_*.json",
    )
    parser.add_argument(
        "--pattern",
        default="loop*/audit_loop_*.json",
        help="批量扫描模式（相对 --run-dir），默认：loop*/audit_loop_*.json",
    )
    parser.add_argument("--out-dir", default="", help="输出目录（默认与输入文件同目录）")
    parser.add_argument("--md-name", default="", help="输出 Markdown 文件名（可选）")
    parser.add_argument("--csv-name", default="", help="输出 CSV 文件名（可选）")
    args = parser.parse_args()

    if not args.input and not args.run_dir:
        raise ValueError("必须提供 --input 或 --run-dir")

    shared_out_dir = Path(args.out_dir) if args.out_dir else None

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{input_path}")
        md_path, csv_path, count, avg = _convert_one(
            input_path=input_path,
            out_dir=shared_out_dir,
            md_name=args.md_name,
            csv_name=args.csv_name,
        )
        print("转换完成：")
        print(f"- Markdown: {md_path}")
        print(f"- CSV: {csv_path}")
        print(f"- 条目数: {count}")
        print(f"- 平均分: {avg:.2f}/10")
        return

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"找不到运行目录：{run_dir}")

    json_files = sorted(run_dir.glob(args.pattern))
    if not json_files:
        raise FileNotFoundError(f"未找到匹配文件：{run_dir / args.pattern}")

    ok = 0
    failed = 0
    for file_path in json_files:
        try:
            target_dir = shared_out_dir if shared_out_dir else file_path.parent
            md_path, csv_path, count, avg = _convert_one(
                input_path=file_path,
                out_dir=target_dir,
                md_name=args.md_name,
                csv_name=args.csv_name,
            )
            ok += 1
            print(f"[OK] {file_path} -> {md_path.name}, {csv_path.name} | 条目={count} | 平均分={avg:.2f}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {file_path} -> {exc}")

    print("批量转换完成：")
    print(f"- 成功: {ok}")
    print(f"- 失败: {failed}")
    print(f"- 总计: {len(json_files)}")


if __name__ == "__main__":
    main()
