#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute success metrics for rewrite loop runs.

Usage:
python scripts/compute_success_metrics.py ^
  --run-dir "data/output/rewrites/rewrite_loop_run_20260213_174529" ^
  --target-score 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_audit_reports(run_dir: Path) -> list[Path]:
    audits = sorted((run_dir / "audits").glob("audit_loop_*.json"))
    if audits:
        return audits
    # fallback: per-loop folders
    out: list[Path] = []
    for loop_dir in sorted(run_dir.glob("loop*")):
        out.extend(sorted(loop_dir.glob("audit_loop_*.json")))
    return out


def _extract_avg(report: dict[str, Any]) -> float:
    summary = report.get("summary", {})
    if isinstance(summary, dict) and "average_score" in summary:
        try:
            return float(summary.get("average_score", 0.0))
        except Exception:
            pass
    results = report.get("results", [])
    vals = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                vals.append(float(item.get("score", 0)))
            except Exception:
                pass
    return (sum(vals) / len(vals)) if vals else 0.0


def _extract_locked(report: dict[str, Any], target_score: int) -> int:
    results = report.get("results", [])
    count = 0
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                score = int(item.get("score", 0) or 0)
            except Exception:
                score = 0
            critical = any(
                bool(item.get(flag))
                for flag in ("hallucination", "omission", "mistranslation")
            ) or item.get("format_ok") is False
            if score >= target_score and not critical and not bool(item.get("needs_human_attention")):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rewrite-loop success metrics")
    parser.add_argument("--run-dir", required=True, help="rewrite_loop_run_xxx directory")
    parser.add_argument("--target-score", type=int, default=9)
    parser.add_argument(
        "--output",
        default="",
        help="output json path (default: <run-dir>/metrics_success.json)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    loop_state_path = run_dir / "loop_state.json"
    loop_state = _load_json(loop_state_path) if loop_state_path.exists() else {}
    reports = _iter_audit_reports(run_dir)
    if not reports:
        raise FileNotFoundError(f"no audit_loop_*.json found under: {run_dir}")

    first = _load_json(reports[0])
    last = _load_json(reports[-1])
    avg_first = _extract_avg(first)
    avg_last = _extract_avg(last)
    locked_first = _extract_locked(first, args.target_score)
    locked_last = _extract_locked(last, args.target_score)

    history = loop_state.get("history", [])
    max_loop = len(history) if isinstance(history, list) else 0
    locked_total = int(loop_state.get("locked_count", 0) or 0)
    total_chunks = int(loop_state.get("total_chunks", 0) or 0)
    human_attention = int(loop_state.get("human_attention_count", 0) or 0)

    metrics = {
        "run_dir": str(run_dir),
        "target_score": args.target_score,
        "report_count": len(reports),
        "first_report": reports[0].name,
        "last_report": reports[-1].name,
        "avg_score_first": round(avg_first, 3),
        "avg_score_last": round(avg_last, 3),
        "avg_score_delta": round(avg_last - avg_first, 3),
        "locked_first_estimate": locked_first,
        "locked_last_estimate": locked_last,
        "locked_delta_estimate": locked_last - locked_first,
        "locked_total_from_state": locked_total,
        "total_chunks": total_chunks,
        "human_attention_count": human_attention,
        "max_completed_loops": max_loop,
    }

    out_path = Path(args.output) if args.output else (run_dir / "metrics_success.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Success metrics written:")
    print(out_path)
    for key in (
        "avg_score_first",
        "avg_score_last",
        "avg_score_delta",
        "locked_last_estimate",
        "locked_delta_estimate",
        "locked_total_from_state",
        "human_attention_count",
        "max_completed_loops",
    ):
        print(f"- {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
