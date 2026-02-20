"""
Strict targeted rewrite batch.

Design goals:
1) Use only source text + glossary.
2) Force segmented rewrite to reduce long-chunk drift.
3) Avoid historical translation contamination.
4) Write outputs into an existing rewrite run overlay directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

from src.agents.writer import WriterAgent
from src.knowledge.glossary import GlossaryManager
from src.utils.config_loader import load_config


def _split_text(text: str, max_chars: int) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return [""]
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [raw]
    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        plen = len(para)
        projected = current_len + (2 if current else 0) + plen
        if current and projected > max_chars:
            segments.append("\n\n".join(current).strip())
            current = [para]
            current_len = plen
        else:
            current.append(para)
            current_len = projected if current_len else plen
    if current:
        segments.append("\n\n".join(current).strip())
    # Hard split very long single segment.
    hard: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            hard.append(seg)
            continue
        step = max(500, max_chars)
        for i in range(0, len(seg), step):
            part = seg[i : i + step].strip()
            if part:
                hard.append(part)
    return hard or [raw]


def _read_source_chunk(source_chunks_dir: Path, chunk_num: int) -> str:
    path = source_chunks_dir / f"chunk_{chunk_num:03d}.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("<!--"):
        return "\n".join(lines[1:]).lstrip("\n")
    return text


def _parse_chunk_ids(chunk_ids_text: str) -> list[int]:
    values: list[int] = []
    for token in chunk_ids_text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return sorted(set(values))


def _extract_proper_nouns(source_text: str, max_terms: int = 64) -> list[str]:
    text = (source_text or "").replace("\n", " ")
    # Capture title-cased terms and short multi-token name phrases.
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'’\-\.]*)"
        r"(?:\s+(?:[A-Z][A-Za-z0-9'’\-\.]*|und|of|the|and|von|der|de|la|le|du|des|zur|zu|auf|in|et|&)){0,5}"
    )
    stop = {
        "The",
        "And",
        "But",
        "Yet",
        "If",
        "As",
        "In",
        "On",
        "For",
        "With",
        "From",
        "By",
        "At",
        "To",
        "Of",
    }
    results: list[str] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        candidate = re.sub(r"\s+", " ", m.group(0).strip(" ,.;:!?()[]{}\"'"))
        if len(candidate) < 3 or candidate in stop:
            continue
        # Skip obvious sentence starts without noun-like shape.
        if " " not in candidate and candidate.endswith(("ing", "ed")):
            continue
        if candidate not in seen:
            seen.add(candidate)
            results.append(candidate)
        if len(results) >= max_terms:
            break
    return results


def _build_whitelist_text(
    source_text: str,
    glossary: GlossaryManager,
    relevant_glossary_text: str,
    max_terms: int,
    manual_pairs: list[tuple[str, str]] | None = None,
) -> str:
    lines: list[str] = []
    used: set[str] = set()
    source_lower = (source_text or "").lower()

    for src, dst in (manual_pairs or []):
        src = str(src).strip()
        dst = str(dst).strip()
        if not src or src in used:
            continue
        if src.lower() not in source_lower:
            continue
        lines.append(f"{src} => {dst}")
        used.add(src)
        if len(lines) >= max_terms:
            return "\n".join(lines)

    for row in (relevant_glossary_text or "").splitlines():
        row = row.strip()
        if not row or ":" not in row:
            continue
        term, tr = row.split(":", 1)
        term = term.strip()
        tr = tr.strip()
        if not term or term in used:
            continue
        lines.append(f"{term} => {tr}")
        used.add(term)
        if len(lines) >= max_terms:
            return "\n".join(lines)

    for term in _extract_proper_nouns(source_text, max_terms=max_terms * 2):
        if term in used:
            continue
        translation = glossary.terms.get(term, "") if isinstance(glossary.terms, dict) else ""
        translation = str(translation).strip()
        if translation and not translation.startswith("["):
            lines.append(f"{term} => {translation}")
        else:
            lines.append(f"{term} => [NO_GUESS_KEEP_ORIGINAL]")
        used.add(term)
        if len(lines) >= max_terms:
            break
    return "\n".join(lines)


def _load_manual_whitelist(path_text: str | None) -> list[tuple[str, str]]:
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Whitelist file not found: {path}")
    pairs: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            left, right = line.split("=>", 1)
            pairs.append((left.strip(), right.strip()))
        elif ":" in line:
            left, right = line.split(":", 1)
            pairs.append((left.strip(), right.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict source-only segmented rewrite for selected chunks.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing rewrite loop run dir (contains chunks overlay).",
    )
    parser.add_argument(
        "--source-chunks-dir",
        default="data/output/source_chunks",
        help="Source chunks directory.",
    )
    parser.add_argument(
        "--chunk-ids",
        required=True,
        help='Comma-separated chunk ids. Example: "31,33,36,38"',
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=1200,
        help="Max chars per rewrite segment.",
    )
    parser.add_argument(
        "--whitelist-max-terms",
        type=int,
        default=64,
        help="Max glossary/proper-noun whitelist terms per segment.",
    )
    parser.add_argument(
        "--whitelist-file",
        default="",
        help="Optional manual whitelist file. Line format: SourceTerm => TargetTerm",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    chunks_overlay = run_dir / "chunks"
    source_chunks_dir = Path(args.source_chunks_dir)
    if not chunks_overlay.is_dir():
        raise FileNotFoundError(f"Overlay chunks dir missing: {chunks_overlay}")
    if not source_chunks_dir.is_dir():
        raise FileNotFoundError(f"Source chunks dir missing: {source_chunks_dir}")

    cfg = load_config()
    glossary_path = Path(cfg.get("directories", {}).get("glossary", "data/glossary.json"))
    glossary = GlossaryManager(str(glossary_path))
    writer = WriterAgent()
    manual_whitelist = _load_manual_whitelist(args.whitelist_file)

    chunk_ids = _parse_chunk_ids(args.chunk_ids)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = run_dir / f"strict_batch_{ts}"
    before_dir = batch_dir / "before"
    after_dir = batch_dir / "after"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STRICT REWRITE BATCH")
    print(f"Run dir: {run_dir}")
    print(f"Chunks: {chunk_ids}")
    print(f"Segment size: {args.segment_size}")
    print("=" * 60)

    for chunk_num in chunk_ids:
        source_text = _read_source_chunk(source_chunks_dir, chunk_num)
        target_path = chunks_overlay / f"chunk_{chunk_num:03d}.md"
        if target_path.exists():
            before_text = target_path.read_text(encoding="utf-8")
            (before_dir / target_path.name).write_text(before_text, encoding="utf-8")

        segments = _split_text(source_text, max_chars=args.segment_size)
        outputs: list[str] = []
        print(f"\nchunk {chunk_num}: {len(segments)} segments")
        for idx, seg in enumerate(segments, 1):
            seg_glossary = glossary.get_relevant_glossary_text(seg)[:1400]
            whitelist_text = _build_whitelist_text(
                source_text=seg,
                glossary=glossary,
                relevant_glossary_text=seg_glossary,
                max_terms=max(8, int(args.whitelist_max_terms)),
                manual_pairs=manual_whitelist,
            )
            glossary_payload = (seg_glossary or "").strip()
            if whitelist_text:
                glossary_payload = (
                    f"{glossary_payload}\n\n[TERM_WHITELIST_MUST_KEEP]\n"
                    f"{whitelist_text}\n\n"
                    "[RULE]\n"
                    "1) Proper nouns and titles must follow whitelist.\n"
                    "2) If term has [NO_GUESS_KEEP_ORIGINAL], keep original spelling (no guessing).\n"
                    "3) Do not add explanations not in source."
                ).strip()
            rewritten = writer.rewrite(
                source_text=seg,
                previous_translation="",
                current_translation="[NO_HISTORY_TRANSLATION]",
                issues=[
                    "STRICT_LITERAL",
                    "SOURCE_ONLY",
                    "NO_PRIOR_TRANSLATION",
                    "PROPER_NOUN_LOCK",
                    "TERM_WHITELIST_LOCK",
                ],
                issue_tags=[
                    "FORCE_SOURCE_ONLY",
                    "STRICT_LITERAL",
                    "NO_HISTORY_TRANSLATION",
                    "PROPER_NOUN_LOCK",
                    "TERM_WHITELIST_LOCK",
                ],
                score=8,
                loop_index=999,
                glossary_text=glossary_payload,
                rag_context=(
                    "[STRONG_TEMPLATE]\n"
                    "- Keep markdown structure only if present in source.\n"
                    "- Keep <table>...</table> HTML tags and structure intact.\n"
                    "- Translate sentence-by-sentence conservatively.\n"
                    "- Preserve names/titles strictly by whitelist.\n"
                    "- No commentary, no invented references."
                ),
                stagnation_rounds=9,
            )
            route = getattr(writer, "last_route", {}) or {}
            print(
                f"  seg {idx}/{len(segments)} done | chars={len(seg)} "
                f"| model={route.get('model','?')} escalated={route.get('escalated', False)}"
            )
            if whitelist_text:
                whitelist_path = after_dir / f"chunk_{chunk_num:03d}_seg_{idx:03d}_whitelist.txt"
                whitelist_path.write_text(whitelist_text, encoding="utf-8")
            outputs.append((rewritten or "").strip())

        final_text = "\n\n".join([x for x in outputs if x]).strip()
        header = (
            f"<!-- Chunk {chunk_num} | STRICT_BATCH {ts} | "
            "source+glossary+segmented | no_history_translation -->\n\n"
        )
        target_path.write_text(header + final_text, encoding="utf-8")
        (after_dir / target_path.name).write_text(header + final_text, encoding="utf-8")
        print(f"  wrote: {target_path}")

    print("\nDone.")
    print(f"Batch artifacts: {batch_dir}")


if __name__ == "__main__":
    main()
