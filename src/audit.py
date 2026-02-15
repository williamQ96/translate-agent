"""
audit.py - Standalone translation quality auditor.

Compares source chunks against translated chunks, flags quality issues,
and can optionally rewrite flagged chunks.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.rag.store import RAGStore
from src.utils.config_loader import load_config


AUDIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是翻译质量审校员。请对比英文原文和中文译文，检查：
1. 幻觉（译文是否包含原文没有的信息）
2. 遗漏（原文关键信息是否丢失）
3. 误译（是否有明显翻译错误）
4. 格式（Markdown 是否保持）

重要：当输入是长文本抽样片段（BEGIN/MIDDLE/END）时，
只能基于对应片段做判断，不要因为压缩翻译或信息位置变化而误判为幻觉。
“相关上下文（RAG）”仅用于消歧和术语一致性辅助，不可把上下文里存在但原文片段没有的内容判为译文必须包含。
判定幻觉/遗漏/误译时，必须以“原文”字段为主依据。
当原文存在明显 OCR 噪声（例如标题缺字：`# pera in History`）时，
若译文在语义上明显对应正确词（如 `Opera`），不得判定为幻觉。
专有名词音译（如 Cage -> 凯奇）在语义对应时不得判定为幻觉。
若你无法给出可核验的具体证据，应降低严重性并避免误报。

输出格式（严格遵守）：
SCORE: [1-10]
HALLUCINATION: [YES/NO] — [说明]
OMISSION: [YES/NO] — [说明]
MISTRANSLATION: [YES/NO] — [说明]
FORMAT_OK: [YES/NO]
VERDICT: [PASS/REWRITE]""",
        ),
        (
            "user",
            "/no_think\n原文：\n{source}\n\n译文：\n{translation}\n\n相关上下文（RAG，仅辅助）：\n{rag_context}",
        ),
    ]
)


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是出版级翻译专家。以下译文存在质量问题，请根据原文重译。

要求：
1. 忠实原文，不添加、不遗漏。
2. 中文流畅自然。
3. 保留 Markdown 格式。
4. 只输出中文译文，不要解释。""",
        ),
        (
            "user",
            "/no_think\n已知问题：\n{issues}\n\n原文：\n{source}",
        ),
    ]
)


AUDIT_SAMPLE_SIZE = 3200
AUDIT_WINDOW_SIZE = 850
AUDIT_WINDOW_POINTS = (
    ("BEGIN", 0.08),
    ("MIDDLE", 0.50),
    ("END", 0.88),
)
DEFAULT_AUDIT_PASS_SCORE = 8
DEFAULT_AUDIT_RAG_K = 2
DEFAULT_AUDIT_RAG_MAX_CHARS = 1200


def _print_safe(text: str) -> None:
    """Print text safely on consoles that are not UTF-8 capable."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode())


def get_llm():
    config = load_config()
    model_config = config.get("model", {})
    return ChatOpenAI(
        model=model_config.get("name", "qwen3:30b"),
        base_url=model_config.get("api_base", "http://localhost:11434/v1"),
        api_key=model_config.get("api_key", "ollama"),
        temperature=0.1,
        max_tokens=512,
        timeout=float(model_config.get("audit_timeout", 120)),
        max_retries=0,
    )


def _parse_chunk_num(name: str) -> int | None:
    if not (name.startswith("chunk_") and name.endswith(".md")):
        return None
    try:
        return int(name.split("_")[1].split(".")[0])
    except (ValueError, IndexError):
        return None


def _strip_chunk_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("<!--"):
        return "\n".join(lines[1:]).lstrip("\n")
    return text.strip()


def _source_chunk_file_map(source_chunks_dir: str) -> dict[int, str]:
    if not source_chunks_dir or not os.path.isdir(source_chunks_dir):
        return {}
    mapping: dict[int, str] = {}
    for name in os.listdir(source_chunks_dir):
        num = _parse_chunk_num(name)
        if num is None:
            continue
        path = os.path.join(source_chunks_dir, name)
        if os.path.isfile(path):
            mapping[num] = path
    return mapping


def _source_chunk_map(source_chunks_dir: str) -> dict[int, str]:
    if not source_chunks_dir or not os.path.isdir(source_chunks_dir):
        return {}
    mapping: dict[int, str] = {}
    for name in os.listdir(source_chunks_dir):
        num = _parse_chunk_num(name)
        if num is None:
            continue
        path = os.path.join(source_chunks_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as file:
            mapping[num] = _strip_chunk_header(file.read())
    return mapping


def load_source_corpus(source_md: str = "", source_chunks_dir: str = "data/output/source_chunks") -> str:
    """Load source corpus from chunk files if available, else from source markdown."""
    chunk_map = _source_chunk_map(source_chunks_dir)
    if chunk_map:
        return "\n\n".join(chunk_map[num] for num in sorted(chunk_map))

    if source_md:
        with open(source_md, "r", encoding="utf-8") as file:
            return file.read()

    raise FileNotFoundError(
        "No source corpus available. Provide --source-chunks-dir with chunk_XXX.md files "
        "or provide --source markdown."
    )


def load_chunks(
    source_md: str,
    chunks_dir: str,
    chunk_numbers: list[int],
    source_chunks_dir: str = "data/output/source_chunks",
):
    """Load aligned source/translation chunk pairs."""
    source_map = _source_chunk_map(source_chunks_dir)
    source_file_map = _source_chunk_file_map(source_chunks_dir)
    total_chunks = len(source_map)

    if not source_map:
        if not source_md:
            raise FileNotFoundError(
                "No source chunks found and --source is empty. "
                "Provide --source-chunks-dir or --source markdown."
            )
        from src.pipeline import chunk_text

        config = load_config()
        with open(source_md, "r", encoding="utf-8") as file:
            source_text = file.read()

        chunk_config = config.get("chunking", {})
        all_source_chunks = chunk_text(
            source_text,
            chunk_size=chunk_config.get("chunk_size", 2000),
            overlap=chunk_config.get("overlap", 200),
        )
        source_map = {idx + 1: text for idx, text in enumerate(all_source_chunks)}
        source_file_map = {}
        total_chunks = len(all_source_chunks)

    pairs = []
    for chunk_num in chunk_numbers:
        source_chunk = source_map.get(chunk_num)
        if source_chunk is None:
            print(f"  WARN source chunk {chunk_num} not found, skipping")
            continue

        chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_num:03d}.md")
        if not os.path.exists(chunk_file):
            print(f"  WARN {chunk_file} not found, skipping")
            continue

        with open(chunk_file, "r", encoding="utf-8") as file:
            translation = file.read()

        lines = translation.split("\n")
        if lines and lines[0].startswith("<!--"):
            translation = "\n".join(lines[1:]).strip()

        pairs.append(
            {
                "chunk_num": chunk_num,
                "source": source_chunk,
                "translation": translation,
                "chunk_file": chunk_file,
                "source_chunk_file": source_file_map.get(chunk_num, ""),
            }
        )

    return pairs, total_chunks


def _extract_window(text: str, point: float, width: int) -> str:
    if not text:
        return ""
    center = int(len(text) * point)
    start = max(0, center - (width // 2))
    end = min(len(text), start + width)
    return text[start:end]


def _build_audit_inputs(source: str, translation: str) -> tuple[str, str]:
    """For long chunks, audit aligned relative windows instead of naive head truncation."""
    if len(source) <= AUDIT_SAMPLE_SIZE and len(translation) <= AUDIT_SAMPLE_SIZE:
        return source, translation

    source_head = source[:500]
    translation_head = translation[:500]
    source_parts = []
    translation_parts = []
    source_parts.append(f"[HEAD]\n{source_head}")
    translation_parts.append(f"[HEAD]\n{translation_head}")
    for label, point in AUDIT_WINDOW_POINTS:
        source_parts.append(f"[{label}]\n{_extract_window(source, point, AUDIT_WINDOW_SIZE)}")
        translation_parts.append(f"[{label}]\n{_extract_window(translation, point, AUDIT_WINDOW_SIZE)}")

    return "\n\n".join(source_parts), "\n\n".join(translation_parts)


def _trim_for_prompt(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text[:max_chars]


def _derive_verdict(parsed: dict, pass_score: int) -> str:
    score = int(parsed.get("score", 0) or 0)
    has_critical_flag = any(
        [
            bool(parsed.get("hallucination")),
            bool(parsed.get("omission")),
            bool(parsed.get("mistranslation")),
            parsed.get("format_ok") is False,
        ]
    )
    if has_critical_flag:
        return "REWRITE"
    if score < pass_score:
        return "REWRITE"
    return "PASS"


def _extract_quoted_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    patterns = [
        r'"([^"\n]{4,120})"',
        r"“([^”\n]{4,120})”",
        r"'([^'\n]{4,120})'",
        r"‘([^’\n]{4,120})’",
    ]
    for pattern in patterns:
        for hit in re.findall(pattern, text):
            frag = hit.strip()
            if frag:
                fragments.append(frag)
    return fragments


def _looks_like_ocr_noisy_fragment(source_text: str) -> bool:
    if not source_text:
        return False
    short_text = source_text[:280]
    if re.search(r"^\s*#+\s+[a-z][A-Za-z]{2,}\b", short_text, re.MULTILINE):
        return True
    if re.search(r"\b[A-Za-z]{1,2}[’']?[A-Za-z]{1,2}\s+[A-Za-z]{1,2}\b", short_text):
        return True
    return False


def _hallucination_evidence_missing(issues: list[str], translation_text: str) -> bool:
    hallucination_lines = [line for line in issues if line.upper().replace("：", ":").startswith("HALLUCINATION:")]
    if not hallucination_lines:
        return False

    checks = 0
    misses = 0
    for line in hallucination_lines:
        for frag in _extract_quoted_fragments(line):
            checks += 1
            if frag not in translation_text:
                misses += 1
    if checks == 0:
        return False
    return misses == checks


def _mark_human_attention(parsed: dict, reason: str) -> None:
    parsed["needs_human_attention"] = True
    current = parsed.get("human_attention_reason", "")
    reasons = [item for item in [current, reason] if item]
    parsed["human_attention_reason"] = "; ".join(dict.fromkeys(reasons))


def _apply_audit_guardrails(parsed: dict, source_for_audit: str, translation_for_audit: str) -> dict:
    parsed.setdefault("needs_human_attention", False)
    parsed.setdefault("human_attention_reason", "")

    is_sampled = "[BEGIN]" in source_for_audit and "[MIDDLE]" in source_for_audit and "[END]" in source_for_audit
    short_fragment = len(source_for_audit) <= 260 and len(translation_for_audit) <= 260
    ocr_noisy = _looks_like_ocr_noisy_fragment(source_for_audit)

    if _hallucination_evidence_missing(parsed.get("issues", []), translation_for_audit):
        parsed["hallucination"] = False
        parsed["issues"] = [
            item for item in parsed.get("issues", []) if not item.upper().replace("：", ":").startswith("HALLUCINATION:")
        ]
        _mark_human_attention(parsed, "AUDIT_HALLUCINATION_EVIDENCE_NOT_FOUND")

    if ocr_noisy and short_fragment and (parsed.get("hallucination") or parsed.get("omission") or parsed.get("score", 0) <= 4):
        _mark_human_attention(parsed, "OCR_NOISY_SHORT_FRAGMENT")

    if is_sampled and parsed.get("score", 0) <= 4 and parsed.get("hallucination") and not parsed.get("mistranslation"):
        _mark_human_attention(parsed, "LOW_CONFIDENCE_SAMPLED_WINDOW")

    return parsed


def _normalize_audit_result(parsed: dict, pass_score: int) -> dict:
    parsed["score"] = max(0, min(10, int(parsed.get("score", 0) or 0)))
    parsed["needs_human_attention"] = bool(parsed.get("needs_human_attention", False))
    parsed["human_attention_reason"] = (parsed.get("human_attention_reason") or "").strip()
    parsed["model_verdict"] = parsed.get("verdict", "UNKNOWN")
    parsed["verdict"] = _derive_verdict(parsed, pass_score)
    if parsed["verdict"] == "REWRITE" and not parsed.get("issues"):
        score = int(parsed.get("score", 0) or 0)
        if score < pass_score:
            parsed["issues"] = [f"SCORE_BELOW_TARGET: {score}/{pass_score}"]
        else:
            parsed["issues"] = ["QUALITY_FLAG_DETECTED: rewrite required"]
    if parsed.get("needs_human_attention"):
        parsed["verdict"] = "REWRITE"
        marker = "HUMAN_REVIEW: this chunk needs human attention"
        if marker not in parsed.get("issues", []):
            parsed.setdefault("issues", []).append(marker)
    return parsed


def audit_chunk(llm, source: str, translation: str, rag_context: str = "") -> dict:
    """Run audit on a single chunk and parse standardized output."""
    chain = AUDIT_PROMPT | llm | StrOutputParser()
    source_for_audit, translation_for_audit = _build_audit_inputs(source, translation)
    config = load_config()
    pass_score = int(config.get("audit", {}).get("pass_score", DEFAULT_AUDIT_PASS_SCORE))
    pass_score = max(1, min(10, pass_score))

    result = chain.invoke(
        {
            "source": source_for_audit,
            "translation": translation_for_audit,
            "rag_context": rag_context or "[NONE]",
        }
    )

    parsed = {
        "raw": result,
        "score": 0,
        "hallucination": False,
        "omission": False,
        "mistranslation": False,
        "format_ok": True,
        "verdict": "PASS",
        "issues": [],
        "needs_human_attention": False,
        "human_attention_reason": "",
    }

    for line in result.strip().split("\n"):
        row = line.strip()
        row_upper = row.upper()
        normalized = row_upper.replace("：", ":")
        if normalized.startswith("SCORE:"):
            try:
                match = re.search(r"(\d+)", normalized)
                if match:
                    parsed["score"] = int(match.group(1))
            except (ValueError, IndexError):
                pass
        elif normalized.startswith("HALLUCINATION:") and "YES" in normalized:
            parsed["hallucination"] = True
            parsed["issues"].append(row)
        elif normalized.startswith("OMISSION:") and "YES" in normalized:
            parsed["omission"] = True
            parsed["issues"].append(row)
        elif normalized.startswith("MISTRANSLATION:") and "YES" in normalized:
            parsed["mistranslation"] = True
            parsed["issues"].append(row)
        elif normalized.startswith("FORMAT_OK:") and "NO" in normalized:
            parsed["format_ok"] = False
            parsed["issues"].append(row)
        elif normalized.startswith("VERDICT:"):
            parsed["verdict"] = "REWRITE" if "REWRITE" in normalized else "PASS"

    parsed = _apply_audit_guardrails(parsed, source_for_audit, translation_for_audit)
    return _normalize_audit_result(parsed, pass_score)


def _issue_tags(item: dict) -> list[str]:
    tags = []
    if item.get("hallucination"):
        tags.append("HALLUCINATION")
    if item.get("omission"):
        tags.append("OMISSION")
    if item.get("mistranslation"):
        tags.append("MISTRANSLATION")
    if item.get("format_ok") is False:
        tags.append("FORMAT")
    return tags


def _build_rewrite_chunk_report(results: list[dict], pass_score: int) -> dict:
    flagged = [item for item in results if item.get("verdict") == "REWRITE"]
    chunk_items = []
    for item in flagged:
        chunk_items.append(
            {
                "chunk_num": item.get("chunk_num"),
                "score": int(item.get("score", 0) or 0),
                "issues": item.get("issues", []),
                "tags": _issue_tags(item),
                "hallucination": bool(item.get("hallucination")),
                "omission": bool(item.get("omission")),
                "mistranslation": bool(item.get("mistranslation")),
                "format_ok": item.get("format_ok", True),
                "needs_human_attention": bool(item.get("needs_human_attention", False)),
                "human_attention_reason": item.get("human_attention_reason", ""),
                "verdict": "REWRITE",
                "chunk_file": item.get("chunk_file", ""),
                "source_chunk_file": item.get("source_chunk_file", ""),
                "model_verdict": item.get("model_verdict", ""),
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "target_pass_score": pass_score,
        "total_flagged": len(chunk_items),
        "results": chunk_items,
    }


def rewrite_chunk(llm, source: str, issues: list[str]) -> str:
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    issues_text = "\n".join(issues) if issues else "质量不达标"
    return chain.invoke({"source": source, "issues": issues_text})


def main():
    parser = argparse.ArgumentParser(description="Translation Quality Auditor")
    parser.add_argument(
        "--source",
        "-s",
        default="",
        help="Path to source markdown (e.g. data/output/ocr/MinerU_processed_combined.md)",
    )
    parser.add_argument(
        "--source-chunks-dir",
        default="data/output/source_chunks",
        help="Directory containing source chunk_XXX.md files (preferred)",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=int,
        nargs="+",
        help="Specific chunk numbers to audit (e.g. --chunks 28 35 42)",
    )
    parser.add_argument(
        "--range",
        "-r",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of chunks to audit (e.g. --range 28 50)",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/output/chunks",
        help="Directory containing chunk_XXX.md files",
    )
    parser.add_argument("--rewrite", action="store_true", help="Auto-rewrite chunks flagged as REWRITE")
    args = parser.parse_args()

    if args.chunks:
        chunk_numbers = args.chunks
    elif args.range:
        chunk_numbers = list(range(args.range[0], args.range[1] + 1))
    else:
        chunk_files = sorted(
            [file_name for file_name in os.listdir(args.chunks_dir) if file_name.startswith("chunk_") and file_name.endswith(".md")]
        )
        chunk_numbers = [int(name.split("_")[1].split(".")[0]) for name in chunk_files]

    if not chunk_numbers:
        print("No chunks to audit.")
        return

    print("\n" + "=" * 50)
    print("  TRANSLATION AUDIT")
    source_label = args.source_chunks_dir if os.path.isdir(args.source_chunks_dir) else args.source
    print(f"  Source: {source_label}")
    print(f"  Chunks: {chunk_numbers[0]}-{chunk_numbers[-1]} ({len(chunk_numbers)} total)")
    print(f"  Rewrite: {'ON' if args.rewrite else 'OFF'}")
    print("=" * 50 + "\n")

    pairs, total_chunks = load_chunks(
        args.source,
        args.chunks_dir,
        chunk_numbers,
        source_chunks_dir=args.source_chunks_dir,
    )
    if not pairs:
        print("No valid chunk pairs found.")
        return

    config = load_config()
    audit_cfg = config.get("audit", {})
    use_rag = bool(audit_cfg.get("use_rag", True))
    audit_rag_k = int(audit_cfg.get("rag_k", DEFAULT_AUDIT_RAG_K))
    audit_rag_max_chars = int(audit_cfg.get("rag_max_chars", DEFAULT_AUDIT_RAG_MAX_CHARS))
    rag_store: RAGStore | None = None
    if use_rag:
        try:
            source_corpus = load_source_corpus(args.source, source_chunks_dir=args.source_chunks_dir)
            rag_store = RAGStore()
            rag_store.index_document(source_corpus)
            print(f"  Audit RAG: ON (k={audit_rag_k}, max_chars={audit_rag_max_chars})")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN Audit RAG init failed, fallback to non-RAG audit: {exc}")
            rag_store = None
    else:
        print("  Audit RAG: OFF")

    llm = get_llm()
    results = []
    flagged = []

    try:
        for pair in pairs:
            chunk_num = pair["chunk_num"]
            start = time.time()
            print(f"  Auditing chunk {chunk_num}/{total_chunks}...", end=" ", flush=True)

            try:
                rag_context = ""
                if rag_store is not None:
                    rag_context = _trim_for_prompt(rag_store.retrieve_context(pair["source"], k=audit_rag_k), audit_rag_max_chars)
                audit = audit_chunk(llm, pair["source"], pair["translation"], rag_context=rag_context)
                elapsed = time.time() - start
                status = "PASS" if audit["verdict"] == "PASS" else "REWRITE"
                issues_str = f" | issues: {', '.join(audit['issues'])}" if audit["issues"] else ""
                _print_safe(f"{status} (score: {audit['score']}/10, {elapsed:.0f}s){issues_str}")

                audit["chunk_num"] = chunk_num
                audit["chunk_file"] = pair["chunk_file"]
                audit["source_chunk_file"] = pair.get("source_chunk_file", "")
                results.append(audit)

                if audit["verdict"] == "REWRITE":
                    flagged.append(pair)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {exc}")
                results.append({"chunk_num": chunk_num, "error": str(exc), "verdict": "ERROR"})
    finally:
        if rag_store is not None:
            rag_store.clear()

    if args.rewrite and flagged:
        print("\n" + "-" * 50)
        print(f"  Rewriting {len(flagged)} flagged chunks...")
        print("-" * 50 + "\n")

        for pair in flagged:
            chunk_num = pair["chunk_num"]
            audit = next(item for item in results if item.get("chunk_num") == chunk_num)

            print(f"  Rewriting chunk {chunk_num}...", end=" ", flush=True)
            start = time.time()

            try:
                new_translation = rewrite_chunk(llm, pair["source"], audit.get("issues", []))
                elapsed = time.time() - start

                backup_path = pair["chunk_file"].replace(".md", "_original.md")
                os.replace(pair["chunk_file"], backup_path)

                with open(pair["chunk_file"], "w", encoding="utf-8") as file:
                    file.write(f"<!-- Chunk {chunk_num} | REWRITTEN by audit | {elapsed:.0f}s -->\n\n")
                    file.write(new_translation)

                print(f"done ({elapsed:.0f}s)")
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {exc}")

    passed = sum(1 for item in results if item.get("verdict") == "PASS")
    rewrite = sum(1 for item in results if item.get("verdict") == "REWRITE")
    errors = sum(1 for item in results if item.get("verdict") == "ERROR")
    needs_human_attention = sum(1 for item in results if item.get("needs_human_attention"))
    avg_score = sum(item.get("score", 0) for item in results if "score" in item) / max(len(results), 1)

    print("\n" + "=" * 50)
    print("  AUDIT SUMMARY")
    print(f"  Passed: {passed} | Flagged: {rewrite} | Errors: {errors}")
    print(f"  Needs human attention: {needs_human_attention}")
    print(f"  Average score: {avg_score:.1f}/10")
    if flagged and not args.rewrite:
        nums = [item["chunk_num"] for item in flagged]
        print("  To rewrite flagged: add --rewrite flag")
        print(f"  Flagged chunks: {nums}")
    print("=" * 50 + "\n")

    audit_dir = os.path.join(os.path.dirname(args.chunks_dir), "audits")
    os.makedirs(audit_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(audit_dir, f"audit_{timestamp}.json")

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_audited": len(results),
                "passed": passed,
                "flagged": rewrite,
                "needs_human_attention": needs_human_attention,
                "avg_score": round(avg_score, 1),
                "results": [{k: v for k, v in item.items() if k != "raw"} for item in results],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    pass_score = int(config.get("audit", {}).get("pass_score", DEFAULT_AUDIT_PASS_SCORE))
    pass_score = max(1, min(10, pass_score))
    rewrite_report = _build_rewrite_chunk_report(results, pass_score)
    rewrite_report_path = os.path.join(audit_dir, f"audit_rewrite_seed_{timestamp}.json")
    with open(rewrite_report_path, "w", encoding="utf-8") as file:
        json.dump(rewrite_report, file, ensure_ascii=False, indent=2)

    print(f"  Report saved: {report_path}")
    print(f"  Rewrite chunk report: {rewrite_report_path}")


_ORIGINAL_APPLY_AUDIT_GUARDRAILS = _apply_audit_guardrails


def _apply_audit_guardrails(parsed: dict, source_for_audit: str, translation_for_audit: str) -> dict:
    """
    Phase 3 hardening:
    - Hallucination penalties require explicit evidence in issues.
    - Weak evidence cases are routed to human attention instead of hard punishment.
    """
    guarded = _ORIGINAL_APPLY_AUDIT_GUARDRAILS(parsed, source_for_audit, translation_for_audit)

    loader = globals().get("load_config")
    cfg = loader().get("audit", {}) if callable(loader) else {}
    require_evidence = bool(cfg.get("require_evidence_for_hallucination", True))
    weak_to_human = bool(cfg.get("weak_evidence_mark_human_attention", True))

    if not require_evidence:
        return guarded

    issues = guarded.get("issues") or []
    issues_text = " | ".join(str(x) for x in issues)
    has_explicit_evidence = any(
        token in issues_text
        for token in ["BEGIN", "MIDDLE", "END", "“", "\"", "原文", "译文", "片段", "evidence", "quote"]
    )

    if bool(guarded.get("hallucination")) and not has_explicit_evidence:
        guarded["hallucination"] = False
        if weak_to_human:
            guarded["needs_human_attention"] = True
            note = "AUDIT_WEAK_EVIDENCE: hallucination claim downgraded; requires human check"
            if note not in issues:
                issues.append(note)
            guarded["issues"] = issues
            # avoid harsh score drop on unsupported claim
            try:
                guarded["score"] = max(int(guarded.get("score", 0) or 0), 6)
            except Exception:
                pass
    return guarded


if __name__ == "__main__":
    main()
