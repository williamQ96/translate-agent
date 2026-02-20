"""
Writer agent for audit-driven rewrites.

Phase 0.31 updates:
- dual-model routing with loop-aware escalation
- per-route base_url/temperature/max_tokens for speed vs precision tuning
- compact fast-path prompts for smaller models
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.utils.config_loader import load_config


def _extract_text(output: Any) -> str:
    if hasattr(output, "content"):
        return str(output.content).strip()
    return str(output).strip()


def _clip(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit]


def _sanitize_rewrite_output(text: str) -> str:
    """
    Remove control/context leak markers that occasionally bleed into model output.
    These tokens are prompt scaffolding, not translation content.
    """
    if not text:
        return ""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        striped = line.strip()
        upper = striped.upper()
        if (
            upper.startswith("[RAG#")
            or upper.startswith("[WEB#")
            or upper.startswith("[SEGMENT_MODE]")
            or striped.startswith("来源:")
            or striped.startswith("Source:")
            or upper.startswith("SCORE:")
            or upper.startswith("VERDICT:")
            or upper.startswith("HALLUCINATION:")
            or upper.startswith("OMISSION:")
            or upper.startswith("MISTRANSLATION:")
            or upper.startswith("FORMAT_OK:")
            or "翻译质量审校报告" in striped
            or striped.startswith("以下是根据")
            or striped.startswith("以下为")
            or striped.startswith("修正后的翻译")
            or striped.startswith("审校报告")
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


class WriterAgent:
    """Writer agent used for audit-driven chunk rewrites."""

    def __init__(self):
        self.config = load_config()
        model_cfg = self.config.get("model", {})
        router_cfg = self.config.get("model_router", {})

        self.api_base = str(model_cfg.get("api_base", "http://localhost:11434/v1"))
        self.api_key = str(model_cfg.get("api_key", "ollama"))
        self.max_tokens = int(model_cfg.get("max_tokens", 4096))
        self.temperature = float(model_cfg.get("temperature", 0.2))

        # Router settings
        self.default_model = str(router_cfg.get("default_model", model_cfg.get("name", "qwen3:8b")))
        self.escalation_model = str(router_cfg.get("escalation_model", model_cfg.get("name", "qwen3:30b")))
        self.default_api_base = str(router_cfg.get("default_api_base", self.api_base))
        self.escalation_api_base = str(router_cfg.get("escalation_api_base", self.api_base))
        self.default_temperature = float(router_cfg.get("default_temperature", self.temperature))
        self.escalation_temperature = float(router_cfg.get("escalation_temperature", min(self.temperature, 0.1)))
        self.default_max_tokens = int(router_cfg.get("default_max_tokens", min(self.max_tokens, 2048)))
        self.escalation_max_tokens = int(router_cfg.get("escalation_max_tokens", self.max_tokens))

        self.escalate_on_human_attention = bool(router_cfg.get("escalate_on_human_attention", True))
        self.escalate_on_persistent_critical_flags = bool(
            router_cfg.get("escalate_on_persistent_critical_flags", True)
        )
        self.escalate_below_score = int(router_cfg.get("escalate_below_score", 5))
        self.escalate_stagnated_below_score = int(router_cfg.get("escalate_stagnated_below_score", 9))
        self.escalation_after_loops = int(router_cfg.get("escalation_after_loops", 2))
        self.require_stagnation_for_escalation = bool(router_cfg.get("require_stagnation_for_escalation", True))
        self.stagnation_threshold_loops = int(router_cfg.get("stagnation_threshold_loops", 3))
        self.request_timeout_default = float(model_cfg.get("rewrite_timeout", model_cfg.get("request_timeout", 180)))
        self.request_timeout_escalation = float(
            router_cfg.get("escalation_timeout", max(90, self.request_timeout_default))
        )
        self.fallback_on_timeout = bool(router_cfg.get("fallback_on_timeout", True))

        self._clients: dict[tuple[str, str, float, float, int], ChatOpenAI] = {}
        self.last_route: dict[str, Any] = {}

    def _client(
        self,
        api_base: str,
        model_name: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
    ) -> ChatOpenAI:
        key = (str(api_base), str(model_name), float(timeout), float(temperature), int(max_tokens))
        if key not in self._clients:
            self._clients[key] = ChatOpenAI(
                base_url=api_base,
                api_key=self.api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        return self._clients[key]

    def _should_escalate(
        self,
        audit_score: int | None,
        audit_issues: list[str] | None,
        loop_index: int | None = None,
        stagnation_rounds: int = 0,
    ) -> bool:
        if loop_index is not None and loop_index < self.escalation_after_loops:
            return False
        if self.require_stagnation_for_escalation and stagnation_rounds < self.stagnation_threshold_loops:
            return False
        # Convergence mode: for stubborn chunks that keep stalling, escalate even when score is
        # not very low (for example repeated 7-8/10 with persistent critical flags).
        if (
            audit_score is not None
            and stagnation_rounds >= self.stagnation_threshold_loops
            and audit_score < self.escalate_stagnated_below_score
        ):
            return True
        if audit_score is not None and audit_score <= self.escalate_below_score:
            return True

        issues = [str(x).upper() for x in (audit_issues or [])]
        has_hallucination = any("HALLUCINATION" in x for x in issues)
        has_mistranslation = any("MISTRANSLATION" in x for x in issues)
        has_format = any("FORMAT" in x for x in issues)
        has_human_attention = any("HUMAN_ATTENTION" in x for x in issues)
        has_omission_only = any("OMISSION" in x for x in issues) and not (
            has_hallucination or has_mistranslation or has_format or has_human_attention
        )

        if self.escalate_on_human_attention and any("HUMAN_ATTENTION" in x for x in issues):
            return True
        # Omission-only signals are common in sampled audits; do not escalate solely on omission.
        if self.escalate_on_persistent_critical_flags and (has_hallucination or has_mistranslation or has_format):
            return True
        if has_omission_only and audit_score is not None and audit_score <= 6:
            return True
        return False

    def _build_prompt(
        self,
        source_text: str,
        previous_translation: str,
        audit_issues: list[str],
        audit_score: int | None,
        glossary_text: str,
        rag_context: str,
        strict_mode: bool = False,
    ) -> str:
        issues_text = "\n".join(f"- {item}" for item in audit_issues) if audit_issues else "- none"
        score_text = "N/A" if audit_score is None else str(audit_score)
        strict_block = (
            "\nStrict mode:\n"
            "5) Align sentence-by-sentence to source.\n"
            "6) If uncertain, choose conservative literal wording.\n"
            "7) Do not add any explanation not present in source.\n"
            if strict_mode
            else ""
        )

        return (
            "/no_think\n"
            "You are a high-reliability translation rewriter.\n"
            "Task: fix the Chinese translation strictly based on source text and audit feedback.\n"
            "Never invent facts. Never omit key information.\n\n"
            "Rules:\n"
            "1) Faithful to source meaning.\n"
            "2) Keep terms consistent with glossary.\n"
            "3) Keep fluent Chinese without style overreach.\n"
            "4) Output only final Chinese translation, no explanation."
            f"{strict_block}\n\n"
            f"Audit score: {score_text}/10\n"
            f"Audit issues:\n{issues_text}\n\n"
            f"Glossary (relevant):\n{glossary_text or '[NONE]'}\n\n"
            f"Context (RAG):\n{rag_context or '[NONE]'}\n\n"
            f"Source:\n{source_text}\n\n"
            f"Current translation to fix:\n{previous_translation or '[NONE]'}\n"
        )

    def rewrite(
        self,
        source_text: str,
        previous_translation: str = "",
        current_translation: str = "",
        issues: list[str] | None = None,
        issue_tags: list[str] | None = None,
        score: int | None = None,
        loop_index: int | None = None,
        audit_issues: list[str] | None = None,
        audit_score: int | None = None,
        glossary_text: str = "",
        rag_context: str = "",
        stagnation_rounds: int = 0,
    ) -> str:
        """Rewrite a single chunk using source text plus structured audit signals."""
        if (not previous_translation.strip()) and current_translation.strip():
            previous_translation = current_translation
        if audit_issues is None and issues is not None:
            audit_issues = issues
        if audit_score is None and score is not None:
            audit_score = score
        if issue_tags:
            merged = list(audit_issues or [])
            merged.extend([f"TAG:{str(tag).strip()}" for tag in issue_tags if str(tag).strip()])
            audit_issues = merged

        issue_list = [str(x).strip() for x in (audit_issues or []) if str(x).strip()]
        escalate = self._should_escalate(
            audit_score,
            issue_list,
            loop_index=loop_index,
            stagnation_rounds=stagnation_rounds,
        )

        model_name = self.escalation_model if escalate else self.default_model
        api_base = self.escalation_api_base if escalate else self.default_api_base
        timeout = self.request_timeout_escalation if escalate else self.request_timeout_default
        temperature = self.escalation_temperature if escalate else self.default_temperature
        max_tokens = self.escalation_max_tokens if escalate else self.default_max_tokens

        # Fast path prompt compaction helps 8b latency and stability.
        prompt_glossary = glossary_text
        prompt_rag = rag_context
        prompt_issues = issue_list
        if not escalate:
            prompt_glossary = _clip(glossary_text, 900)
            prompt_rag = _clip(rag_context, 1200)
            prompt_issues = issue_list[:2]
        else:
            prompt_glossary = _clip(glossary_text, 1800)
            prompt_rag = _clip(rag_context, 2800)
            prompt_issues = issue_list[:4]

        prompt = self._build_prompt(
            source_text=source_text,
            previous_translation=previous_translation,
            audit_issues=prompt_issues,
            audit_score=audit_score,
            glossary_text=prompt_glossary,
            rag_context=prompt_rag,
            strict_mode=escalate,
        )

        self.last_route = {
            "escalated": escalate,
            "model": model_name,
            "api_base": api_base,
            "timeout": timeout,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stagnation_rounds": int(stagnation_rounds),
            "stagnation_threshold": int(self.stagnation_threshold_loops),
        }

        try:
            output = self._client(api_base, model_name, timeout, temperature, max_tokens).invoke(prompt)
            text = _extract_text(output)
            if text:
                return _sanitize_rewrite_output(text)
        except Exception:
            if not (escalate and self.fallback_on_timeout):
                raise

        fallback_output = self._client(
            self.default_api_base,
            self.default_model,
            self.request_timeout_default,
            self.default_temperature,
            self.default_max_tokens,
        ).invoke(prompt)
        return _sanitize_rewrite_output(_extract_text(fallback_output))
