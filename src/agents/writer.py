"""
Writer agent for audit-driven rewrites.

Phase 2.5 dual-model routing:
- default fast model for normal chunks
- escalation model for persistently hard chunks
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.utils.config_loader import load_config


def _extract_text(output: Any) -> str:
    if hasattr(output, "content"):
        return str(output.content).strip()
    return str(output).strip()


class WriterAgent:
    """Writer agent used for audit-driven chunk rewrites."""

    def __init__(self):
        self.config = load_config()
        model_cfg = self.config.get("model", {})
        router_cfg = self.config.get("model_router", {})

        self.api_base = model_cfg.get("api_base", "http://localhost:11434/v1")
        self.api_key = model_cfg.get("api_key", "ollama")
        self.max_tokens = int(model_cfg.get("max_tokens", 4096))
        self.temperature = float(model_cfg.get("temperature", 0.2))

        # Router settings
        self.default_model = router_cfg.get("default_model", model_cfg.get("name", "qwen3:8b"))
        self.escalation_model = router_cfg.get("escalation_model", model_cfg.get("name", "qwen3:30b"))
        self.escalate_on_human_attention = bool(router_cfg.get("escalate_on_human_attention", True))
        self.escalate_on_persistent_critical_flags = bool(
            router_cfg.get("escalate_on_persistent_critical_flags", True)
        )
        self.escalate_below_score = int(router_cfg.get("escalate_below_score", 5))
        self.request_timeout_default = float(model_cfg.get("rewrite_timeout", model_cfg.get("request_timeout", 180)))
        self.request_timeout_escalation = float(router_cfg.get("escalation_timeout", max(90, self.request_timeout_default)))
        self.fallback_on_timeout = bool(router_cfg.get("fallback_on_timeout", True))

        self._clients: dict[tuple[str, float], ChatOpenAI] = {}
        self.last_route: dict[str, Any] = {}

    def _client(self, model_name: str, timeout: float) -> ChatOpenAI:
        key = (str(model_name), float(timeout))
        if key not in self._clients:
            self._clients[key] = ChatOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout,
            )
        return self._clients[key]

    def _should_escalate(self, audit_score: int | None, audit_issues: list[str] | None) -> bool:
        if audit_score is not None and audit_score <= self.escalate_below_score:
            return True
        issues = [str(x).upper() for x in (audit_issues or [])]
        if self.escalate_on_human_attention and any("HUMAN_ATTENTION" in x for x in issues):
            return True
        if self.escalate_on_persistent_critical_flags and any(
            x.startswith("HALLUCINATION") or x.startswith("OMISSION") or x.startswith("MISTRANSLATION")
            for x in issues
        ):
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
    ) -> str:
        issues_text = "\n".join(f"- {item}" for item in audit_issues) if audit_issues else "- （无）"
        score_text = "N/A" if audit_score is None else str(audit_score)

        return (
            "/no_think\n"
            "你是高可靠翻译重写器。任务：只基于原文修正译文问题，不新增事实，不删减关键信息。\n\n"
            "要求：\n"
            "1) 忠实原文语义，不脑补。\n"
            "2) 术语统一，优先遵循术语表。\n"
            "3) 句子自然可读，但不要风格过度发挥。\n"
            "4) 仅输出最终中文译文，不要解释。\n\n"
            f"审计分数：{score_text}/10\n"
            f"审计问题：\n{issues_text}\n\n"
            f"术语表（相关项）：\n{glossary_text or '[NONE]'}\n\n"
            f"相关上下文（RAG）：\n{rag_context or '[NONE]'}\n\n"
            f"原文：\n{source_text}\n\n"
            f"当前译文（待修正）：\n{previous_translation or '[NONE]'}\n"
        )

    def rewrite(
        self,
        source_text: str,
        previous_translation: str = "",
        current_translation: str = "",
        issues: list[str] | None = None,
        issue_tags: list[str] | None = None,
        score: int | None = None,
        audit_issues: list[str] | None = None,
        audit_score: int | None = None,
        glossary_text: str = "",
        rag_context: str = "",
    ) -> str:
        """Rewrite a single chunk using source text plus structured audit signals."""
        # Backward compatibility: old call sites pass `current_translation`.
        if (not previous_translation.strip()) and current_translation.strip():
            previous_translation = current_translation
        # Backward compatibility: old call sites pass `issues` and `score`.
        if audit_issues is None and issues is not None:
            audit_issues = issues
        if audit_score is None and score is not None:
            audit_score = score
        if issue_tags:
            merged = list(audit_issues or [])
            merged.extend([f"TAG:{str(tag).strip()}" for tag in issue_tags if str(tag).strip()])
            audit_issues = merged

        issues = [str(x).strip() for x in (audit_issues or []) if str(x).strip()]
        escalate = self._should_escalate(audit_score, issues)

        model_name = self.escalation_model if escalate else self.default_model
        timeout = self.request_timeout_escalation if escalate else self.request_timeout_default
        prompt = self._build_prompt(
            source_text=source_text,
            previous_translation=previous_translation,
            audit_issues=issues,
            audit_score=audit_score,
            glossary_text=glossary_text,
            rag_context=rag_context,
        )

        self.last_route = {
            "escalated": escalate,
            "model": model_name,
            "timeout": timeout,
        }

        try:
            output = self._client(model_name, timeout).invoke(prompt)
            text = _extract_text(output)
            if text:
                return text
        except Exception:
            if not (escalate and self.fallback_on_timeout):
                raise

        # fallback path: escalation failed -> default model once
        fallback_output = self._client(self.default_model, self.request_timeout_default).invoke(prompt)
        return _extract_text(fallback_output)
