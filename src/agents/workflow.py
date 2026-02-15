from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import TranslationState
from src.agents.prompts import TRANSLATOR_PROMPT, REVIEWER_PROMPT
from src.utils.config_loader import load_config


def _get_llm():
    """Create LLM client pointing to Ollama's OpenAI-compatible API."""
    config = load_config()
    model_config = config.get("model", {})
    request_timeout = float(model_config.get("translate_timeout", model_config.get("request_timeout", 180)))
    translate_max_tokens = min(int(model_config.get("translate_max_tokens", model_config.get("max_tokens", 4096))), 4096)
    return ChatOpenAI(
        model=model_config.get("name", "qwen3:30b"),
        base_url=model_config.get("api_base", "http://localhost:11434/v1"),
        api_key=model_config.get("api_key", "ollama"),
        temperature=model_config.get("temperature", 0.3),
        max_tokens=translate_max_tokens,
        timeout=request_timeout,
        max_retries=0,
    )


llm = _get_llm()


def translate_node(state: TranslationState):
    """Agent 1: Literal Translator"""
    chain = TRANSLATOR_PROMPT | llm | StrOutputParser()
    draft = chain.invoke({
        "source_text": state["source_text"],
        "glossary": state.get("glossary", ""),
        "rag_context": state.get("rag_context", ""),
    })
    return {"draft_translation": draft}


def review_node(state: TranslationState):
    """Agent 2: Combined Reviewer (critic + polisher in one pass)"""
    chain = REVIEWER_PROMPT | llm | StrOutputParser()
    final = chain.invoke({
        "source_text": state["source_text"],
        "translation": state["draft_translation"],
        "glossary": state.get("glossary", ""),
        "rag_context": state.get("rag_context", ""),
        "polish_style": state.get("polish_style", "NONE"),
    })
    return {"final_translation": final}


def build_graph():
    """Build and compile the 2-step translation workflow graph."""
    workflow = StateGraph(TranslationState)

    workflow.add_node("translator", translate_node)
    workflow.add_node("reviewer", review_node)

    workflow.set_entry_point("translator")
    workflow.add_edge("translator", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow.compile()


if __name__ == "__main__":
    try:
        app = build_graph()
        print("Graph built successfully (2-step: translator → reviewer).")
    except Exception as e:
        print(f"Graph build failed: {e}")
