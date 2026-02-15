from typing import TypedDict, Optional


class TranslationState(TypedDict):
    """
    Represents the state of the translation process for a single chunk.
    """
    chunk_id: int
    source_text: str
    glossary: str
    rag_context: str   # Cross-reference context from RAG
    polish_style: str  # Optional style injected only in polish/review stage
    
    # Agent outputs
    draft_translation: Optional[str]
    final_translation: Optional[str]
    
    # Metrics
    iteration_count: int
