from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.utils.config_loader import load_config


class TermExtractor:
    def __init__(self):
        config = load_config()
        model_config = config.get("model", {})
        self.llm = ChatOpenAI(
            model=model_config.get("name", "qwen3:30b"),
            base_url=model_config.get("api_base", "http://localhost:11434/v1"),
            api_key=model_config.get("api_key", "ollama"),
            temperature=0.1,
        )

    def extract_terms(self, text: str):
        """Extract proper nouns and technical terms from text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a terminology extraction expert.
Analyze the following text and extract:
1. People's names
2. Location names
3. Organization names
4. Technical terms specific to the domain

Output purely as a JSON list of objects: {{"term": "...", "category": "...", "suggested_translation": "..."}}
Do not output markdown code blocks, just raw JSON."""),
            ("user", "{text}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            return chain.invoke({"text": text})
        except Exception as e:
            print(f"    ⚠ Extraction failed for chunk: {e}")
            return []

    def extract_from_full_document(self, text: str, chunk_size: int = 3000) -> list:
        """
        Extract terms from the ENTIRE document by scanning it in chunks.
        This ensures terms appearing in chapter 15 are also captured.
        """
        all_terms = {}
        
        # Split document into overlapping sections for extraction
        words = text.split()
        section_words = int(chunk_size * 1.3)
        sections = []
        start = 0
        while start < len(words):
            end = start + section_words
            sections.append(" ".join(words[start:end]))
            start = end  # No overlap needed for extraction
        
        print(f"    📖 Scanning {len(sections)} sections for terminology...")
        
        for i, section in enumerate(sections):
            terms = self.extract_terms(section)
            if terms:
                for t in terms:
                    key = t.get("term", "").strip()
                    if key and key not in all_terms:
                        all_terms[key] = t
        
        result = list(all_terms.values())
        print(f"    📖 Total unique terms found: {len(result)}")
        return result


if __name__ == "__main__":
    extractor = TermExtractor()
    print("Extractor initialized (using Ollama).")
