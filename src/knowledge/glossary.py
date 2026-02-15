import json
import os
from typing import List, Dict


class GlossaryManager:
    def __init__(self, glossary_path="data/glossary.json"):
        self.glossary_path = glossary_path
        self.terms = {}
        self._load_glossary()

    def _load_glossary(self):
        if os.path.exists(self.glossary_path):
            with open(self.glossary_path, 'r', encoding='utf-8') as f:
                self.terms = json.load(f)

    def save_glossary(self):
        os.makedirs(os.path.dirname(self.glossary_path) or ".", exist_ok=True)
        with open(self.glossary_path, 'w', encoding='utf-8') as f:
            json.dump(self.terms, f, ensure_ascii=False, indent=2)

    def add_terms(self, new_terms: List[Dict]):
        """
        Add new terms from extractor output.
        Format: [{"term": "Name", "category": "Person", "suggested_translation": "..."}]
        Only adds if the term doesn't already exist (preserves first translation for consistency).
        """
        added = 0
        for item in new_terms:
            key = item.get("term", "").strip()
            if key and key not in self.terms:
                translation = item.get("suggested_translation", item.get("translation", ""))
                self.terms[key] = translation
                added += 1
        if added:
            self.save_glossary()
        return added

    def update_from_translation(self, source_chunk: str, translated_chunk: str):
        """
        Translation Memory: placeholder for future glossary learning.
        Previously this flagged terms as [NEEDS REVIEW], which polluted the glossary.
        Disabled for now — the initial extraction already provides translations.
        """
        pass

    def get_glossary_text(self) -> str:
        """
        Return a string representation of the FULL glossary for LLM context.
        Format: "English Term: 中文翻译" per line.
        """
        lines = []
        for term, translation in sorted(self.terms.items()):
            if isinstance(translation, str) and translation and not translation.startswith("["):
                lines.append(f"{term}: {translation}")
        return "\n".join(lines)

    def get_relevant_glossary_text(self, source_text: str) -> str:
        """
        Return only glossary terms that appear in the given source text.
        This dramatically reduces prompt size (e.g. 1048 terms → ~20-50 relevant).
        """
        source_lower = source_text.lower()
        lines = []
        for term, translation in sorted(self.terms.items()):
            if isinstance(translation, str) and translation and not translation.startswith("["):
                if term.lower() in source_lower:
                    lines.append(f"{term}: {translation}")
        return "\n".join(lines)


    def get_term_count(self) -> int:
        """Return number of terms with valid translations."""
        return sum(1 for v in self.terms.values() 
                   if isinstance(v, str) and v and not v.startswith("["))

    def clear(self):
        """Clear all terms."""
        self.terms = {}
        self.save_glossary()


if __name__ == "__main__":
    gm = GlossaryManager()
    print(f"Glossary: {gm.get_term_count()} terms")
    print(gm.get_glossary_text())
