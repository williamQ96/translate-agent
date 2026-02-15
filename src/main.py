import argparse
import os
from src.agents.workflow import build_graph
from src.utils.config_loader import load_config

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
    """
    Split text into overlapping chunks by approximate token count.
    Uses simple word-based splitting (~1.3 words per token estimate).
    """
    words = text.split()
    words_per_chunk = int(chunk_size * 1.3)
    words_overlap = int(overlap * 1.3)
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - words_overlap
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Translate Agent - PDF to Publication-Quality Translation")
    parser.add_argument("--source", "-s", required=True, help="Path to source Markdown or text file")
    parser.add_argument("--glossary", "-g", default="", help="Glossary text or path to glossary file")
    parser.add_argument(
        "--style",
        default="",
        help="Optional polish style applied in review stage only",
    )
    parser.add_argument(
        "--no-style-prompt",
        action="store_true",
        help="Do not ask style when --style is empty",
    )
    args = parser.parse_args()

    # Load Config
    config = load_config()
    print(f"Project: {config['project']['name']}")
    
    # Read source file
    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return
    
    with open(args.source, "r", encoding="utf-8") as f:
        source_text = f.read()
    
    print(f"Loaded source: {args.source} ({len(source_text)} chars)")
    
    # Load glossary
    glossary = ""
    if args.glossary and os.path.exists(args.glossary):
        with open(args.glossary, "r", encoding="utf-8") as f:
            glossary = f.read()
    
    polish_style = args.style.strip()
    if not polish_style and not args.no_style_prompt:
        print("\nPolish style (optional, review stage only).")
        print("Example: 能看懂，保持原作风格，中文本土化")
        print("Press Enter for neutral polish style.")
        try:
            polish_style = input("Style> ").strip()
        except EOFError:
            polish_style = ""

    # Chunk the text
    chunk_config = config.get("chunking", {})
    chunks = chunk_text(
        source_text, 
        chunk_size=chunk_config.get("chunk_size", 2000),
        overlap=chunk_config.get("overlap", 200)
    )
    print(f"Split into {len(chunks)} chunks")
    
    # Build Workflow
    app = build_graph()
    print("Translation pipeline ready. Starting...\n")
    
    # Process each chunk
    results = []
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1}/{len(chunks)} ---")
        
        state = {
            "chunk_id": i,
            "source_text": chunk,
            "glossary": glossary,
            "rag_context": "",
            "polish_style": polish_style,
            "draft_translation": None,
            "final_translation": None,
            "iteration_count": 0,
        }
        
        result = app.invoke(state)
        results.append(result["final_translation"])
        print(f"  ✓ Chunk {i+1} done.\n")
    
    # Assemble output
    output_dir = config["directories"]["output"]
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.source))[0]
    output_path = os.path.join(output_dir, f"{base_name}_translated.md")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results))
    
    print("\n=== Translation Complete ===")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()
