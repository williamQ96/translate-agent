import os
import subprocess
import sys
from src.utils.config_loader import load_config


class PDFProcessor:
    def __init__(self):
        self.config = load_config()
        self.output_dir = self.config['directories']['output']

    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF file using MinerU (magic-pdf) CLI or combine pre-processed directories.
        Returns the path to the generated Markdown file.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Case 1: Source is a directory (MinerU pre-processed pages)
        if os.path.isdir(pdf_path):
            return self.combine_processed_dir(pdf_path)

        # Case 2: Source is a PDF file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ocr_output_dir = os.path.join(self.output_dir, "ocr", base_name)
        
        # MinerU outputs to: <output_dir>/<base_name>/auto/<base_name>.md
        expected_md = os.path.join(ocr_output_dir, "auto", f"{base_name}.md")

        # Skip if already processed
        if os.path.exists(expected_md):
            print(f"  ⏭ OCR already done, using cached: {expected_md}")
            return expected_md

        os.makedirs(ocr_output_dir, exist_ok=True)

        print(f"  📄 Running MinerU OCR on: {pdf_path}")
        print(f"     Output dir: {ocr_output_dir}")

        try:
            # Set environment variable for magic-pdf config
            env = os.environ.copy()
            env['MINERU_TOOLS_CONFIG_JSON'] = os.path.join(self.config['project']['root_dir'], 'magic-pdf.json')

            result = subprocess.run(
                [sys.executable, "-m", "magic_pdf.tools.cli", 
                 "-p", pdf_path, "-o", ocr_output_dir, "-m", "auto"],
                capture_output=True, text=True, timeout=3600, env=env  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"  ⚠ MinerU failed: {result.stderr}")
                # Check for common configuration error
                if "son not found" in result.stderr or "models.json" in result.stderr:
                     print("  ⚠ Hint: detailed configuration might be missing in magic-pdf.json.")
                raise RuntimeError(f"MinerU OCR failed with exit code {result.returncode}")
            
        except FileNotFoundError:
            print("  ⚠ MinerU (magic-pdf) not found. Please install it or provide a pre-processed Markdown file.")
            raise

        # Find the output markdown (MinerU uses various output structures)
        if os.path.exists(expected_md):
            return expected_md
        
        # Search for any .md file in output
        for root, dirs, files in os.walk(ocr_output_dir):
            for f in files:
                if f.endswith(".md"):
                    return os.path.join(root, f)
        
        raise FileNotFoundError(f"MinerU produced no Markdown output in {ocr_output_dir}")
        
    def combine_processed_dir(self, dir_path: str) -> str:
        """Collate multiple pre-processed MinerU folders into one Markdown file."""
        print(f"  📂 Collating pre-processed MinerU results from: {dir_path}")
        
        # 1. Gather and sort directories by page number
        subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        
        def get_page_num(name):
            try:
                # Extract number from "10.png-uuid" or similar
                return int(name.split('.')[0])
            except (ValueError, IndexError):
                return 999999
        
        subdirs.sort(key=get_page_num)
        
        combined_content = []
        for subdir in subdirs:
            md_file = os.path.join(dir_path, subdir, "full.md")
            if os.path.exists(md_file):
                with open(md_file, "r", encoding="utf-8") as f:
                    combined_content.append(f.read())
            else:
                print(f"  ⚠ Warning: full.md not found in {subdir}")

        # 2. Save combined result
        base_name = os.path.basename(dir_path.rstrip("/\\"))
        output_md = os.path.join(self.output_dir, "ocr", f"{base_name}_combined.md")
        os.makedirs(os.path.dirname(output_md), exist_ok=True)
        
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_content))
            
        print(f"  ✅ Collated {len(combined_content)} pages into: {output_md}")
        return output_md
