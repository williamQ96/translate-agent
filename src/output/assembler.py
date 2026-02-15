def assemble_chapters(chapter_files: list, output_path: str):
    """
    Combine multiple markdown files into one.
    """
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for fname in chapter_files:
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

def convert_to_pdf(markdown_path: str, pdf_path: str):
    """
    Convert markdown to PDF using pandoc.
    """
    try:
        import pypandoc
        # Ensure pandoc is installed
        pypandoc.convert_file(markdown_path, 'pdf', outputfile=pdf_path, extra_args=['--pdf-engine=xelatex', '-V', 'mainfont=Microsoft YaHei'])
        print(f"Generated PDF at: {pdf_path}")
    except ImportError:
        print("pypandoc not installed.")
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    pass
