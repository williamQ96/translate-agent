"""
Convenience entrypoint for iterative rewrite/audit loop.

Usage:
    python run_rewrite_audit_loop.py --source-chunks-dir data/output/source_chunks --chunks-dir data/output/chunks
"""

from src.rewrite_audit_loop import main


if __name__ == "__main__":
    main()
