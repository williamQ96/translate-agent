"""
Convenience entrypoint for iterative rewrite/audit loop.

Usage:
    python scripts/run_rewrite_audit_loop.py --source-chunks-dir data/output/source_chunks --chunks-dir data/output/chunks
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rewrite_audit_loop import main


if __name__ == "__main__":
    main()
