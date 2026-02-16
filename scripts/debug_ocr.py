
import os
import subprocess
import sys
import time

def debug_ocr():
    # 1. Config path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    config_path = os.path.join(root_dir, "config", "magic-pdf.json")
    
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}")
        return

    # 2. Env setup
    env = os.environ.copy()
    env['MINERU_TOOLS_CONFIG_JSON'] = config_path
    
    print(f"DEBUG: sys.executable = {sys.executable}")
    try:
        import rapid_table
        print(f"DEBUG: rapid_table found at {rapid_table.__file__}")
    except ImportError as e:
        print(f"DEBUG: rapid_table import failed: {e}")

    # 3. Command
    pdf_path = os.path.join(root_dir, "data", "input", "opera in histroy.pdf")
    output_dir = os.path.join(root_dir, "data", "output", "debug_ocr")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try using magic-pdf.exe directly which is in the same folder as python.exe
    exe_dir = os.path.dirname(sys.executable)
    magic_pdf_exe = os.path.join(exe_dir, "magic-pdf.exe")
    
    if os.path.exists(magic_pdf_exe):
        cmd = [
            magic_pdf_exe,
            "-p", pdf_path,
            "-o", output_dir,
            "-m", "auto",
            "-d", "True" # Enable debug
        ]
    else:
        # Fallback
        cmd = [
            sys.executable, "-m", "magic_pdf.tools.cli",
            "-p", pdf_path,
            "-o", output_dir,
            "-m", "auto",
            "-d", "True"
        ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Config path: {config_path}")
    print("-" * 50)
    
    start = time.time()
    print("-" * 50)
    
    start = time.time()
    result = subprocess.run(
        cmd,
        env=env
    )
    print(f"Exit Code: {result.returncode}")
        
    print("Log written to debug_ocr.log")

if __name__ == "__main__":
    debug_ocr()
