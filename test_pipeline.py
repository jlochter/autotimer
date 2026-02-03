import os
import subprocess
import sys
import shutil

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    print("SUCCESS")
    return result.stdout

def main():
    print("=== STARTING PIPELINE TEST ===")
    
    # Input files
    video_input = "test.mp4"
    pdf_input = "test.pdf"
    
    if not os.path.exists(video_input) or not os.path.exists(pdf_input):
        print(f"Error: {video_input} or {pdf_input} not found.")
        sys.exit(1)

    # Output files (temporary)
    video_short = os.path.join("output", "test_short.mp4")
    whisper_out = os.path.join("output", "test_whisper.json")
    ocr_out = os.path.join("output", "test_ocr.txt")
    final_ass = os.path.join("output", "test_final.ass")

    # Clean up previous run
    for f in [video_short, whisper_out, ocr_out, final_ass]:
        if os.path.exists(f):
            os.remove(f)

    # Step 0: Cut video to 4 minutes
    print("\n[Step 0] Cutting video to first 4 minutes...")
    # -t 240 seconds = 4 minutes
    # -c copy is fast and preserves streams
    cmd_cut = f"ffmpeg -i {video_input} -t 240 -c copy -y {video_short}"
    run_command(cmd_cut)

    # Step 1: Whisper
    print("\n[Step 1] Running Whisper Transcription...")
    # Using 'tiny' model for speed in test, unless user really wants quality. 
    # Use 'turbo' or 'small' for a balance. Let's use 'tiny' for speed if just testing pipeline.
    # Actually, user said "Process...". Let's use the default or a faster one. 
    # generate_whisper.py defaults to large-v3. That's slow on CPU.
    # I'll pass --model tiny for this test to be quick, or turbo.
    cmd_whisper = f"python generate_whisper.py --video {video_short} --output {whisper_out} --model tiny"
    run_command(cmd_whisper)
    
    if not os.path.exists(whisper_out):
        print("Error: Whisper output not found.")
        sys.exit(1)

    # Step 2: OCR
    print("\n[Step 2] Running OCR Extraction (First 2 pages)...")
    cmd_ocr = f"python extract_jscript.py --pdf {pdf_input} --output {ocr_out} --limit_pages 2"
    run_command(cmd_ocr)
    
    if not os.path.exists(ocr_out):
        print("Error: OCR output not found.")
        sys.exit(1)
        
    print(f"OCR Content Preview (first 200 chars):\n{'-'*20}")
    with open(ocr_out, 'r', encoding='utf-8') as f:
        print(f.read()[:200])
    print(f"{'-'*20}")

    # Step 3: Alignment
    print("\n[Step 3] Running Alignment...")
    # check for api key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY") or "your_api_key" in os.environ.get("GEMINI_API_KEY", ""):
        print("WARNING: Valid GEMINI_API_KEY not found. Alignment step might fail.")
    
    cmd_align = f"python align_scripts.py --whisper {whisper_out} --ocr {ocr_out} --output {final_ass}"
    # We allow this to fail without exiting the test script hard, just to show the result
    try:
        run_command(cmd_align)
        print(f"\nSUCCESS: Generated {final_ass}")
        print("First 10 lines of ASS file:")
        with open(final_ass, 'r', encoding='utf-8') as f:
            for _ in range(10):
                print(f.readline().strip())
    except Exception:
        print("Alignment failed (likely due to API Key). Pipeline partially successful.")

if __name__ == "__main__":
    main()
