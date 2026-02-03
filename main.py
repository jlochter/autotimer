import argparse
import os
import sys
from generate_whisper import generate_whisper_script
from extract_jscript import extract_jscript
from align_scripts import align_scripts
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Japanese Video Script Alignment Tool")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--script", required=True, help="Path to input PDF script")
    parser.add_argument("--output", default="aligned.ass", help="Path to output ASS file")
    parser.add_argument("--api_key", help="Gemini API Key (optional if GEMINI_API_KEY env var is set)")
    parser.add_argument("--whisper_model", default="large-v3", help="Whisper model size (default: large-v3)")
    
    args = parser.parse_args()

    # Derived paths for intermediate files
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    whisper_json_path = os.path.join(output_dir, f"{base_name}_whisper.json")
    script_text_path = os.path.join(output_dir, f"{base_name}_script.txt")

    print("=== Step 1: Generating Whisper Transcription ===")
    if os.path.exists(whisper_json_path):
         print(f"File {whisper_json_path} already exists, skipping transcription (delete file to re-run).")
    else:
        generate_whisper_script(args.video, whisper_json_path, model_size=args.whisper_model)

    print("\n=== Step 2: Extracting Japanese Text from PDF ===")
    if os.path.exists(script_text_path):
        print(f"File {script_text_path} already exists, skipping OCR (delete file to re-run).")
    else:
        extract_jscript(args.script, script_text_path)

    print("\n=== Step 3: Aligning Scripts ===")
    align_scripts(whisper_json_path, script_text_path, args.output, api_key=args.api_key)

    print(f"\nSUCCESS: Output saved to {args.output}")

if __name__ == "__main__":
    main()
