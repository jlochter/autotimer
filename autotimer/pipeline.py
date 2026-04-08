"""Main pipeline orchestrator for AutoTimer."""

import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from .generate_whisper import generate_whisper_script
from .extract_jscript import extract_jscript
from .align_scripts import align_scripts


def run(api_key, gdrive_path, chunk_length=30, translate=False):
    """
    Run the full AutoTimer pipeline.

    Finds video and script files in gdrive_path, transcribes the video
    with Whisper, extracts dialogue from the script with Gemini, aligns
    them, and exports an .ass subtitle file.

    Intermediate results (Whisper transcription and script extraction) are
    cached to disk and reused on subsequent runs if the files are found.

    Args:
        api_key: Google Gemini API key.
        gdrive_path: Path to directory containing video (.mp4/.mov) and script (.pdf/.docx/.doc) files.
        chunk_length: The length of the sliding window for transcription (default: 30).
        translate: If True, subtitles are formatted as "portuguese {japanese}" (default: False).
    """
    if not api_key:
        raise ValueError("api_key is required. Get one at https://aistudio.google.com/")

    if not os.path.isdir(gdrive_path):
        raise FileNotFoundError(f"Directory not found: {gdrive_path}")

    # ── Step 1: Find files ──────────────────────────────────────────────
    print("=" * 60)
    print("[1/4] Finding files...")
    print("=" * 60)

    video_extensions = (".mp4", ".mov")
    script_extensions = (".pdf", ".docx", ".doc")

    video_files = [f for f in os.listdir(gdrive_path) if f.lower().endswith(video_extensions)]
    script_files = [f for f in os.listdir(gdrive_path) if f.lower().endswith(script_extensions)]

    if not video_files:
        raise FileNotFoundError(f"No video files ({', '.join(video_extensions)}) found in {gdrive_path}")
    if not script_files:
        raise FileNotFoundError(f"No script files ({', '.join(script_extensions)}) found in {gdrive_path}")

    video_path = os.path.join(gdrive_path, video_files[0])
    script_path = os.path.join(gdrive_path, script_files[0])

    video_base = os.path.splitext(video_files[0])[0]
    script_base = os.path.splitext(script_files[0])[0]

    whisper_cache = os.path.join(gdrive_path, f"{video_base}.whisper.json")
    jscript_cache = os.path.join(gdrive_path, f"{script_base}.jscript.txt")
    output_path = os.path.join(gdrive_path, f"{video_base}.ass")

    print(f"  Video:  {video_files[0]}")
    print(f"  Script: {script_files[0]}")

    # ── Step 2: Check ffmpeg ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("[2/4] Checking ffmpeg...")
    print("=" * 60)

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("  ffmpeg is already installed.")
    except FileNotFoundError:
        print("  Installing ffmpeg...")
        subprocess.run(["apt-get", "install", "-qq", "ffmpeg"], check=True, capture_output=True)
        print("  ffmpeg installed.")

    # ── Step 3: Whisper + Jscript extraction (parallel, with caching) ───
    print()
    print("=" * 60)
    print("[3/4] Running Whisper transcription & script extraction...")
    print("=" * 60)

    transcription = None
    jscript_text = None

    whisper_cached = os.path.exists(whisper_cache)
    jscript_cached = os.path.exists(jscript_cache)

    if whisper_cached:
        print(f"  Cached transcription found, loading {os.path.basename(whisper_cache)}...")
        with open(whisper_cache, "r", encoding="utf-8") as f:
            transcription = json.load(f)
        print(f"  ✓ Whisper transcription loaded ({len(transcription)} segments).")

    if jscript_cached:
        print(f"  Cached script found, loading {os.path.basename(jscript_cache)}...")
        with open(jscript_cache, "r", encoding="utf-8") as f:
            jscript_text = f.read()
        print("  ✓ Script extraction loaded.")

    # Run whichever steps are still needed, in parallel if both are missing
    future_to_name = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        if not whisper_cached:
            future_to_name[executor.submit(
                generate_whisper_script, video_path, chunk_length=chunk_length
            )] = "whisper"
        if not jscript_cached:
            future_to_name[executor.submit(
                extract_jscript, script_path, api_key=api_key
            )] = "jscript"

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            result = future.result()
            if name == "whisper":
                transcription = result
                with open(whisper_cache, "w", encoding="utf-8") as f:
                    json.dump(transcription, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Whisper transcription complete. Saved to {os.path.basename(whisper_cache)}.")
            else:
                jscript_text = result
                with open(jscript_cache, "w", encoding="utf-8") as f:
                    f.write(jscript_text)
                print(f"  ✓ Script extraction complete. Saved to {os.path.basename(jscript_cache)}.")

    # ── Step 4: Align and export ────────────────────────────────────────
    print()
    print("=" * 60)
    print("[4/4] Aligning transcription with script...")
    print("=" * 60)

    align_scripts(transcription, jscript_text, output_path, api_key=api_key, translate=translate)

    print()
    print("=" * 60)
    print(f"✅ DONE! Subtitle file saved to: {output_path}")
    print("=" * 60)
