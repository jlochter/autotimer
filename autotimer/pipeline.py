"""Main pipeline orchestrator for AutoTimer."""

import os
import subprocess
from .generate_whisper import generate_whisper_script
from .extract_jscript import extract_jscript
from .align_scripts import align_scripts


def run(api_key, gdrive_path, chunk_length=30):
    """
    Run the full AutoTimer pipeline.

    Finds video and script files in gdrive_path, transcribes the video
    with Whisper, extracts dialogue from the script with Gemini, aligns
    them, and exports an .ass subtitle file.

    Args:
        api_key: Google Gemini API key.
        gdrive_path: Path to directory containing video (.mp4/.mov) and script (.pdf/.docx/.doc) files.
        chunk_length: The length of the sliding window for transcription (default: 30).
    """
    if not api_key:
        raise ValueError("api_key is required. Get one at https://aistudio.google.com/")

    if not os.path.isdir(gdrive_path):
        raise FileNotFoundError(f"Directory not found: {gdrive_path}")

    # ── Step 1: Find files ──────────────────────────────────────────────
    print("=" * 60)
    print("[1/5] Finding files...")
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

    print(f"  Video:  {video_files[0]}")
    print(f"  Script: {script_files[0]}")

    # ── Step 2: Install ffmpeg ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("[2/5] Checking ffmpeg...")
    print("=" * 60)

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("  ffmpeg is already installed.")
    except FileNotFoundError:
        print("  Installing ffmpeg...")
        subprocess.run(["apt-get", "install", "-qq", "ffmpeg"], check=True, capture_output=True)
        print("  ffmpeg installed.")

    # ── Step 3: Whisper transcription ───────────────────────────────────
    print()
    print("=" * 60)
    print("[3/5] Running Whisper transcription...")
    print("=" * 60)

    transcription = generate_whisper_script(video_path, chunk_length=chunk_length)

    # ── Step 4: Extract script text ─────────────────────────────────────
    print()
    print("=" * 60)
    print("[4/5] Extracting dialogue from script...")
    print("=" * 60)

    jscript_text = extract_jscript(script_path, api_key=api_key)

    # ── Step 5: Align and export ────────────────────────────────────────
    print()
    print("=" * 60)
    print("[5/5] Aligning transcription with script...")
    print("=" * 60)

    # Output .ass file next to the video
    base_name = os.path.splitext(video_files[0])[0]
    output_path = os.path.join(gdrive_path, f"{base_name}.ass")

    align_scripts(transcription, jscript_text, output_path, api_key=api_key)

    print()
    print("=" * 60)
    print(f"✅ DONE! Subtitle file saved to: {output_path}")
    print("=" * 60)
