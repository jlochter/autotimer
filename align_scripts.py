from google import genai
from google.genai import types
import json
import os
import argparse
import pysubs2
from dotenv import load_dotenv

load_dotenv()

def align_scripts(whisper_path, ocr_path, output_path, api_key=None):
    """
    Aligns Whisper transcription with OCR text using Gemini 2.5 Flash.
    Generates ASS subtitles using pysubs2.
    
    Args:
        whisper_path (str): Path to Whisper JSON output.
        ocr_path (str): Path to OCR text file.
        output_path (str): Path to output .ass subtitle file.
        api_key (str): Gemini API Key.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments.")

    client = genai.Client(api_key=api_key)
    
    print(f"Loading Whisper data from {whisper_path}...")
    with open(whisper_path, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)
        
    print(f"Loading OCR text from {ocr_path}...")
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    formatted_transcription = []
    for t in whisper_data:
        formatted_transcription.append({
            "start": float(t['start']),
            "end": float(t['end']),
            "text": t['text']
        })
        
    prompt = f"""
You are an expert in Japanese script analysis and transcription.
Your task is to align an automatic transcription with a golden reference script.

The automatic transcription has many dialogues with start, end and text properties.
The golden reference script has each dialogue, with actor and text properties in the format ACTOR:TEXT.

Your task is fix or replace the text in the whisper transcription the text using golden reference. Also figure out who is the actor.

Output in the format:
START; END; ACTOR; TEXT

Golden Reference:
{ocr_text}

Whisper Transcriptions:
{formatted_transcription}

Do not output comments or anything else.
"""

    print("Sending request to Gemini 2.5 Flash with Thinking...")
    model_id = 'gemini-2.5-flash' 

    response = client.models.generate_content(
        model=model_id,
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=20000)
        )
    )

    print("Request finished.")
    if response.usage_metadata:
        print(f"Token Usage: Prompt: {response.usage_metadata.prompt_token_count}, Candidates: {response.usage_metadata.candidates_token_count}, Total: {response.usage_metadata.total_token_count}")

    print("Generating ASS file using pysubs2...")
    subs = pysubs2.SSAFile()

    # Set default style
    default_style = pysubs2.SSAStyle(
        fontname="Arial",
        fontsize=22,
        alignment=2 # Bottom-center
    )
    subs.styles["Default"] = default_style

    lines = response.text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue

        parts = [p.strip() for p in line.split(';')]
        if len(parts) < 4:
            continue

        try:
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            actor = parts[2]
            text = parts[3]

            # pysubs2 uses milliseconds for start and end times
            event = pysubs2.SSAEvent(
                start=int(start_sec * 1000),
                end=int(end_sec * 1000),
                text=text,
                name=actor
            )
            subs.append(event)
        except ValueError:
            continue

    print(f"Saving aligned subtitles to {output_path}...")
    subs.save(output_path)
    print(f"Exported {len(subs)} lines.")
        
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align Whisper JSON and OCR Text to ASS subtitles.")
    parser.add_argument("--whisper", required=True, help="Path to Whisper JSON file")
    parser.add_argument("--ocr", required=True, help="Path to OCR text file")
    parser.add_argument("--output", required=True, help="Path to output ASS file")
    parser.add_argument("--api_key", help="Gemini API Key")
    
    args = parser.parse_args()
    
    align_scripts(args.whisper, args.ocr, args.output, args.api_key)
