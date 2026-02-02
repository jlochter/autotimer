from google import genai
import json
import os
import argparse
import datetime
import pysubs2
from dotenv import load_dotenv

load_dotenv()

def align_scripts(whisper_path, ocr_path, output_path, api_key=None):
    """
    Aligns Whisper transcription with OCR text using Gemini 2.5 Flash Lite.
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

    # Prepare prompt
    whisper_text_block = ""
    for seg in whisper_data:
        whisper_text_block += f"ID:{seg['id']} T:{seg['start']}-{seg['end']} Text:{seg['text']}\n"
        
    prompt = f"""
    You are an expert subtitle aligner. 
    I have a noisy phonetic Japanese transcription from Whisper (with timestamps) and a clean official script extracted via OCR (without timestamps and potentially with different line breaks).
    
    Your task is to return a JSON list of subtitle events.
    1. Match the meaning/kanji from the OCR Text to the corresponding timestamps in the Whisper Data.
    2. Replace the Whisper text with the correct clean OCR text.
    3. Identify the SPEAKER/ACTOR for each line from the OCR text (e.g., if it says "Naruto: Hello", the actor is "Naruto").
    4. If the OCR text has extra lines not found in audio, or vice versa, do your best to align what is audible.
    
    Output Format:
    Return ONLY a raw JSON list of objects. Do not wrap in markdown code blocks.
    [
        {{
            "start": float, // Start time in seconds
            "end": float,   // End time in seconds
            "actor": "string or null", // Name of speaker/actor
            "text": "string" // The aligned subtitle text
        }},
        ...
    ]

    ---
    WHISPER DATA:
    {whisper_text_block}
    
    ---
    OCR TEXT:
    {ocr_text}
    """

    print("Sending request to Gemini 2.5 Flash Lite...")
    model_id = 'gemini-2.0-flash-lite-preview-02-05' 

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
    except Exception as e:
        print(f"Error with model '{model_id}': {e}")
        print("Falling back to 'gemini-1.5-flash'...")
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )

    json_content = response.text
    
    # Strip markdown code blocks if present (just in case model ignores instructions)
    if json_content.startswith("```"):
        json_content = json_content.split("\n", 1)[1]
    if json_content.endswith("```"):
        json_content = json_content.rsplit("\n", 1)[0]
        
    try:
        aligned_data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        # Save raw response for debugging
        with open(output_path + ".debug.txt", "w") as f:
            f.write(json_content)
        raise

    print("Generating ASS file using pysubs2...")
    subs = pysubs2.SSAFile()
    
    # Optional: Customize default style
    style = pysubs2.SSAStyle(fontsize=50, primarycolor=pysubs2.Color(255, 255, 255))
    subs.styles["Default"] = style
    
    for item in aligned_data:
        start_ms = int(item.get("start", 0) * 1000)
        end_ms = int(item.get("end", 0) * 1000)
        text = item.get("text", "")
        actor = item.get("actor", "") or ""
        
        event = pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text, name=actor)
        subs.events.append(event)
    
    print(f"Saving aligned subtitles to {output_path}...")
    subs.save(output_path)
        
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align Whisper JSON and OCR Text to ASS subtitles.")
    parser.add_argument("--whisper", required=True, help="Path to Whisper JSON file")
    parser.add_argument("--ocr", required=True, help="Path to OCR text file")
    parser.add_argument("--output", required=True, help="Path to output ASS file")
    parser.add_argument("--api_key", help="Gemini API Key")
    
    args = parser.parse_args()
    
    align_scripts(args.whisper, args.ocr, args.output, args.api_key)
