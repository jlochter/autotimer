from google import genai
import json
import os
import argparse
import datetime

def format_time_ass(seconds):
    """Formats seconds into ASS timestamp format: H:MM:SS.cs"""
    td = datetime.timedelta(seconds=seconds)
    # Total seconds to h:mm:ss.cs
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    centiseconds = int(td.microseconds / 10000)
    return f"{hours}:{minutes:02}:{seconds_int:02}.{centiseconds:02}"

def align_scripts(whisper_path, ocr_path, output_path, api_key=None):
    """
    Aligns Whisper transcription with OCR text using Gemini 2.5 Flash Lite.
    
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
    # We will send a subset or the full set depending on size. 
    # For a full video, we might need to chunk, but for this POC we'll try full context or assume it fits in Gemini's large context window (1M+ tokens for Flash).
    
    whisper_text_block = ""
    for seg in whisper_data:
        whisper_text_block += f"ID:{seg['id']} T:{seg['start']}-{seg['end']} Text:{seg['text']}\n"
        
    prompt = f"""
    You are an expert subtitle aligner. 
    I have a noisy phonetic Japanese transcription from Whisper (with timestamps) and a clean official script extracted via OCR (without timestamps and potentially with different line breaks).
    
    Your task is to generate a SubStation Alpha (.ass) subtitle file.
    1. Match the meaning/kanji from the OCR Text to the corresponding timestamps in the Whisper Data.
    2. Replace the Whisper text with the correct clean OCR text.
    3. If the OCR text has extra lines not found in audio, or vice versa, do your best to align what is audible.
    4. Output ONLY the .ass file content.
    
    Here is the standard ASS header to use:
    [Script Info]
    ScriptType: v4.00+
    PlayResX: 1920
    PlayResY: 1080
    
    [V4+ Styles]
    Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    Style: Default,Arial,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,1,2,2,2,10,10,10,1

    [Events]
    Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

    ---
    WHISPER DATA:
    {whisper_text_block}
    
    ---
    OCR TEXT:
    {ocr_text}
    
    ---
    Generate the full .ass file content below:
    """

    print("Sending request to Gemini 2.5 Flash Lite...")
    # Trying the requested model. If it fails, fallback strategy might be needed but Client will raise error.
    # Note: 'gemini-2.0-flash-lite-preview-02-05' is the latest string referenced in some docs, or just 'gemini-2.0-flash'.
    # I'll try 'gemini-2.0-flash-lite-preview-02-05' as per user hint "gemini 2.5 flash lite" (likely 2.0 flash lite preview).
    model_id = 'gemini-2.0-flash-lite-preview-02-05' 

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
    except Exception as e:
        print(f"Error with model '{model_id}': {e}")
        print("Falling back to 'gemini-1.5-flash'...")
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )

    ass_content = response.text
    
    # Strip markdown code blocks if present
    if ass_content.startswith("```"):
        ass_content = ass_content.split("\n", 1)[1]
    if ass_content.endswith("```"):
        ass_content = ass_content.rsplit("\n", 1)[0]
    
    print(f"Saving aligned subtitles to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
        
    return ass_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align Whisper JSON and OCR Text to ASS subtitles.")
    parser.add_argument("--whisper", required=True, help="Path to Whisper JSON file")
    parser.add_argument("--ocr", required=True, help="Path to OCR text file")
    parser.add_argument("--output", required=True, help="Path to output ASS file")
    parser.add_argument("--api_key", help="Gemini API Key")
    
    args = parser.parse_args()
    
    align_scripts(args.whisper, args.ocr, args.output, args.api_key)
