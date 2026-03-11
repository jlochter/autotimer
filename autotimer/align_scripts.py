import pysubs2
import os
import json
from google import genai
from google.genai import types
from .utils import load_prompt


def align_scripts(transcription, jscript_text, output_path, api_key):
    """
    Aligns Whisper transcription with extracted script text using Gemini.
    Generates ASS subtitles using pysubs2.

    Args:
        transcription: List of dicts with start, end, text keys.
        jscript_text: Extracted script text (ACTOR:TEXT format).
        output_path: Path to output .ass subtitle file.
        api_key: Gemini API Key.

    Returns:
        Path to the generated .ass file.
    """
    client = genai.Client(api_key=api_key)

    # Format transcription for the prompt
    formatted_transcription = []
    for t in transcription:
        formatted_transcription.append({
            "start": float(t["start"]),
            "end": float(t["end"]),
            "text": t["text"],
        })

    # Load prompt from file
    prompt_template = load_prompt("align_scripts.md")

    prompt = prompt_template.format(
        jscript_text=jscript_text,
        formatted_transcription=json.dumps(formatted_transcription, indent=2)
    )

    print("  Sending alignment request to Gemini gemini-3-flash-preview (with Thinking)...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=20000)
        ),
    )

    if response.usage_metadata:
        print(f"  Tokens — Prompt: {response.usage_metadata.prompt_token_count}, "
              f"Output: {response.usage_metadata.candidates_token_count}, "
              f"Total: {response.usage_metadata.total_token_count}")

    # Generate ASS file using pysubs2
    print(f"  Generating ASS subtitle file...")
    subs = pysubs2.SSAFile()

    # Define Styles
    subs.styles["Default"] = pysubs2.SSAStyle(
        fontname="Trebuchet MS",
        fontsize=22,
        primarycolor=pysubs2.Color.from_string("&H00FFFFFF"),
        secondarycolor=pysubs2.Color.from_string("&H000000FF"),
        outlinecolor=pysubs2.Color.from_string("&H00000000"),
        backcolor=pysubs2.Color.from_string("&H00000000"),
        bold=True,
        scalex=100.0,
        scaley=100.0,
        borderstyle=1,
        outline=2.0,
        shadow=1.0,
        alignment=2,
        marginl=40,
        marginr=40,
        marginv=15,
        encoding=0
    )

    subs.styles["Sign"] = pysubs2.SSAStyle(
        fontname="Tahoma",
        fontsize=22,
        primarycolor=pysubs2.Color.from_string("&H00000000"),
        secondarycolor=pysubs2.Color.from_string("&H000000FF"),
        outlinecolor=pysubs2.Color.from_string("&H00FFFFFF"),
        backcolor=pysubs2.Color.from_string("&H00000000"),
        bold=True,
        scalex=100.0,
        scaley=100.0,
        borderstyle=1,
        outline=2.0,
        shadow=0.0,
        alignment=2,
        marginl=40,
        marginr=40,
        marginv=15,
        encoding=1
    )

    lines = response.text.strip().split("\n")
    for line in lines:
        if not line.strip():
            continue

        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 4:
            continue

        try:
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            actor = parts[2]
            text = parts[3]

            event = pysubs2.SSAEvent(
                start=int(start_sec * 1000),
                end=int(end_sec * 1000),
                text=text,
                name=actor,
            )
            subs.append(event)
        except ValueError:
            continue

    subs.save(output_path)
    print(f"  Exported {len(subs)} subtitle lines to {output_path}")
    return output_path
