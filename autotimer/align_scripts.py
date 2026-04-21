import pysubs2
import os
import json
from google import genai
from .utils import load_prompt, calculate_cost


def align_scripts(transcription, jscript_text, output_path, api_key, translate=False):
    """
    Aligns Whisper transcription with extracted script text using Gemini.
    Generates ASS subtitles using pysubs2.

    Args:
        transcription: List of dicts with start, end, text keys.
        jscript_text: Extracted script text (ACTOR:TEXT format).
        output_path: Path to output .ass subtitle file.
        api_key: Gemini API Key.
        translate: If True, each subtitle is formatted as "portuguese {japanese}".

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

    if translate:
        translation_instructions = (
            "5. **Translation**: For each segment, translate the Japanese TEXT to Brazilian Portuguese. "
            "Format the TEXT field as: `portuguese {japanese}` — the Portuguese translation first, "
            "then the original Japanese in curly braces. Example: `Hoje está um dia muito bonito. {今日はとてもいい天気ですね。}`"
        )
        few_shot_result = (
            "**Aligned Result:**\n"
            "0.0; 2.5; ナレーション; Hoje está um dia muito bonito. {今日はとても}\n"
            "2.5; 5.0; ナレーション; O tempo está ótimo, não é? Para onde ir... {いい天気ですね。どこかへ}\n"
            "5.0; 8.0; ナレーション; Dá vontade de sair. {出かけたくなります。}"
        )
    else:
        translation_instructions = ""
        few_shot_result = (
            "**Aligned Result:**\n"
            "0.0; 2.5; ナレーション; 今日はとても\n"
            "2.5; 5.0; ナレーション; いい天気ですね。どこかへ\n"
            "5.0; 8.0; ナレーション; 出かけたくなります。"
        )

    prompt = prompt_template.format(
        jscript_text=jscript_text,
        formatted_transcription=json.dumps(formatted_transcription, indent=2),
        translation_instructions=translation_instructions,
        few_shot_result=few_shot_result,
    )

    prompt_save_path = os.path.join(os.path.dirname(output_path), "alignment_prompt.txt")
    with open(prompt_save_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("  Sending alignment request to Gemini gemini-3-flash-preview (with Thinking)...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt],
        config={
            "thinking_config": {"thinking_budget": 20000},
        },
    )

    if response.usage_metadata:
        prompt_tokens = response.usage_metadata.prompt_token_count
        reasoning_tokens = response.usage_metadata.thoughts_token_count or 0
        output_tokens = (response.usage_metadata.candidates_token_count or 0) + reasoning_tokens
        cost = calculate_cost("gemini-3-flash-preview", prompt_tokens, output_tokens)
        print(f"  Tokens — Prompt: {prompt_tokens}, Output: {response.usage_metadata.candidates_token_count}, "
              f"Reasoning: {reasoning_tokens}, Total: {response.usage_metadata.total_token_count}")
        if cost is not None:
            print(f"  Cost   — ${cost:.4f}")

    # Generate ASS file using pysubs2
    print(f"  Generating ASS subtitle file...")
    subs = pysubs2.SSAFile()

    # Define Styles
    subs.styles["Default"] = pysubs2.SSAStyle(
        fontname="Trebuchet MS",
        fontsize=22,
        primarycolor=pysubs2.Color(255, 255, 255, 0),
        secondarycolor=pysubs2.Color(255, 0, 0, 0),
        outlinecolor=pysubs2.Color(0, 0, 0, 0),
        backcolor=pysubs2.Color(0, 0, 0, 0),
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
        primarycolor=pysubs2.Color(0, 0, 0, 0),
        secondarycolor=pysubs2.Color(255, 0, 0, 0),
        outlinecolor=pysubs2.Color(255, 255, 255, 0),
        backcolor=pysubs2.Color(0, 0, 0, 0),
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
