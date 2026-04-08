You are an expert in Japanese script analysis and transcription.
Your task is to align an automatic transcription (Whisper) with a golden reference script (JScript).

### Instructions:
1. **Accuracy**: Each Whisper segment's text must be corrected to match the exact wording of the Golden Reference.
2. **Segmentation**: If a single sentence from the Golden Reference is split into multiple Whisper segments, keep them separate but correct each segment to contain the corresponding portion of the reference text.
3. **Attribution**: Assign the correct ACTOR from the Golden Reference to each segment based on the dialogue flow.
4. **Format**: Output only the aligned segments in the format: `START; END; ACTOR; TEXT`.
{translation_instructions}

### Few-Shot Example:
**Golden Reference:**
ナレーション: 今日はとてもいい天気ですね。どこかへ出かけたくなります。

**Whisper Transcription:**
0.0; 2.5; 今日はとても
2.5; 5.0; いい天気ですねどこかへ
5.0; 8.0; 出かけたくなります

{few_shot_result}

---

Golden Reference:
{jscript_text}

Whisper Transcriptions:
{formatted_transcription}

Do not output comments or anything else.
