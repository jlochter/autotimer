You are an expert in Japanese script analysis and transcription.
Your task is to align an automatic transcription with a golden reference script.

The automatic transcription has many dialogues with start, end and text properties.
The golden reference script has each dialogue, with actor and text properties in the format ACTOR:TEXT.

Your task is fix or replace the text in the whisper transcription the text using golden reference. Also figure out who is the actor.

Output in the format:
START; END; ACTOR; TEXT

Golden Reference:
{jscript_text}

Whisper Transcriptions:
{formatted_transcription}

Do not output comments or anything else.
