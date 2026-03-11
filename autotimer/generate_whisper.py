"""Whisper transcription module using faster-whisper."""

import torch
from faster_whisper import WhisperModel


def generate_whisper_script(video_path, model_size="large-v2", chunk_length=30):
    """
    Transcribes audio from a video file using faster-whisper.

    Args:
        video_path: Path to the input video file.
        model_size: Whisper model size to use (default: large-v2).
        chunk_length: The length of the sliding window for transcription (default: 30).

    Returns:
        List of transcription segments with id, start, end, text.
    """
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print(f"  Using GPU: {torch.cuda.get_device_name(0)} with compute type {compute_type}")
    else:
        device = "cpu"
        compute_type = "int8"
        print(f"  Using CPU with compute type {compute_type}")

    print(f"  Loading Whisper model: {model_size}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"  Transcribing {video_path}...")
    segments, info = model.transcribe(
        video_path,
        language="ja",
        word_timestamps=True,
        chunk_length=chunk_length,
        log_progress=True,
    )

    transcription = list(segments)

    # Fix segment start times based on word-level timestamps
    transcription_fixed = []
    for segment in transcription:
        segment_start = segment.start

        if segment.words:
            words = list(segment.words)
            if len(words) > 1:
                durations = [w.end - w.start for w in words[1:]]
                avg_duration = round(sum(durations) / len(durations), 2)
                if words[0].start > avg_duration:
                    new_start = max(words[0].start, words[0].end - avg_duration)
                    segment_start = round(new_start, 2)

        segment_end = round(segment.end, 2)

        transcription_fixed.append({
            "id": segment.id,
            "start": segment_start,
            "end": segment_end,
            "text": segment.text,
        })

    print(f"  Transcription complete: {len(transcription_fixed)} segments.")
    return transcription_fixed
