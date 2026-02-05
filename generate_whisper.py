from faster_whisper import WhisperModel
import json
import os
import argparse

def generate_whisper_script(video_path, output_path, model_size="turbo", device="auto", compute_type="default", duration=None):
    """
    Transcribes audio from a video file using faster-whisper.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output JSON file.
        model_size (str): Whisper model size to use.
        device (str): Device to use for inference ("cuda", "cpu", "auto").
        compute_type (str): Compute type for inference ("float16", "int8_float16", "int8", "default").
        duration (float): Optional limit to transcribe only up to this many seconds.
    """
    print(f"Loading Whisper model: {model_size} on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing {video_path}...")
    segments, info = model.transcribe(video_path,
                                      language="ja",
                                      word_timestamps=True,
                                      chunk_length=20,
                                      log_progress=True)

    transcript_data = []
    # Using list(segments) as faster-whisper's log_progress=True handles the progress bar
    for segment in list(segments):
        if duration and segment.start >= duration:
            break

        words_data = []
        segment_start = segment.start
        
        if segment.words:
            words = list(segment.words)
            if len(words) > 1:
                # Calculate average duration of words in the segment (excluding the first word)
                durations = [w.end - w.start for w in words[1:]]
                avg_duration = round(sum(durations) / len(durations), 2)
                
                # Adjust segment start
                new_start = max(words[0].start, words[0].end - avg_duration)
                segment_start = round(new_start,2)

        segment_end = round(segment.end, 2)
        if duration and segment_end > duration:
            segment_end = duration

        segment_data = {
            "id": segment.id,
            "start": segment_start,
            "end": segment_end,
            "text": segment.text
        }

        transcript_data.append(segment_data)

    print(f"\nSaving transcription to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=4)

    return transcript_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Whisper transcription from video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--model", default="turbo", help="Whisper model size")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--duration", type=float, help="Limit transcription to first N seconds")
    
    args = parser.parse_args()
    
    generate_whisper_script(args.video, args.output, args.model, args.device, duration=args.duration)
