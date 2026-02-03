from faster_whisper import WhisperModel, BatchedInferencePipeline
import json
import os
import argparse
from tqdm import tqdm

def generate_whisper_script(video_path, output_path, model_size="medium", device="auto", compute_type="default"):
    """
    Transcribes audio from a video file using faster-whisper.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output JSON file.
        model_size (str): Whisper model size to use.
        device (str): Device to use for inference ("cuda", "cpu", "auto").
        compute_type (str): Compute type for inference ("float16", "int8_float16", "int8", "default").
    """
    print(f"Loading Whisper model: {model_size} on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model)

    print(f"Transcribing {video_path}...")
    segments, info = batched_model.transcribe(video_path, batch_size=16, language="ja")

    total_duration = round(info.duration, 2)
    print(f"Detected language '{info.language}' with probability {info.language_probability}")

    transcript_data = []
    with tqdm(total=total_duration, unit="s", desc="Transcribing") as pbar:
        for segment in segments:
            segment_data = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            transcript_data.append(segment_data)
            
            segment_duration = segment.end - segment.start
            pbar.update(segment_duration)

    print(f"Saving transcription to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=4)

    return transcript_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Whisper transcription from video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    generate_whisper_script(args.video, args.output, args.model, args.device)
