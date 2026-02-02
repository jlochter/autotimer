# AutoTimer: Japanese Video Script Aligner

AutoTimer is a Python tool designed to create perfect `.ass` subtitles for Japanese videos by aligning audio transcription with an official PDF script. 

It leverages **faster-whisper** for transcription, **EasyOCR** for script extraction, and **Gemini 2.5 Flash Lite** (via `google-genai`) for intelligent semantic alignment, ensuring that the final subtitles use the correct Kanji and official phrasing from your script, rather than just phonetic transcription.

## Features

- **Transcription**: Uses `faster-whisper` (default `large-v3`) to generate timestamped Japanese text from video files.
- **OCR Extraction**: Converts PDF scripts to text using `pdf2image` and `EasyOCR`.
- **Smart Alignment**: Uses Google's Gemini models to align the "noisy" Whisper transcription with the "clean" PDF text, preserving the timestamps of the audio but the text of the script.
- **Privacy Focused**: Only text is sent to the Gemini API (no audio/video uploads), saving bandwidth and tokens.

## Application Flow

```mermaid
graph TD
    A[Video File (.mp4)] -->|generate_whisper.py| B(Whisper JSON)
    C[Script File (.pdf)] -->|extract_jscript.py| D(Script Text)
    
    B --> E{align_scripts.py}
    D --> E
    
    E -->|Gemini API| F[Aligned Subtitles (.ass)]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#9f9,stroke:#333,stroke-width:4px
```

## Prerequisites

- **Python 3.9+**
- **Poppler**: Required for PDF processing.
  - macOS: `brew install poppler`
- **FFmpeg**: Required for audio extraction (usually installed with faster-whisper/audio libraries, or `brew install ffmpeg`).
- **Google Gemini API Key**: You need a valid API key from Google AI Studio.

## Installation

1. Clone this repository (if applicable) or download the source.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Set your API Key

You can configure your API key in a `.env` file (recommended) or as an environment variable.

**Option A: Using .env (Recommended)**
Create a `.env` file in the project root:
```ini
GEMINI_API_KEY=your_api_key_here
```

**Option B: Environment Variable**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 2. Run the Main Script

Run `main.py` with your video file and PDF script:

```bash
python main.py --video episode_01.mp4 --script script_01.pdf
```

The output will be saved as `aligned.ass` by default. You can specify a custom output filename:

```bash
python main.py --video episode_01.mp4 --script script_01.pdf --output episode_01.ass
```

### 3. Individual Modules

You can also run modules independently:

- **Transcribe only**:
  ```bash
  python generate_whisper.py --video ep01.mp4 --output whisper.json
  ```

- **Extract text only**:
  ```bash
  python extract_jscript.py --pdf script.pdf --output script.txt
  ```

- **Align only**:
  ```bash
  python align_scripts.py --whisper whisper.json --ocr script.txt --output subtitles.ass --api_key "YOUR_KEY"
  ```

## Project Structure

- `main.py`: Orchestrates the entire flow.
- `generate_whisper.py`: Handles audio transcription.
- `extract_jscript.py`: Handles PDF OCR extraction.
- `align_scripts.py`: Handles text alignment using Gemini.
- `requirements.txt`: Python package dependencies.

## License

[MIT](LICENSE)
