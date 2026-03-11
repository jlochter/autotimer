# AutoTimer: Japanese Video Script Aligner

AutoTimer is a Python tool designed to create perfect `.ass` subtitles for Japanese videos by aligning audio transcription with an official PDF/DOCX script.

It leverages **faster-whisper** (large-v2) for transcription and **Gemini 3 Flash Preview** for script extraction and intelligent semantic alignment.

## 🚀 Quick Start (Google Colab)

The easiest way to use AutoTimer is via the provided Jupyter Notebook. It's designed to run in a single cell on Google Colab with GPU support.

1.  Open [AutoTimer.ipynb](AutoTimer.ipynb) in Google Colab.
2.  Set your `GOOGLE_API_KEY` (Get one at [Google AI Studio](https://aistudio.google.com/)).
3.  Set your `GDRIVE_PATH` to the folder containing your video and script.
4.  Run the cell.

The pipeline will:
- Install AutoTimer directly from GitHub.
- Mount your Google Drive.
- Transcribe the video.
- Extract dialogue from the script.
- Align them and save the `.ass` subtitle file next to your video.

## 📦 Installation (Local/CLI)

You can install AutoTimer as a Python package directly from GitHub:

```bash
pip install git+https://github.com/jlochter/autotimer.git
```

## 🛠️ Python Usage

```python
import autotimer

autotimer.run(
    api_key="YOUR_GEMINI_API_KEY",
    gdrive_path="/path/to/your/files"
)
```

## 📂 Requirements

- **Python 3.9+**
- **FFmpeg** (automatically installed in Colab)
- **NVIDIA GPU** (recommended for Whisper `large-v2` speed)
- **Google Gemini API Key**

## 🏗️ Project Structure

- `autotimer/`: Core Python package.
  - `pipeline.py`: Main orchestrator (`run` function).
  - `generate_whisper.py`: Whisper transcription logic.
  - `extract_jscript.py`: Script extraction via Gemini.
  - `align_scripts.py`: Semantic alignment via Gemini.
- `AutoTimer.ipynb`: The single-cell Colab notebook.
- `pyproject.toml`: Package configuration and dependencies.

## 📄 License

[MIT](LICENSE)

