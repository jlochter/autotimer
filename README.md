# AutoTimer: Japanese Video Script Aligner

AutoTimer is a Python tool designed to create perfect `.ass` subtitles for Japanese videos by aligning audio transcription with an official PDF/DOCX script.

It leverages **faster-whisper** (large-v2) for transcription, **Gemini 2.5 Pro** for script extraction (OCR), and **Gemini 2.0 Flash Preview** for intelligent semantic alignment — with optional Brazilian Portuguese translation done in the same alignment pass.

## 🚀 Quick Start (Google Colab)

The easiest way to use AutoTimer is via the provided Jupyter Notebook. It's designed to run in a single cell on Google Colab with GPU support.

1.  Open [AutoTimer.ipynb](AutoTimer.ipynb) in Google Colab.
2.  Set your `GOOGLE_API_KEY` (Get one at [Google AI Studio](https://aistudio.google.com/)).
3.  Set your `GDRIVE_PATH` to the folder containing your video and script.
4.  Run the cell.

The pipeline will:
- Install AutoTimer directly from GitHub.
- Mount your Google Drive.
- Transcribe the video (cached to `{video}.whisper.json` for reuse).
- Extract dialogue from the script (cached to `{script}.jscript.txt` for reuse).
- Align them and save the `.ass` subtitle file next to your video.

To enable translation, set `translate=True` in the notebook cell (see [Python Usage](#️-python-usage) below).

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

### Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Google Gemini API key |
| `gdrive_path` | `str` | — | Path to the folder containing the video and script files |
| `chunk_length` | `int` | `30` | Whisper sliding window length in seconds |
| `translate` | `bool` | `False` | Translate subtitles to Brazilian Portuguese |

### Caching

After the first run, intermediate results are saved next to your files:

| File | Contents |
|---|---|
| `{video}.whisper.json` | Whisper transcription segments |
| `{script}.jscript.txt` | Extracted dialogue lines |

On subsequent runs these files are loaded automatically, skipping the Whisper and/or Gemini extraction steps. Delete them to force a fresh run.

### Translation

When `translate=True`, Gemini translates each subtitle to Brazilian Portuguese during the alignment step — no extra API call needed. Each subtitle line is formatted as:

```
portuguese {japanese}
```

Example:
```
Hoje está um dia muito bonito, não é? {今日はとてもいい天気ですね。}
```

```python
autotimer.run(
    api_key="YOUR_GEMINI_API_KEY",
    gdrive_path="/path/to/your/files",
    translate=True
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
  - `align_scripts.py`: Semantic alignment (and optional translation) via Gemini.
- `AutoTimer.ipynb`: The single-cell Colab notebook.
- `pyproject.toml`: Package configuration and dependencies.

## 📄 License

[MIT](LICENSE)

