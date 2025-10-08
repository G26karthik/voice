# 🎓 Lecture Voice-to-Notes Generator# Lecture Voice-to-Notes Generator



> An intelligent AI-powered educational tool that transforms lecture audio into comprehensive study materials with automatic transcription, summarization, quiz generation, and flashcard creation.An end-to-end AI EdTech tool that:



[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


1. Transcribes lecture audio (upload MP3/WAV or live microphone) using Whisper with automatic backend selection (faster-whisper -> transformers -> openai-whisper fallback).

2. Summarizes long transcripts into structured notes (adjustable length) using T5 or BART.

3. Generates quizzes (MCQs) & flashcards (JSON export) via keyword extraction (KeyBERT / spaCy fallback).

4. Exports PDF, DOCX, and Flashcards JSON.

5. Simple Streamlit web UI (tabs: Upload Mode, Live Mode, Results & Export). Ready for deployment on Hugging Face Spaces or Streamlit Community Cloud.

---

## Features

## 📋 Table of Contents

- [Overview](#overview)- Automatic language detection (Whisper)

- [Key Features](#key-features)- Adjustable summarization detail: short / medium / detailed

- [Tech Stack](#tech-stack)- Continuous pseudo-streaming transcription (buffered time chunks with incremental diff updates)

- [System Architecture](#system-architecture)- Chunked parallel transcription for large files (configurable chunk & overlap seconds)

- [Installation](#installation)- Optional simple energy-based VAD (skip silent spans when chunking)

- [Usage](#usage)- Progress bar during long transcription jobs

- [Project Structure](#project-structure)- Mixed precision (float16) on GPU paths when available

- [Performance Optimization](#performance-optimization)- Caching of embeddings & QA answers to accelerate repeated MCQ / flashcard regeneration

- [Troubleshooting](#troubleshooting)- MCQs (5) and Flashcards (5) generation

- [License](#license)- Download buttons for PDF, DOCX, JSON

- Graceful fallbacks if KeyBERT or spaCy are missing

---

## Project Structure

## 🎯 Overview

```

An end-to-end AI EdTech solution that converts lecture audio into structured study materials:app.py                # Streamlit UI

utils/

1. **📝 Transcription** - Upload audio files (MP3/WAV) or record live lectures with automatic speech recognition	transcribe.py       # File + live transcription helpers

2. **📄 Summarization** - Generate concise, structured notes with adjustable detail levels	summarize.py        # Chunking & summarization

3. **❓ Quiz Generation** - Create multiple-choice questions with semantic distractor generation	quiz.py             # Keyword extraction + MCQs + flashcards

4. **📚 Flashcard Creation** - Automatically generate study flashcards with context-aware definitions	exporters.py        # PDF / DOCX / JSON builders

5. **📤 Multi-format Export** - Export to PDF, DOCX, and JSON formatsrequirements.txt

README.md

---```



## ✨ Key Features## Installation



### 🎙️ **Advanced Transcription**### 1. Clone & Install Dependencies

- **Multi-backend ASR**: Automatic fallback chain (Faster-Whisper → Transformers → OpenAI-Whisper)

- **GPU Acceleration**: CUDA support with float16 mixed precision```bash

- **Parallel Processing**: Chunked transcription with configurable overlap for accuracypip install -r requirements.txt

- **Voice Activity Detection (VAD)**: Skip silent segments to reduce compute time```

- **Live Recording**: Real-time microphone capture with buffered streaming

- **Language Auto-detection**: Automatically identifies spoken languageWhisper & transformers will download models on first run. For spaCy English model (optional, improves keyword extraction):



### 📊 **Intelligent Summarization**```bash

- **Adjustable Length**: Short, Medium, or Detailed summariespython -m spacy download en_core_web_sm

- **Chunked Processing**: Handles long transcripts (30+ minutes)```

- **Structured Notes**: Optional topic segmentation with semantic embeddings

- **Model Selection**: T5-small/base with BART fallbackIf `pyaudio` fails on Windows, you may use `sounddevice` (already included). Ensure a working microphone is available.



### 🧠 **Smart Quiz & Flashcard Generation**### 2. Run the App

- **Semantic Understanding**: Uses Sentence-Transformers for context-aware content

- **Quality Improvements**:```bash

  - **Keyword Extraction**: Stop word filtering, NER prioritization, length-based rankingstreamlit run app.py

  - **MCQ Generation**: Unique distractor validation, prevents answer duplication```

  - **Flashcard Definitions**: 3-tier strategy (QA model → sentence extraction → context fallback)

- **Answer Validation**: Ensures logical consistencyOpen the provided local URL in your browser.

- **Caching**: Embedded QA answer caching for faster regeneration

## Usage

### 🎨 **Enhanced User Interface**

- **Card-based Display**: Visual flashcard layout with color-coded sections1. (Optional) In the sidebar, click "Warmup models" to pre-load.

- **Interactive MCQs**: Option labels (A/B/C/D), expandable answers, correct answer highlighting2. Choose a tab:

- **Progress Tracking**: Real-time progress bars during transcription	 - Upload Mode: upload an MP3/WAV, wait for transcription, then click "Process Transcript".

- **Responsive Design**: Clean, intuitive Streamlit interface	 - Live Mode: click Start Live to begin capturing mic audio (buffered every few seconds). Click Stop Live, then "Process Live Transcript".

3. Switch to Results & Export tab to review transcript, summary, MCQs, flashcards.

### 📤 **Flexible Export Options**4. Download PDF, DOCX, or flashcards JSON.

- **PDF**: Professional formatted notes with ReportLab

- **DOCX**: Microsoft Word compatible documents## Environment Notes

- **JSON**: Flashcard export (Anki-compatible format)

- CPU is supported; GPU (if available) speeds up inference.

---- For extremely long lectures, consider splitting audio externally; memory use grows with transcript size.

- Live mode currently re-processes the whole rolling buffer each chunk (simpler, not true streaming). For production low-latency, integrate a streaming ASR approach.

## 🛠️ Tech Stack

## Deployment

### **Core Technologies**

| Component | Technology | Purpose |### Hugging Face Spaces (Recommended)

|-----------|-----------|---------|

| **Frontend** | Streamlit | Interactive web UI |1. Create a new Space (SDK: Streamlit).

| **Backend** | Python 3.13 | Core application logic |2. Upload repo files (or connect Git). Ensure `requirements.txt` present.

| **Deep Learning** | PyTorch 2.7+ (CUDA 11.8) | GPU acceleration |3. First build will download models.



### **AI/ML Models**### Streamlit Community Cloud

| Model | Library | Use Case |

|-------|---------|----------|1. Create a new app pointing to `app.py`.

| **Whisper** | Faster-Whisper / Transformers / OpenAI-Whisper | Speech-to-text transcription |2. Add `requirements.txt`.

| **T5-small** | Hugging Face Transformers | Text summarization |3. Deploy; models cached after first run.

| **DistilBERT** | Hugging Face Transformers | Question answering for definitions |

| **Sentence-BERT** | Sentence-Transformers | Semantic embeddings & similarity |### Caching & Model Size

| **KeyBERT** | KeyBERT | Keyword extraction |

| **spaCy** | spaCy (en_core_web_sm) | Named entity recognition |`openai/whisper-small` ~ 1.4GB. Ensure environment has sufficient storage. If resource constrained, swap to `openai/whisper-base` in `utils/transcribe.py`.



### **Key Libraries**## Configuration & Performance Tweaks

```

streamlit              # Web UI framework- Whisper backend & size:

torch                  # Deep learning backend	- In the sidebar choose model size (`tiny`, `base`, `small`, etc.).

transformers           # Hugging Face models	- Set env var `WHISPER_MODEL_SIZE` to override default.

faster-whisper         # Optimized Whisper inference	- Faster-whisper auto-selected if installed; falls back gracefully.

sentence-transformers  # Semantic embeddings- Parallel chunk transcription:

keybert                # Keyword extraction	- Sidebar: `Chunk seconds` (e.g. 30) & `Overlap seconds` (e.g. 4) for accuracy at boundaries.

spacy                  # NLP processing	- Increase `Parallel workers` (default CPU core heuristic) to speed multi-core processing.

sounddevice            # Audio recording- Voice Activity Detection (VAD): enable `Use VAD` to skip low-energy segments and reduce compute.

reportlab              # PDF generation- Beam size: accuracy vs speed tradeoff (1 = fastest, >1 = better). Adjust in sidebar.

python-docx            # DOCX generation- Fast Mode:

numpy                  # Numerical computing	- Uses distilled summarizer, smaller ASR model selection hint, reduced beam size.

```- Mixed precision:

	- Automatically used for GPU inference (torch autocast) when transformers path is active; faster-whisper already uses efficient kernels.

---- Caching:

	- Embedding vectors for distractors and QA answers cached in-memory (`quiz.py`). Re-running MCQ/flashcard generation avoids recomputation.

## 🏗️ System Architecture- Structured notes toggle: disable if you want the absolute fastest run (skips embedding-based topic segmentation).



### **Processing Pipeline**## Error Handling



```- Missing optional libs (KeyBERT, spaCy) → fallback logic prints fewer / simpler keywords.

┌─────────────────┐- Audio issues (unsupported format) → exception surfaced in UI.

│  Audio Input    │- Live mode start failure (no microphone) → error message.

│ (File / Live)   │

└────────┬────────┘## Flashcards JSON Format

         │

         ▼```json

┌─────────────────┐[

│  Transcription  │	{ "term": "Backpropagation", "definition": "First sentence in transcript mentioning Backpropagation" }

│  - Faster-Whisper (GPU)]

│  - Chunked parallel processing```

│  - VAD filteringImport mapping for Anki: term -> Front, definition -> Back.

└────────┬────────┘

         │## Roadmap / Possible Enhancements

         ▼

┌─────────────────┐- True streaming ASR with partial segment updates

│  Summarization  │- Semantic answer generation for definitions (QA model integration)

│  - T5/BART models- Theming & advanced note structuring (headings detection)

│  - Chunked processing- Persistence / user accounts

│  - Structured notes

└────────┬────────┘## License

         │

         ▼MIT (adjust as needed).

┌─────────────────┐

│  Quiz/Flashcards│## Troubleshooting

│  - KeyBERT keywords

│  - Semantic MCQs| Issue | Fix |

│  - QA definitions|-------|-----|

└────────┬────────┘| `pyaudio` install fails | Use sounddevice (already used). Remove pyaudio line if not needed. |

         │| Slow transcription | Use Fast Mode, reduce beam size, enable chunking + VAD, ensure GPU. |

         ▼| No keywords | Install KeyBERT or spaCy model. |

┌─────────────────┐| Large PDF truncated | Summary first; consider shorter summaries. |

│  Export Outputs │| Error: *We expect a numpy ndarray or torch tensor as input, got <class '_io.BytesIO'>* | Fixed in latest version: uploaded audio is written to a temp file before calling ASR. If persisting, ensure `transcribe_file` uses temp path logic. |

│ PDF/DOCX/JSON   │

└─────────────────┘## Performance Tips

```

| Goal | Recommendation |

### **Module Breakdown**|------|----------------|

| Faster transcription (GPU) | Install CUDA-enabled PyTorch; faster-whisper auto-uses GPU with float16 & parallel chunking. |

```| Faster transcription (CPU) | Enable Fast Mode, choose `base` or `tiny`, set chunking (30s) & beam size = 1. |

app.py                    # Main Streamlit UI orchestrator| Better accuracy | Increase beam size to 3–5 and select `small`/`medium`. |

├── utils/| Faster summarization | Fast Mode uses distilled BART model. |

│   ├── transcribe.py     # ASR backend manager (Whisper variants)| Large audio (>30 min) | Split into smaller files; summarize each then combine. |

│   ├── summarize.py      # Chunked summarization with T5/BART| Memory constraints | Use `tiny` model, disable structured notes, reduce beam size. |

│   ├── quiz.py           # Keyword extraction, MCQ & flashcard generation| Re-run MCQs/flashcards faster | Benefit from embedding & QA caching automatically. |

│   ├── exporters.py      # PDF/DOCX/JSON export handlers

│   ├── structure.py      # Topic segmentation & structured notesEnvironment variables (optional):

│   └── models.py         # Model loading & caching utilities

``````

set WHISPER_MODEL_SIZE=base

---set USE_FAST_WHISPER=1

```

## 📦 Installation

Or on Unix shells:

### **Prerequisites**```

- Python 3.13+export WHISPER_MODEL_SIZE=base

- NVIDIA GPU (optional, but recommended for 5-10x speed improvement)export USE_FAST_WHISPER=1

- CUDA 11.8+ (for GPU support)```

- 4GB+ RAM (8GB+ recommended)

---

### **Step 1: Clone Repository**Happy studying!

```bash
git clone https://github.com/yourusername/lecture-voice-to-notes.git
cd lecture-voice-to-notes
```

### **Step 2: Create Virtual Environment**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### **Step 3: Install Dependencies**

**For CPU-only:**
```bash
pip install -r requirements.txt
```

**For GPU (NVIDIA CUDA 11.8):**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

**Verify GPU Installation:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Expected output: CUDA available: True
```

### **Step 4: Download spaCy Model (Optional)**
```bash
python -m spacy download en_core_web_sm
```
> Improves keyword extraction and named entity recognition

### **Step 5: Run Application**
```bash
streamlit run app.py
```
> Opens browser at `http://localhost:8501`

---

## 🚀 Usage

### **1. Upload Mode (Audio Files)**
1. Navigate to **"Upload Audio"** tab
2. Upload an audio file (MP3, WAV, M4A, FLAC)
3. Wait for transcription to complete (progress bar shown)
4. Click **"Process Transcript"** to generate summary, MCQs, and flashcards

### **2. Live Recording Mode**
1. Navigate to **"Live Recording"** tab
2. Click **"Start Live Recording"**
3. Speak into your microphone (captures in 5-second chunks)
4. Click **"Stop Recording"**
5. Click **"Process Live Transcript"**

### **3. View Results**
Navigate to **"Results & Export"** tab to see:
- 📝 **Full Transcript** - Complete transcribed text
- 📄 **Summary** - Structured notes (adjustable length in sidebar)
- ❓ **MCQs** - Multiple choice questions with expandable answers
- 📚 **Flashcards** - Term/definition pairs with visual card layout

### **4. Export Options**
- **📄 Download PDF** - Professional formatted document
- **📝 Download DOCX** - Microsoft Word document
- **📚 Download Flashcards JSON** - Anki-compatible format

### **5. Configuration (Sidebar)**
- **Model Size**: tiny / base / small / medium / large
- **Summarization Length**: Short / Medium / Detailed
- **Chunking**: Enable parallel processing (recommended for >5 min audio)
- **VAD**: Skip silent segments
- **Fast Mode**: Trade quality for speed
- **Warmup Models**: Pre-load models to reduce first-run latency

---

## 📁 Project Structure

```
Voice/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git exclusion rules
│
├── utils/                          # Core modules
│   ├── __init__.py
│   ├── transcribe.py               # Whisper ASR backends (faster-whisper/transformers/openai-whisper)
│   ├── summarize.py                # T5/BART summarization with chunking
│   ├── quiz.py                     # Keyword extraction, MCQ generation, flashcard creation
│   ├── exporters.py                # PDF/DOCX/JSON export utilities
│   ├── structure.py                # Topic segmentation & structured notes
│   └── models.py                   # Model loading & caching
│
└── .venv/                          # Virtual environment (excluded from Git)
```

### **File Responsibilities**

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~400 | Streamlit UI, session state management, tab navigation |
| `utils/transcribe.py` | ~380 | ASR model loading, file/live transcription, chunked parallel processing |
| `utils/summarize.py` | ~200 | Chunked summarization, model selection, structured notes |
| `utils/quiz.py` | ~280 | Keyword extraction, semantic MCQ generation, 3-tier flashcard definitions |
| `utils/exporters.py` | ~150 | PDF/DOCX formatting, JSON export |
| `utils/structure.py` | ~120 | Topic segmentation using sentence embeddings |
| `utils/models.py` | ~80 | Shared model loading, caching utilities |

---

## ⚡ Performance Optimization

### **GPU Acceleration**
✅ **Enabled** - RTX 4060 / 3060 / A100 / V100 / etc.

**Expected Speed Improvements:**
| Audio Length | CPU (base) | GPU (small) | Speedup |
|--------------|------------|-------------|---------|
| 5 minutes    | ~180s      | ~25s        | **7.2x** |
| 15 minutes   | ~540s      | ~70s        | **7.7x** |
| 30 minutes   | ~1080s     | ~140s       | **7.7x** |

### **Optimization Strategies**

#### **For Speed**
```python
# Sidebar Settings
Model Size: base or tiny
Fast Mode: Enabled
Beam Size: 1
Chunking: Enabled (30s chunks, 4s overlap)
VAD: Enabled
Parallel Workers: 4-8 (based on CPU cores)
```

#### **For Accuracy**
```python
# Sidebar Settings
Model Size: small or medium
Fast Mode: Disabled
Beam Size: 5
Chunking: Enabled (60s chunks, 8s overlap)
VAD: Disabled
Structured Notes: Enabled
```

#### **For Memory Efficiency**
```python
# Sidebar Settings
Model Size: tiny
Chunking: Enabled (15s chunks, 2s overlap)
Structured Notes: Disabled
```

### **Caching Mechanisms**
- ✅ **Embedding Cache**: Stores sentence embeddings for MCQ distractor generation
- ✅ **QA Answer Cache**: Caches DistilBERT answers for flashcard definitions
- ✅ **Model Cache**: Keeps loaded models in memory across sessions

---

## 🐛 Troubleshooting

### **Common Issues**

#### **1. GPU Not Detected**
**Problem:** `Device set to use cpu`

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **2. Poor Quiz/Flashcard Quality**
**Problem:** Generic questions, weak definitions

**Causes:**
- Audio too short (<5 minutes) or lacks educational content
- Generic conversational audio (not lecture content)

**Solutions:**
- Use 15-30 minute educational lectures
- Ensure audio has technical terms and concepts
- Check keyword extraction output (should have domain-specific terms)

#### **3. Slow Transcription**
**Problem:** Takes >10 minutes for 5-minute audio

**Solutions:**
```python
# Enable optimizations in sidebar:
✓ Fast Mode
✓ Chunking (30s, 4s overlap)
✓ VAD enabled
✓ Model size: base or tiny
✓ Beam size: 1
✓ Parallel workers: 4+
```

#### **4. Memory Errors**
**Problem:** `CUDA out of memory` or `RAM exceeded`

**Solutions:**
- Reduce model size: `small` → `base` → `tiny`
- Enable chunking with smaller chunk size (15-30s)
- Disable structured notes
- Close other GPU-intensive applications

#### **5. No Keywords Found**
**Problem:** MCQs/flashcards generation fails

**Solutions:**
```bash
# Install optional dependencies
pip install keybert
python -m spacy download en_core_web_sm
```

#### **6. Microphone Not Working**
**Problem:** Live recording fails

**Solutions:**
- Check microphone permissions (Windows Settings → Privacy)
- Try different audio input device (sidebar dropdown)
- Install alternative audio library:
  ```bash
  pip install sounddevice
  ```

### **Performance Benchmarks**

**Test System:** RTX 4060 Laptop GPU, 16GB RAM, i7-12700H

| Configuration | 5-min Audio | 15-min Audio | 30-min Audio |
|---------------|-------------|--------------|--------------|
| **CPU (base)** | 180s | 540s | 1080s |
| **GPU (small, beam=1)** | 25s | 70s | 140s |
| **GPU (small, beam=5)** | 40s | 115s | 230s |
| **GPU (small, chunked+VAD)** | 20s | 55s | 110s |

### **Quality Validation**

**Recommended Test:**
1. Use 15-30 minute educational lecture (e.g., Khan Academy, MIT OpenCourseWare)
2. Check keyword extraction: Should have 10+ domain-specific terms
3. Validate MCQs: Questions should be answerable from transcript
4. Review flashcards: Definitions should be contextually accurate

---

## 🎨 Recent Improvements

### **v2.0 - UI Enhancement (Oct 2025)**
- ✨ Card-based flashcard display with visual hierarchy
- ✨ Interactive MCQ layout with expandable answers
- ✨ Color-coded correct answers (green checkmarks)
- ✨ A/B/C/D option labeling for quiz questions

### **v1.9 - Quality Improvements (Oct 2025)**
- ✨ Enhanced keyword extraction (stop words, NER prioritization)
- ✨ Improved MCQ distractor validation (unique answers)
- ✨ 3-tier flashcard definition strategy (QA → sentence → context)
- ✨ Answer caching for faster regeneration

### **v1.8 - Bug Fixes (Oct 2025)**
- 🐛 Fixed tensor stacking error in semantic distractor generation
- 🐛 Simplified embedding cache logic (removed index errors)
- 🐛 Improved error handling for edge cases

### **v1.7 - GPU Support (Oct 2025)**
- ⚡ Added CUDA 11.8 support for PyTorch
- ⚡ Enabled float16 mixed precision on GPU
- ⚡ 7-10x speed improvement with GPU acceleration

---

## 📊 Use Cases

### **Students**
- 📝 Convert recorded lectures into study notes
- ❓ Generate practice quizzes for exam preparation
- 📚 Create flashcards for spaced repetition learning
- 📄 Export notes to PDF for offline studying

### **Educators**
- 🎓 Generate supplementary materials from lecture recordings
- ✏️ Create quiz banks from course content
- 📤 Provide students with structured study guides
- 🔍 Quickly review and summarize long lectures

### **Content Creators**
- 🎙️ Transcribe podcast episodes
- 📝 Generate show notes and summaries
- 🔑 Extract key topics and concepts
- 📄 Create episode guides for audiences

### **Researchers**
- 📊 Transcribe conference talks and seminars
- 📝 Summarize research presentations
- 🔍 Extract key findings and methodologies
- 📚 Create reference flashcards for literature review

---

## 🔮 Future Enhancements

- [ ] True streaming ASR with partial segment updates
- [ ] Multi-language support (Spanish, French, German, etc.)
- [ ] User authentication and session persistence
- [ ] Cloud storage integration (Google Drive, Dropbox)
- [ ] Collaborative note-sharing features
- [ ] Advanced theming and customization options
- [ ] Mobile app (React Native / Flutter)
- [ ] Integration with Notion, Obsidian, Roam Research
- [ ] Diagram/image extraction from lecture slides
- [ ] Video lecture support (YouTube, Zoom recordings)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI Whisper** - Robust speech recognition models
- **Hugging Face** - Transformers library and model hub
- **Streamlit** - Elegant web app framework
- **KeyBERT** - Efficient keyword extraction
- **spaCy** - Industrial-strength NLP

---

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

**Made with ❤️ by the Lecture Voice-to-Notes Team**

**⭐ If you find this project useful, please consider giving it a star!**
