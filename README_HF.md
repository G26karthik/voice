---
title: Lecture Voice-to-Notes Generator
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: mit
tags:
  - education
  - ai
  - nlp
  - speech-recognition
  - summarization
  - quiz-generation
  - flashcards
  - whisper
  - transformers
  - edtech
models:
  - openai/whisper-small
  - t5-small
  - distilbert-base-uncased
  - sentence-transformers/all-MiniLM-L6-v2
python_version: 3.10
---

# Lecture Voice-to-Notes Generator

An intelligent AI-powered educational tool that transforms lecture audio into comprehensive study materials with automatic transcription, summarization, quiz generation, and flashcard creation.

## Features

- 🎙️ **Advanced Transcription** - Multi-backend ASR with GPU acceleration
- 📄 **Intelligent Summarization** - Adjustable length with structured notes
- ❓ **Smart Quiz Generation** - Context-aware multiple choice questions
- 📚 **Flashcard Creation** - Automatic term/definition extraction
- 📤 **Multi-format Export** - PDF, DOCX, and JSON outputs

## Tech Stack

- **Whisper** (Speech-to-text)
- **T5** (Summarization)
- **DistilBERT** (Question answering)
- **Sentence-Transformers** (Semantic embeddings)
- **KeyBERT** (Keyword extraction)
- **Streamlit** (Web UI)

## Usage

1. Upload an audio file (MP3/WAV) or use live recording
2. Wait for automatic transcription
3. Click "Process Transcript" to generate study materials
4. View summary, MCQs, and flashcards
5. Export to PDF, DOCX, or JSON

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
