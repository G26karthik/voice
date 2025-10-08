"""Transcription utilities for Lecture Voice-to-Notes Generator.

Features:
- Load Whisper (openai/whisper-small) via transformers pipeline or whisper library fallback.
- File transcription (mp3/wav)
- Streaming microphone capture (chunked) using sounddevice (preferred) or pyaudio fallback.
- Language auto-detection.

Note: True low-latency streaming with Whisper requires segment-level processing; here we simulate streaming by buffering N seconds then transcribing.
"""
from __future__ import annotations
import io
import tempfile
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Generator, Optional, List, Callable

import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # fallback later

try:
    import whisper  # openai-whisper package
except Exception:  # pragma: no cover
    whisper = None

from transformers import pipeline
import torch

_FAST_WHISPER_AVAILABLE = False
try:  # optional faster-whisper
    from faster_whisper import WhisperModel  # type: ignore
    _FAST_WHISPER_AVAILABLE = True
except Exception:  # pragma: no cover
    WhisperModel = None  # type: ignore
import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "openai/whisper-small"

# runtime config (can be overridden by caller / UI)
SELECTED_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")  # base, small, medium, large-v2, etc.
USE_FAST_WHISPER = os.environ.get("USE_FAST_WHISPER", "1") == "1"

@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    segments: Optional[List[dict]] = None

_whisper_model = None
_pipe_asr = None
_fast_whisper_model = None


def load_whisper(model_name: str = _MODEL_NAME, prefer_fast: bool = True):
    """Load an ASR model with preference order:
    1. faster-whisper (if available & requested)
    2. transformers pipeline
    3. openai-whisper python package
    """
    global _whisper_model, _pipe_asr, _fast_whisper_model
    if _fast_whisper_model is not None and prefer_fast:
        return _fast_whisper_model
    if _pipe_asr is not None:
        return _pipe_asr
    if _whisper_model is not None:
        return _whisper_model

    # Attempt faster-whisper
    if prefer_fast and USE_FAST_WHISPER and _FAST_WHISPER_AVAILABLE:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            _fast_whisper_model = WhisperModel(SELECTED_MODEL_SIZE, device=device, compute_type=compute_type)
            logger.info(f"Loaded faster-whisper model {SELECTED_MODEL_SIZE} on {device} ({compute_type})")
            return _fast_whisper_model
        except Exception as e:
            logger.warning(f"Failed loading faster-whisper: {e}; falling back to transformers pipeline.")

    # transformers pipeline
    try:
        _pipe_asr = pipeline("automatic-speech-recognition", model=model_name, device=0 if torch.cuda.is_available() else None)
        return _pipe_asr
    except Exception as e:
        logger.warning(f"Falling back to whisper library due to pipeline error: {e}")
        if whisper is None:
            raise RuntimeError("No ASR backend available.")
        _whisper_model = whisper.load_model(model_name.split('/')[-1])  # e.g., 'small'
        return _whisper_model


def _simple_vad(samples: np.ndarray, sr: int, frame_ms: int = 30, energy_thresh: float = 0.0005) -> np.ndarray:
    """Return boolean mask of frames (kept=True) using simple short-time energy threshold."""
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0:
        return np.ones_like(samples, dtype=bool)
    frames = [samples[i:i+frame_len] for i in range(0, len(samples), frame_len)]
    energies = [float(np.mean(f**2)) for f in frames]
    keep = [e > energy_thresh for e in energies]
    expanded = np.concatenate([np.full(len(frames[i]), keep[i], dtype=bool) for i in range(len(frames))])
    return expanded[:len(samples)]

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def transcribe_file(file_bytes: bytes, model_name: str = _MODEL_NAME, suffix: str = ".mp3", beam_size: int = 1,
                    chunk_seconds: int = 0, overlap_seconds: int = 1, parallel_workers: int = 2,
                    use_vad: bool = False, progress_cb: Optional[Callable[[float], None]] = None) -> TranscriptionResult:
    """Transcribe uploaded audio bytes.

    Some transformer ASR pipelines require either a path or a numpy array. Passing a BytesIO object can raise
    a TypeError (as seen in the user's environment). To maximize compatibility:
    1. Persist bytes to a NamedTemporaryFile with appropriate suffix.
    2. Try pipeline(path) first.
    3. If that fails, load with soundfile/librosa into numpy array and retry.
    4. Fall back to whisper library interface if pipeline unavailable.
    """
    backend = load_whisper(model_name)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    # Handle chunked path early (for all backends except we reuse faster-whisper per chunk)
    if chunk_seconds and chunk_seconds > 0:
        try:
            import soundfile as sf
            audio, sr = sf.read(tmp_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if use_vad:
                mask = _simple_vad(audio, sr)
                audio = audio[mask]
            total_dur = len(audio)/sr
            step = chunk_seconds - overlap_seconds if chunk_seconds > overlap_seconds else chunk_seconds
            starts = np.arange(0, total_dur, step)
            tasks = []
            def _run_chunk(start_s: float):
                end_s = min(start_s + chunk_seconds, total_dur)
                s_idx = int(start_s * sr)
                e_idx = int(end_s * sr)
                piece = audio[s_idx:e_idx]
                # Write piece temp
                temp_seg = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                try:
                    sf.write(temp_seg.name, piece, sr)
                    if _fast_whisper_model is not None and backend is _fast_whisper_model:
                        segs, info_local = _fast_whisper_model.transcribe(temp_seg.name, beam_size=beam_size, vad_filter=False)
                        collected_local = []
                        full_text_local = []
                        for s in segs:
                            off_start = s.start + start_s
                            off_end = s.end + start_s
                            collected_local.append({"start": off_start, "end": off_end, "text": s.text})
                            full_text_local.append(s.text)
                        return collected_local, " ".join(full_text_local)
                    elif _pipe_asr is not None and backend is _pipe_asr:
                        out = _pipe_asr(temp_seg.name, return_timestamps=True)
                        segs = out.get('chunks') or out.get('segments') or []
                        collected_local = []
                        for s in segs:
                            st_s = s.get('timestamp', (0,0))[0] if isinstance(s.get('timestamp'), (list, tuple)) else s.get('start',0)
                            en_s = s.get('timestamp', (0,0))[1] if isinstance(s.get('timestamp'), (list, tuple)) else s.get('end',0)
                            collected_local.append({"start": (st_s or 0) + start_s, "end": (en_s or 0) + start_s, "text": s.get('text','')})
                        return collected_local, out.get('text','')
                    else:
                        assert whisper is not None and _whisper_model is not None
                        res = _whisper_model.transcribe(temp_seg.name, verbose=False, beam_size=beam_size)
                        segs = res.get('segments', [])
                        collected_local = []
                        for s in segs:
                            collected_local.append({"start": s.get('start',0)+start_s, "end": s.get('end',0)+start_s, "text": s.get('text','')})
                        return collected_local, res.get('text','')
                finally:
                    try: os.unlink(temp_seg.name)
                    except Exception: pass

            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                fut_map = {ex.submit(_run_chunk, float(s)): float(s) for s in starts}
                all_segments = []
                texts = []
                for idx, fut in enumerate(as_completed(fut_map)):
                    segs, t_text = fut.result()
                    all_segments.extend(segs)
                    texts.append(t_text)
                    if progress_cb:
                        progress_cb((idx+1)/len(fut_map))
            # Sort segments by start
            all_segments.sort(key=lambda x: x['start'])
            full_text = " ".join(texts).strip()
            try: os.unlink(tmp_path)
            except Exception: pass
            return TranscriptionResult(text=full_text, language=None, segments=all_segments)
        except Exception as e:
            logger.warning(f"Chunked transcription failed, reverting to single pass: {e}")

    # faster-whisper path (single pass)
    if _fast_whisper_model is not None and backend is _fast_whisper_model:
        try:
            segments, info = _fast_whisper_model.transcribe(tmp_path, beam_size=beam_size, vad_filter=use_vad)
            collected = []
            full_text_parts = []
            for seg in segments:
                collected.append({"start": seg.start, "end": seg.end, "text": seg.text})
                full_text_parts.append(seg.text)
            text = " ".join(full_text_parts).strip()
            try: os.unlink(tmp_path)
            except Exception: pass
            if progress_cb: progress_cb(1.0)
            return TranscriptionResult(text=text, language=info.language if hasattr(info, 'language') else None, segments=collected)
        except Exception as e:
            logger.warning(f"faster-whisper failed, retrying with pipeline: {e}")

    if _pipe_asr is not None and backend is _pipe_asr:
        try:
            if torch.cuda.is_available():
                # mixed precision autocast for possible speedup
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = _pipe_asr(tmp_path, return_timestamps=True)
            else:
                out = _pipe_asr(tmp_path, return_timestamps=True)
        except TypeError:
            # Attempt manual loading to numpy
            try:
                import soundfile as sf
                audio, sr = sf.read(tmp_path)
                if torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = _pipe_asr({"array": audio, "sampling_rate": sr}, return_timestamps=True)
                else:
                    out = _pipe_asr({"array": audio, "sampling_rate": sr}, return_timestamps=True)
            except Exception as e:
                logger.error(f"ASR pipeline failed after retry: {e}")
                raise RuntimeError(f"Transcription failed after retry: {e}")
        except Exception as e:
            # Generic unexpected error; escalate
            logger.exception("Unexpected ASR pipeline error")
            raise RuntimeError(f"Transcription pipeline error: {e}")
        text = out.get('text', '').strip()
        lang = out.get('language', None)
        segments = out.get('chunks') or out.get('segments')
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return TranscriptionResult(text=text, language=lang, segments=segments)

    # whisper library fallback
    assert whisper is not None and _whisper_model is not None
    result = _whisper_model.transcribe(tmp_path, verbose=False, beam_size=beam_size)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return TranscriptionResult(text=result.get('text', '').strip(), language=result.get('language'), segments=result.get('segments'))


############ Streaming Support ############
@dataclass
class StreamingConfig:
    samplerate: int = 16000
    block_duration: float = 5.0  # seconds per chunk
    max_buffer: float = 60.0  # seconds before trimming
    model_name: str = _MODEL_NAME

class LiveTranscriber:
    """Incremental pseudo-streaming transcriber.

    Strategy:
    - Maintain rolling audio buffer.
    - On each interval, transcribe buffer.
    - Diff new text vs previous best transcript and emit only the newly appended portion.
    - Accumulate canonical transcript text.
    """
    def __init__(self, config: StreamingConfig, callback: Optional[Callable[[TranscriptionResult], None]] = None):
        if sd is None:
            raise RuntimeError("sounddevice not available. Install or use file upload mode.")
        self.config = config
        self.callback = callback
        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[np.ndarray] = []
        self._model = None
        self._full_text: str = ""
        self._last_text: str = ""

    def _audio_callback(self, indata, frames, time_info, status):  # pragma: no cover - realtime
        if status:
            # Could log status
            pass
        self._audio_q.put(indata.copy())

    def start(self):  # pragma: no cover - realtime
        if self._thread is not None:
            return
        self._model = load_whisper(self.config.model_name)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        sd.InputStream(channels=1, samplerate=self.config.samplerate, callback=self._audio_callback).start()

    def stop(self):  # pragma: no cover - realtime
        self._stop.set()
        # Give time to finish
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def _loop(self):  # pragma: no cover - realtime
        last_transcribe = time.time()
        block_samples = int(self.config.block_duration * self.config.samplerate)
        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.25)
                self._buffer.append(chunk[:, 0])
            except queue.Empty:
                pass
            now = time.time()
            if now - last_transcribe >= self.config.block_duration and self._buffer:
                # Concatenate buffer
                audio = np.concatenate(self._buffer)
                # Trim if exceeds max_buffer
                max_samples = int(self.config.max_buffer * self.config.samplerate)
                if audio.shape[0] > max_samples:
                    audio = audio[-max_samples:]
                result = self._transcribe_array(audio)
                # incremental diff
                new_part = result.text[len(self._last_text):].strip()
                if new_part:
                    self._full_text += (" " if self._full_text else "") + new_part
                    incremental = TranscriptionResult(text=self._full_text, language=result.language, segments=result.segments)
                    if self.callback:
                        self.callback(incremental)
                self._last_text = result.text
                last_transcribe = now

    def _transcribe_array(self, audio: np.ndarray) -> TranscriptionResult:
        # Use pipeline if available
        if _pipe_asr is not None:
            # Provide raw array; pipeline may expect sampling rate argument in newer versions
            out = _pipe_asr(audio, return_timestamps=True)
            # Build segments if chunks exist for timing potential future use
            return TranscriptionResult(text=out.get('text','').strip(), language=out.get('language'), segments=out.get('chunks') or out.get('segments'))
        assert whisper is not None
        # whisper library expects 16k mono float32
        # Save temporary file path approach for simplicity
        temp_name = "_live_temp.wav"
        import soundfile as sf  # lightweight dependency (could add to requirements) but fallback if missing
        try:
            sf.write(temp_name, audio, self.config.samplerate)
            res = _whisper_model.transcribe(temp_name, verbose=False)
            return TranscriptionResult(text=res.get('text','').strip(), language=res.get('language'), segments=res.get('segments'))
        except Exception as e:
            return TranscriptionResult(text=f"[Streaming error: {e}]", language=None)

__all__ = [
    "TranscriptionResult",
    "StreamingConfig",
    "LiveTranscriber",
    "transcribe_file",
    "load_whisper"
]
