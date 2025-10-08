import streamlit as st
from utils.transcribe import transcribe_file, LiveTranscriber, StreamingConfig, TranscriptionResult, load_whisper, SELECTED_MODEL_SIZE
from utils.summarize import summarize_text
from utils.structure import segment_topics
from utils.quiz import extract_keywords, generate_mcqs, generate_flashcards, export_flashcards_json
from utils.exporters import build_pdf, build_docx, build_flashcards_json
import time

st.set_page_config(page_title="Lecture Voice-to-Notes Generator", layout="wide")

st.title("Lecture Voice-to-Notes Generator")
st.caption("Transcribe lectures (upload or live), summarize, generate quizzes & flashcards, and export study materials.")

@st.cache_resource(show_spinner=False)
def _load_asr():
	return load_whisper()

@st.cache_resource(show_spinner=False)
def _warmup_models():
	_ = summarize_text("This is a warmup text for summarization model.")
	return True

with st.sidebar:
	st.header("Settings")
	summary_length = st.selectbox("Summary Length", ["short","medium","detailed"], index=1)
	use_structure = st.checkbox("Generate Structured Notes (Headings)")
	fast_mode = st.checkbox("Fast Mode (smaller models)", value=True, help="Use smaller ASR model and distilled summarizer for speed.")
	model_size = st.selectbox("ASR Model Size", ["tiny","base","small","medium"], index=2 if 'small' in SELECTED_MODEL_SIZE else 1)
	beam_size = st.slider("Beam Size", 1, 5, 1, help="Higher improves accuracy but slows decoding.")
	chunk_seconds = st.number_input("Chunk Seconds (0=off)", min_value=0, max_value=600, value=0, step=10, help="Split audio into parallel chunks for faster throughput.")
	overlap_seconds = st.number_input("Overlap Seconds", min_value=0, max_value=30, value=1, step=1, help="Overlap between chunks to reduce word cuts.")
	use_vad = st.checkbox("Silence Skipping (VAD)", value=False, help="Remove low-energy frames before chunking to speed up.")
	model_ready = st.checkbox("Warmup models (optional)")
	if model_ready:
		with st.spinner("Loading models..."):
			_load_asr()
			_warmup_models()
		st.success("Models loaded.")
	st.markdown("---")
	st.caption("Tip: Medium+ models on CPU are slow. Use GPU for speed.")

tabs = st.tabs(["Upload Mode", "Live Mode", "Results & Export"])  # third tab shows last run outputs

if 'session' not in st.session_state:
	st.session_state.session = {
		'transcript': '',
		'summary': '',
		'mcqs': [],
		'flashcards': [],
		'language': None,
		'structured': []
	}

def process_pipeline(transcript: str):
	if not transcript.strip():
		st.warning("No transcript to process.")
		return
	with st.spinner("Summarizing..."):
		# Fast mode chooses distilled summarizer
		summary = summarize_text(transcript, length=summary_length, fast=fast_mode)
	if not summary:
		st.warning("Summary empty; using original transcript segment.")
		summary = transcript[:2000]
	if len(transcript.split()) > 15000:
		st.warning("Transcript is very large; consider external pre-chunking for faster processing.")
	structured_sections = []
	if use_structure:
		with st.spinner("Detecting sections..."):
			try:
				structured_sections = segment_topics(transcript)
			except Exception as e:
				st.info(f"Section detection skipped: {e}")
	with st.spinner("Generating keywords & quizzes..."):
		keywords = extract_keywords(summary or transcript, top_n=15)
		mcqs = generate_mcqs(keywords, summary or transcript, num=5)
		flashcards = generate_flashcards(keywords, summary or transcript, num=5, use_qa=True)
	st.session_state.session['summary'] = summary
	st.session_state.session['mcqs'] = mcqs
	st.session_state.session['flashcards'] = flashcards
	st.session_state.session['structured'] = structured_sections


def _build_notes_markdown(sess):
	md = ["# Lecture Notes", "## Summary", sess['summary'] or '(none)']
	if sess.get('structured'):
		md.append("## Sections")
		for sec in sess['structured']:
			md.append(f"### {sec['heading']}\n{sec['content']}")
	md.append("## Flashcards")
	for fc in sess['flashcards']:
		md.append(f"- **{fc['term']}**: {fc['definition']}")
	md.append("## MCQs")
	for q in sess['mcqs']:
		md.append(f"**Q:** {q['question']}\nOptions: " + ", ".join(q['options']) + f"\nAnswer: **{q['answer']}**")
	return "\n\n".join(md)

with tabs[0]:
	st.subheader("Upload Audio File")
	uploaded = st.file_uploader("Upload MP3 or WAV", type=["mp3","wav","m4a"])
	if uploaded:
		bytes_data = uploaded.read()
		progress = st.progress(0.0, text="Preparing transcription...")
		with st.spinner("Transcribing..."):
			try:
				if fast_mode:
					import os
					os.environ['WHISPER_MODEL_SIZE'] = model_size
				def _pb(p):
					progress.progress(min(0.999, p), text=f"Transcribing... {int(p*100)}%")
				res = transcribe_file(bytes_data, beam_size=beam_size, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds, use_vad=use_vad, progress_cb=_pb if chunk_seconds else None)
				st.session_state.session['transcript'] = res.text
				st.session_state.session['language'] = res.language
				st.success("Transcription complete.")
				progress.progress(1.0, text="Done")
				if st.button("Process Transcript", type="primary"):
					process_pipeline(res.text)
			except Exception as e:
				st.error(f"Transcription failed: {e}")
	else:
		st.info("Upload an audio file to begin.")

with tabs[1]:
	st.subheader("Live Microphone Capture")
	st.write("Start microphone capture to build a rolling transcript.")
	if 'live_text' not in st.session_state:
		st.session_state.live_text = ''
	if 'live_running' not in st.session_state:
		st.session_state.live_running = False
	placeholder = st.empty()

	def _live_callback(result: TranscriptionResult):  # pragma: no cover realtime
		# Append incremental text (naive approach: replace with full for now)
		st.session_state.live_text = result.text
		placeholder.text_area("Live Transcript", value=st.session_state.live_text, height=300)

	if st.button("Start Live" if not st.session_state.live_running else "Stop Live"):
		if not st.session_state.live_running:
			try:
				config = StreamingConfig()
				transcriber = LiveTranscriber(config, callback=_live_callback)
				st.session_state._transcriber = transcriber
				transcriber.start()
				st.session_state.live_running = True
			except Exception as e:
				st.error(f"Could not start live transcription: {e}")
		else:
			t = st.session_state.get('_transcriber')
			if t:
				t.stop()
			st.session_state.live_running = False

	placeholder.text_area("Live Transcript", value=st.session_state.live_text, height=300)
	if st.session_state.live_text:
		if st.button("Process Live Transcript", type="primary"):
			st.session_state.session['transcript'] = st.session_state.live_text
			process_pipeline(st.session_state.live_text)

with tabs[2]:
	st.subheader("Results")
	sess = st.session_state.session
	st.text_area("Transcript", value=sess['transcript'], height=250)
	st.text_area("Summarized Notes", value=sess['summary'], height=200)
	if sess.get('structured'):
		st.markdown("### Structured Sections")
		for i, sec in enumerate(sess['structured'], 1):
			with st.expander(f"{i}. {sec['heading']}"):
				st.write(sec['content'])
	if sess['mcqs']:
		st.markdown("### ❓ Multiple Choice Questions")
		st.caption(f"{len(sess['mcqs'])} quiz questions to test your understanding")
		
		for i, q in enumerate(sess['mcqs'], 1):
			with st.container():
				st.markdown(f"**Question {i}:** {q['question']}")
				
				# Display options in a cleaner format
				for j, opt in enumerate(q['options'], start=ord('A')):
					option_letter = chr(j)
					# Highlight the correct answer
					if opt == q['answer']:
						st.success(f"{option_letter}. {opt} ✓")
					else:
						st.write(f"{option_letter}. {opt}")
				
				# Show answer in an expandable section
				with st.expander("Show Answer"):
					st.write(f"**Correct Answer:** {q['answer']}")
				
				st.divider()  # Visual separator between questions
	if sess['flashcards']:
		st.markdown("### 📚 Flashcards")
		st.caption(f"{len(sess['flashcards'])} flashcards generated for study")
		
		# Display flashcards in an organized, card-like format
		for i, fc in enumerate(sess['flashcards'], 1):
			with st.container():
				# Use columns for better layout
				col1, col2 = st.columns([1, 3])
				with col1:
					st.markdown(f"**Card {i}**")
					st.info(f"**{fc['term']}**")
				with col2:
					st.markdown("**Definition:**")
					st.write(fc['definition'])
				st.divider()  # Visual separator between cards

	if sess['summary'] or sess['mcqs'] or sess['flashcards']:
		pdf_bytes = build_pdf(sess['summary'], sess['mcqs'], sess['flashcards'])
		docx_bytes = build_docx(sess['summary'], sess['mcqs'], sess['flashcards'])
		flash_json = build_flashcards_json(sess['flashcards'])
		notes_md = _build_notes_markdown(sess).encode('utf-8')
		st.download_button("Download PDF", data=pdf_bytes, file_name="lecture_notes.pdf", mime="application/pdf")
		st.download_button("Download DOCX", data=docx_bytes, file_name="lecture_notes.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
		st.download_button("Download Flashcards JSON", data=flash_json, file_name="flashcards.json", mime="application/json")
		st.download_button("Download Notes (Markdown)", data=notes_md, file_name="lecture_notes.md", mime="text/markdown")

st.caption("Note: Live streaming is pseudo-streamed in time chunks; for very long sessions consider periodic processing.")
