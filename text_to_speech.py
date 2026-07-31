"""
Text to Speech module — Kokoro backend
----------------------------------------
Drop this file into your app's `modules/` folder alongside your existing
STT / diarization / noise-cancellation modules, then call
`render_tts_tab()` from your main Streamlit app.

Voices:
    - af_heart  (female)
    - am_adam   (male)

Handles up to 10,000 words. Long text is automatically split into
sentence-safe chunks, synthesized one by one, and stitched into a single
audio file. There is no artificial time limit — a 10,000-word input will
simply produce a longer audio file than a 200-word input.

Install (one-time):
    pip install kokoro soundfile numpy --break-system-packages
"""

import io
import re
import time

import numpy as np
import soundfile as sf
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_WORDS = 10_000
SAMPLE_RATE = 24_000

VOICES = {
    "Female (af_heart)": "af_heart",
    "Male (am_adam)": "am_adam",
}

# Kokoro's own chunker works best on shorter spans (roughly one paragraph
# or a few sentences at a time). We split on sentence boundaries and then
# group sentences into chunks under this rough character budget, so the
# model doesn't choke on very long single chunks and pacing stays natural.
CHUNK_CHAR_BUDGET = 400


# ---------------------------------------------------------------------------
# Kokoro backend (loaded once, cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline(lang_code: str = "a"):
    """Load the Kokoro pipeline once per session. 'a' = American English."""
    from kokoro import KPipeline
    return KPipeline(lang_code=lang_code)


def split_into_chunks(text: str, char_budget: int = CHUNK_CHAR_BUDGET):
    """Split text into sentence-safe chunks so long input doesn't overload
    a single generation call, and so pauses land at natural sentence
    boundaries in the final audio."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= char_budget:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    return chunks


def synthesize_long_text(text: str, voice: str, progress_callback=None):
    """Generate audio for arbitrarily long text by chunking, synthesizing
    each chunk with Kokoro, and concatenating the resulting waveforms."""
    pipeline = load_pipeline()
    chunks = split_into_chunks(text)

    audio_segments = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        for _, _, audio in pipeline(chunk, voice=voice):
            audio_segments.append(audio)
        if progress_callback:
            progress_callback((i + 1) / total, i + 1, total)

    if not audio_segments:
        return np.zeros(0, dtype=np.float32)

    # Small silence gap between chunks so sentences don't run together
    gap = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
    full_audio = audio_segments[0]
    for seg in audio_segments[1:]:
        full_audio = np.concatenate([full_audio, gap, seg])

    return full_audio


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render_tts_tab():
    st.subheader("🔊 Text to Speech")
    st.caption(
        f"Type or paste up to {MAX_WORDS:,} words. Longer text produces "
        "longer audio — there's no fixed time cap."
    )

    text_input = st.text_area(
        "Text to convert",
        height=260,
        placeholder="Paste or type your text here...",
        key="tts_text_input",
    )

    voice_label = st.selectbox("Voice", list(VOICES.keys()), key="tts_voice_select")
    voice_id = VOICES[voice_label]

    words = word_count(text_input) if text_input else 0
    word_color = "red" if words > MAX_WORDS else "gray"
    st.markdown(
        f"<span style='color:{word_color}'>{words:,} / {MAX_WORDS:,} words</span>",
        unsafe_allow_html=True,
    )

    generate_disabled = not text_input.strip() or words > MAX_WORDS

    if words > MAX_WORDS:
        st.error(
            f"Text exceeds the {MAX_WORDS:,}-word limit by {words - MAX_WORDS:,} "
            "words. Please trim it before generating."
        )

    if st.button("🎙️ Generate Speech", disabled=generate_disabled, type="primary"):
        progress_bar = st.progress(0.0, text="Starting...")
        status = st.empty()
        start_time = time.time()

        def update_progress(fraction, done, total):
            progress_bar.progress(
                fraction, text=f"Synthesizing chunk {done}/{total}..."
            )

        with st.spinner("Generating audio — this may take a while for long text..."):
            audio = synthesize_long_text(
                text_input, voice_id, progress_callback=update_progress
            )

        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text="Done")
        duration_sec = len(audio) / SAMPLE_RATE

        status.success(
            f"Generated {duration_sec:.0f}s of audio from {words:,} words "
            f"in {elapsed:.1f}s."
        )

        wav_bytes = audio_to_wav_bytes(audio)

        st.audio(wav_bytes, format="audio/wav")
        st.download_button(
            label="⬇️ Download .wav",
            data=wav_bytes,
            file_name="speech_output.wav",
            mime="audio/wav",
        )


# ---------------------------------------------------------------------------
# Standalone run (for testing this module on its own)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Text to Speech", page_icon="🔊", layout="centered")
    render_tts_tab()
