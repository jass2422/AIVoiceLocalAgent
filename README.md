# Voice Accessibility App — Text to Speech Module

A local, privacy-first text-to-speech feature built into a broader voice
accessibility app. Converts typed or spoken input into natural speech using
the open-source **Kokoro-82M** model.

**No API key. No cloud dependency. No per-word or per-character cost.**
Speech synthesis runs entirely on your own hardware — there is no
artificial limit on text length or audio duration built into the model
itself. This app currently sets a configurable ceiling of 10,000 words per
request (see `MAX_WORDS` in `modules/text_to_speech.py`) purely as a UI
safeguard against accidental huge pastes — raise it, lower it, or remove
it entirely depending on your hardware and use case.

## Features

- 🔊 **Text to Speech tab** — type or paste text and get back
  playable/downloadable audio. No fixed time limit — longer text simply
  produces longer audio. Default ceiling: 10,000 words (configurable, see
  `MAX_WORDS`).
- 🎙️ **Live Voice Agent tab** — ask a question by typing or speaking, get
  a spoken reply back. Audio plays in chunks as it's generated, so you
  hear the start of the answer without waiting for the full response to
  finish synthesizing.
- Two voices: **af_heart** (female), **am_adam** (male)
- Runs entirely on CPU — no GPU required

## Tech stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Text-to-speech | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| Speech-to-text | Whisper *(existing module — see `modules/stt.py`)* |
| Answer generation | Python Library |

## Project structure
VoiceAgent/
├── app.py # main entry point
├── modules/
│ ├── stt.py # Whisper speech-to-text
│ ├── diarization.py # pyannote speaker diarization
│ ├── noise_cancellation.py # DeepFilterNet noise reduction
│ ├── voice_assistant_loop.py # continuous voice assistant loop
│ ├── text_to_speech.py # manual TTS tab
│ └── live_voice_agent.py # Q&A tab with chunked streaming playback
├── requirements.txt
└── README.md

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Corporate network note
If model download fails with an SSL certificate error (common behind
corporate proxies), run:
```bash
pip install --upgrade certifi pip-system-certs
```

## Running

```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Configuration needed before full use

Two functions in `modules/live_voice_agent.py` are stubbed and need wiring:

1. **`transcribe_audio()`** — connect to your existing Whisper STT function
   in `modules/stt.py`.
2. **`generate_answer()`** — connect to your TCS GenAI Lab endpoint
   (`genailab.tcs.in`) once available. Currently returns a placeholder
   response so the pipeline can be tested end-to-end.

## Roadmap

- [ ] Wire live voice agent to GenAI Lab endpoint
- [ ] Wire live voice agent to existing Whisper STT module
- [ ] True low-latency audio streaming (separate lightweight server) if
      chunked playback isn't fast enough in practice
- [ ] Additional voice options
- [ ] Multi-language support

## License

Internal / private project.
