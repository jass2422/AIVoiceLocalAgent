import streamlit as st
from modules.text_to_speech import render_tts_tab

st.set_page_config(page_title="Voice Accessibility App", page_icon="🔊", layout="centered")
render_tts_tab()