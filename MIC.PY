import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
import tempfile
import soundfile as sf

# Load Whisper model (cache to avoid reloading)
@st.cache_resource
def load_model():
    return WhisperModel("large-v3", device="cpu", compute_type="int8")

model = load_model()

# UI
# st.title("🎙️ Audio Recorder with Whisper Transcription")

# Record audio
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    # Play the recorded audio
    st.audio(audio.export().read(), format="audio/wav")

    # Show audio info
    st.write(f"Duration: {audio.duration_seconds:.2f} seconds")
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}")

    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio.export(tmpfile.name, format="wav")
        tmp_path = tmpfile.name

    # Transcribe using faster-whisper
    with st.spinner("Transcribing..."):
        segments, info = model.transcribe(tmp_path, beam_size=5)
        full_text = " ".join([seg.text for seg in segments])

    st.success("Transcription:")
    st.markdown(f"**{full_text}**")
