from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import wave
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API settings
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
DIAGNOSTIC_MODEL_API = "https://api-inference.huggingface.co/models/shanover/medbot_godel_v3"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Audio processor class to capture and save audio
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return av.AudioFrame.from_ndarray(audio, layout="mono")  # <-- Fixed: Explicit layout

    def save_wav(self, path="audio.wav"):
        if not self.frames:
            return False
        audio_data = np.concatenate(self.frames)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(audio_data.tobytes())
        wf.close()
        return True

def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_RECOGNITION, headers=headers, data=data)
    if response.status_code != 200:
        return "Speech recognition failed"
    return response.json().get("text", "Speech recognition failed")

def diagnose(text):
    response = requests.post(DIAGNOSTIC_MODEL_API, headers=headers, json={"inputs": [text]})
    try:
        return response.json()[0]['generated_text']
    except:
        return "Diagnosis failed."

# Streamlit App UI
st.set_page_config(
    page_title="Voice-Based Medical Diagnosis",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Voice-Based Medical Diagnosis")

# Check for API key
if not os.getenv('HUGGINGFACE_API_KEY'):
    st.error("⚠️ Please set your HUGGINGFACE_API_KEY in the Streamlit Cloud secrets.")
    st.stop()

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode="sendonly",
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=lambda: AudioRecorder(),
)


# History management
if "history" not in st.session_state:
    st.session_state.history = []

if webrtc_ctx.audio_processor:
    if st.button("📝 Analyze"):
        saved = webrtc_ctx.audio_processor.save_wav()
        if saved:
            st.success("✅ Audio recorded successfully.")
            text = recognize_speech("audio.wav")
            st.write(f"🗣 You said: `{text}`")
            result = diagnose(text)
            st.success(f"🧠 Diagnosis: {result}")
            st.session_state.history.append({"message": text, "is_user": True})
            st.session_state.history.append({"message": result, "is_user": False})
        else:
            st.warning("⚠️ No audio captured yet. Try speaking into the mic.")

# Display chat history
for chat in st.session_state.history:
    st.write(f"{'👤' if chat['is_user'] else '🤖'}: {chat['message']}")
