import numpy as np
import streamlit as st
import requests
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config first (must be at the top)
st.set_page_config(
    page_title="Voice-Based Medical Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# Hugging Face API settings
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
DIAGNOSTIC_MODEL_API = "https://api-inference.huggingface.co/models/shanover/medbot_godel_v3"
api_key = os.getenv('HUGGINGFACE_API_KEY')
headers = {"Authorization": f"Bearer {api_key}"}

# Check for API key
if not api_key:
    st.error("⚠️ Please set your HUGGINGFACE_API_KEY in the Streamlit Cloud secrets.")
    st.stop()

st.title("🩺 Voice-Based Medical Diagnosis")

# History management (set up early)
if "history" not in st.session_state:
    st.session_state.history = []

try:
    # Import WebRTC components
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    import av
    import wave
    
    # Display success message if imports work
    st.success("✅ WebRTC successfully loaded!")
    
    # Audio processor class - fixed to handle the 'name' attribute issue
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            logger.info("AudioProcessor initialized")
        
        def recv(self, frame):
            try:
                # Handle the frame correctly regardless of its type
                if hasattr(frame, 'to_ndarray'):
                    audio = frame.to_ndarray()
                    self.frames.append(audio)
                    return av.AudioFrame.from_ndarray(audio, layout="mono")
                else:
                    logger.warning(f"Received unexpected frame type: {type(frame)}")
                    return frame
            except Exception as e:
                logger.error(f"Error in recv: {str(e)}")
                return frame
        
        def save_wav(self, path="audio.wav"):
            try:
                if not self.frames:
                    logger.warning("No frames captured")
                    return False
                audio_data = np.concatenate(self.frames)
                wf = wave.open(path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())
                wf.close()
                logger.info(f"Audio saved to {path}")
                return True
            except Exception as e:
                logger.error(f"Error saving audio: {str(e)}")
                return False
    
    # Define the WebRTC streamer with simpler configuration
    webrtc_ctx = webrtc_streamer(
        key="audio_only",
        mode="sendonly",
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True},
        audio_processor_factory=AudioProcessor,
    )
    
    # Process audio with WebRTC
    if webrtc_ctx.audio_processor:
        if st.button("📝 Analyze", key="webrtc_analyze"):
            with st.spinner("Processing audio..."):
                saved = webrtc_ctx.audio_processor.save_wav()
                if saved:
                    st.success("✅ Audio recorded successfully.")
                    
                    # Speech recognition
                    with open("audio.wav", "rb") as f:
                        data = f.read()
                    
                    response = requests.post(API_URL_RECOGNITION, headers=headers, data=data)
                    if response.status_code != 200:
                        st.error(f"API error: {response.status_code}")
                        st.text(response.text)
                        text = "Speech recognition failed"
                    else:
                        text = response.json().get("text", "Speech recognition failed")
                    
                    st.write(f"🗣 You said: `{text}`")
                    
                    # Diagnosis
                    response = requests.post(
                        DIAGNOSTIC_MODEL_API, 
                        headers=headers, 
                        json={"inputs": [text]}
                    )
                    
                    try:
                        result = response.json()[0]['generated_text']
                    except Exception as e:
                        logger.error(f"Error in diagnosis: {str(e)}")
                        result = "Diagnosis failed."
                    
                    st.success(f"🧠 Diagnosis: {result}")
                    
                    # Update history
                    st.session_state.history.append({"message": text, "is_user": True})
                    st.session_state.history.append({"message": result, "is_user": False})
                else:
                    st.warning("⚠️ No audio captured yet. Try speaking into the mic.")
                    
except Exception as e:
    st.error(f"⚠️ WebRTC error: {str(e)}")
    st.info("Falling back to text input mode")
    
    # Text input as fallback
    text_input = st.text_input("Type your symptoms here:")
    
    if text_input and st.button("Analyze Text"):
        with st.spinner("Processing..."):
            response = requests.post(
                DIAGNOSTIC_MODEL_API, 
                headers=headers, 
                json={"inputs": [text_input]}
            )
            
            try:
                result = response.json()[0]['generated_text']
                st.success(f"🧠 Diagnosis: {result}")
                
                # Update history
                st.session_state.history.append({"message": text_input, "is_user": True})
                st.session_state.history.append({"message": result, "is_user": False})
            except Exception as e:
                st.error(f"Diagnosis failed: {str(e)}")

# Display chat history
st.subheader("Consultation History")
for chat in st.session_state.history:
    st.write(f"{'👤' if chat['is_user'] else '🤖'}: {chat['message']}") 
