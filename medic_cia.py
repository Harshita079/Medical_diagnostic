import numpy as np
import streamlit as st
import requests
import os
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config first (must be at the top)
st.set_page_config(
    page_title="Voice-Based Medical Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# Hugging Face API settings
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
DIAGNOSTIC_MODEL_API = "https://api-inference.huggingface.co/models/shanover/medbot_godel_v3"
api_key = st.secrets["HUGGINGFACE_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}"}

# Check for API key
if not api_key:
    st.error("⚠️ Please set your HUGGINGFACE_API_KEY in the Streamlit Cloud secrets.")
    st.stop()

st.title("🩺 Voice-Based Medical Diagnosis")

# History management (set up early)
if "history" not in st.session_state:
    st.session_state.history = []

# Enhanced error handling and logging for API requests - defined OUTSIDE the try block
def make_api_request(url, headers, data=None, json=None, max_retries=3, backoff_time=1):
    retry_count = 0
    while retry_count < max_retries:
        try:
            if data:
                response = requests.post(url, headers=headers, data=data)
            elif json:
                response = requests.post(url, headers=headers, json=json)
            else:
                raise ValueError("Either 'data' or 'json' must be provided for the API request.")

            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return None, f"API error: {response.status_code}"

            return response.json(), None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                continue
            return None, "Network error occurred. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None, "An unexpected error occurred. Please try again later."

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
                    # Fix for 'str' object has no attribute 'name' error
                    if isinstance(frame, str):
                        logger.warning(f"Received string frame instead of expected type")
                        return frame
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
    
    # Define the WebRTC streamer with simpler configuration and error handling
    try:
        webrtc_ctx = webrtc_streamer(
            key="audio_only",
            mode="sendonly",
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True},
            audio_processor_factory=AudioProcessor,
        )
    except Exception as e:
        st.error(f"WebRTC configuration error: {str(e)}")
        webrtc_ctx = None

    # Process audio with WebRTC
    if webrtc_ctx and hasattr(webrtc_ctx, 'audio_processor'):
        if st.button("📝 Analyze", key="webrtc_analyze"):
            with st.spinner("Processing audio..."):
                saved = webrtc_ctx.audio_processor.save_wav()
                if saved:
                    st.success("✅ Audio recorded successfully.")

                    with open("audio.wav", "rb") as f:
                        data = f.read()

                    response_json, error = make_api_request(API_URL_RECOGNITION, headers, data=data)
                    if error:
                        st.error(error)
                        text = "Speech recognition failed"
                    else:
                        text = response_json.get("text", "Speech recognition failed")

                    st.write(f"🗣 You said: `{text}`")

                    # Diagnosis
                    response_json, error = make_api_request(DIAGNOSTIC_MODEL_API, headers, json={"inputs": [text]})
                    if error:
                        st.error(error)
                        result = "Diagnosis failed."
                    else:
                        try:
                            result = response_json[0]['generated_text']
                            
                            # Clean up the result if needed
                            if "Doctor: Based on your symptoms, " in result:
                                parts = result.split("Doctor: Based on your symptoms, ")
                                if len(parts) > 1:
                                    result = "Based on your symptoms, " + parts[1]
                        except Exception as e:
                            logger.error(f"Error in parsing diagnosis response: {str(e)}")
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
        response_json, error = make_api_request(DIAGNOSTIC_MODEL_API, headers, json={"inputs": [text_input]})
        if error:
            st.error(error)
            result = "Diagnosis failed."
        else:
            try:
                result_json = response_json[0]['generated_text']
                result = str(result_json)
                
                # Clean up the result for GPT-2 outputs
                # Extract just the doctor's response
                if "Doctor: Based on your symptoms, " in result:
                    parts = result.split("Doctor: Based on your symptoms, ")
                    if len(parts) > 1:
                        result = "Based on your symptoms, " + parts[1]
                
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
