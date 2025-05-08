import numpy as np
import streamlit as st
import requests
import os
import logging
import time
import json

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

# Try a more reliable medical model since the original one might be having issues
# Original: DIAGNOSTIC_MODEL_API = "https://api-inference.huggingface.co/models/shanover/medbot_godel_v3"
DIAGNOSTIC_MODEL_API = "https://api-inference.huggingface.co/models/m42-health/Llama3-Med42-8B"

# Attempt to get API key from secrets or environment variables
try:
    api_key = st.secrets["HUGGINGFACE_API_KEY"]
except Exception as e:
    # Fallback to environment variable
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        st.error("⚠️ HUGGINGFACE_API_KEY not found in secrets or environment variables.")

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
    
    # For debugging
    request_type = "data" if data else "json"
    logger.info(f"Making API request to {url} with {request_type}")
    
    while retry_count < max_retries:
        try:
            if data:
                logger.info(f"Sending data request, size: {len(data) if data else 'None'}")
                response = requests.post(url, headers=headers, data=data, timeout=30)
            elif json:
                logger.info(f"Sending JSON request: {str(json)[:100]}...")
                response = requests.post(url, headers=headers, json=json, timeout=30)
            else:
                raise ValueError("Either 'data' or 'json' must be provided for the API request.")

            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 503:
                logger.warning("Model is loading... waiting to retry")
                retry_count += 1
                time.sleep(backoff_time * 2)  # Double wait time for model loading
                backoff_time *= 2
                continue
                
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return None, f"API error: {response.status_code} - {response.text[:100]}"
            
            try:
                return response.json(), None
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {response.text[:100]}")
                return None, "Response was not valid JSON"
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                continue
            return None, f"Connection error: {str(e)}"
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            return None, "Request timed out. The server might be overloaded."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            return None, f"Network error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None, f"An unexpected error occurred: {str(e)}"
    
    return None, "Maximum retry attempts reached"

# Function to handle diagnosis - moved outside of the main code for reusability
def get_diagnosis(input_text):
    # Format prompt for the new model
    prompt = {
        "inputs": [
            {
                "role": "system",
                "content": "You are a medical assistant. Provide a brief diagnosis based on the symptoms described."
            },
            {
                "role": "user", 
                "content": f"Based on these symptoms, what might be the condition and what precautions should I take? Symptoms: {input_text}"
            }
        ]
    }
    
    response_json, error = make_api_request(DIAGNOSTIC_MODEL_API, headers, json=prompt)
    
    if error:
        st.error(error)
        return "Diagnosis failed."
    
    try:
        # Handle different response formats based on the model
        if isinstance(response_json, list) and 'generated_text' in response_json[0]:
            # Original model format
            result = response_json[0]['generated_text']
        elif isinstance(response_json, dict) and 'generated_text' in response_json:
            # Some models return this format
            result = response_json['generated_text']
        else:
            # Newer models might return a different format
            logger.info(f"Response format: {str(response_json)[:200]}")
            result = str(response_json)
            # Try to extract relevant content
            if isinstance(response_json, list):
                for item in response_json:
                    if isinstance(item, dict) and 'content' in item:
                        result = item['content']
                        break
        
        # Clean up the result for better presentation
        if "Doctor: Based on your symptoms, " in result:
            parts = result.split("Doctor: Based on your symptoms, ")
            if len(parts) > 1:
                result = "Based on your symptoms, " + parts[1]
                
        return result
    except Exception as e:
        logger.error(f"Error parsing diagnosis response: {str(e)}")
        logger.error(f"Raw response: {str(response_json)[:500]}")
        return f"Diagnosis failed: {str(e)}"

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
                        if isinstance(response_json, dict) and 'text' in response_json:
                            text = response_json.get("text", "Speech recognition failed")
                        else:
                            logger.warning(f"Unexpected response format: {str(response_json)[:100]}")
                            text = str(response_json)

                    st.write(f"🗣 You said: `{text}`")

                    # Diagnosis
                    result = get_diagnosis(text)
                    st.success(f"🧠 Diagnosis: {result}")

                    # Update history
                    st.session_state.history.append({"message": text, "is_user": True})
                    st.session_state.history.append({"message": result, "is_user": False})
                else:
                    st.warning("⚠️ No audio captured yet. Try speaking into the mic.")
                    
except Exception as e:
    st.error(f"⚠️ WebRTC error: {str(e)}")
    st.info("Falling back to text input mode")

# Text input as fallback (always available)
st.markdown("### Text Input Mode")
st.markdown("If audio mode isn't working, you can type your symptoms below:")

text_input = st.text_input("Type your symptoms here:")

if text_input and st.button("Analyze Text"):
    with st.spinner("Processing..."):
        result = get_diagnosis(text_input)
        st.success(f"🧠 Diagnosis: {result}")
        
        # Update history
        st.session_state.history.append({"message": text_input, "is_user": True})
        st.session_state.history.append({"message": result, "is_user": False})

# Display chat history
st.subheader("Consultation History")
for chat in st.session_state.history:
    st.write(f"{'👤' if chat['is_user'] else '🤖'}: {chat['message']}") 
