import numpy as np
import streamlit as st
import requests
import os
import logging
import time
import random
from dotenv import load_dotenv

# Enhanced logging for better debugging
logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG for more detailed logs
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

# Display deployment mode information
st.sidebar.info("Running on Streamlit Cloud")

# Add a debug expander
with st.expander("Debug Information"):
    st.markdown("If you're experiencing issues, this section provides diagnostic information.")
    st.markdown("### Troubleshooting Steps:")
    st.markdown("1. Make sure your browser allows microphone access")
    st.markdown("2. Try using Chrome or Firefox instead of Safari")
    st.markdown("3. Ensure no other applications are using your microphone")
    st.markdown("4. Speak clearly and loudly into your microphone")
    debug_info = st.empty()

# History management (set up early)
if "history" not in st.session_state:
    st.session_state.history = []

# Add a retry function for API calls
def call_huggingface_api_with_retry(api_url, headers, data=None, json_data=None, max_retries=3):
    """Call Hugging Face API with retry logic for handling 503 errors"""
    retry_count = 0
    backoff_time = 1  # Start with 1 second backoff
    
    while retry_count < max_retries:
        try:
            if json_data:
                response = requests.post(api_url, headers=headers, json=json_data)
            else:
                response = requests.post(api_url, headers=headers, data=data)
            
            # If success or not a 503 error, return immediately
            if response.status_code != 503:
                return response
            
            # Log the retry attempt
            retry_count += 1
            logger.warning(f"Received 503 error from API. Retry {retry_count}/{max_retries}")
            
            if retry_count < max_retries:
                # Add jitter to backoff time (avoid thundering herd problem)
                sleep_time = backoff_time + random.uniform(0, 0.5)
                logger.info(f"Waiting {sleep_time:.2f} seconds before retry...")
                time.sleep(sleep_time)
                # Exponential backoff
                backoff_time *= 2
            else:
                logger.error("Max retries reached. API might be unavailable.")
                return response
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                # Create a custom response to indicate network error
                class ErrorResponse:
                    def __init__(self):
                        self.status_code = 0
                        self.text = str(e)
                return ErrorResponse()
    
    # This should never be reached, but just in case
    class FinalErrorResponse:
        def __init__(self):
            self.status_code = 500
            self.text = "Maximum retries exceeded"
    return FinalErrorResponse()

try:
    # Create a simple audio recorder option
    def use_simple_audio_recorder():
        try:
            # Try to import audiorecorder, but provide a more graceful fallback if it fails
            try:
                from audiorecorder import AudioRecorder
                st.subheader("Audio Recorder")
                st.write("Record audio to get a diagnosis:")
                
                recorder = AudioRecorder(text="Click to record", recording_color="#e8b62c", neutral_color="#6aa36f")
                audio = recorder()
                
                if len(audio) > 0:
                    # Save to file
                    audio.export("simple_audio.wav", format="wav")
                    st.audio(audio.export().read())
                    st.success("✅ Audio recorded successfully!")
                    
                    # Process with API
                    with open("simple_audio.wav", "rb") as f:
                        audio_bytes = f.read()
                    
                    file_size = len(audio_bytes) / 1024
                    st.info(f"Audio file size: {file_size:.2f} KB")
                    
                    if st.button("Process this recording"):
                        # Call API with the audio data
                        with st.spinner("Processing audio... (this may take a moment)"):
                            response = call_huggingface_api_with_retry(API_URL_RECOGNITION, headers, data=audio_bytes)
                            if response.status_code == 200:
                                text = response.json().get("text", "Speech recognition failed")
                                st.success(f"You said: {text}")
                                
                                # Diagnosis
                                response = call_huggingface_api_with_retry(
                                    DIAGNOSTIC_MODEL_API, 
                                    headers, 
                                    json_data={"inputs": [text]}
                                )
                                
                                try:
                                    if response.status_code == 200:
                                        result = response.json()[0]['generated_text']
                                        st.success(f"🧠 Diagnosis: {result}")
                                        
                                        # Update history
                                        st.session_state.history.append({"message": text, "is_user": True})
                                        st.session_state.history.append({"message": result, "is_user": False})
                                    else:
                                        st.error(f"Diagnosis API error: {response.status_code}")
                                        st.error("The Hugging Face service is currently unavailable. Please try again later.")
                                except Exception as e:
                                    st.error(f"Diagnosis failed: {str(e)}")
                            else:
                                st.error(f"API error: {response.status_code}")
                                st.error("The Hugging Face service is currently unavailable. Please try again later.")
                                st.text(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                return audio
            except ImportError:
                # If audiorecorder package is not available
                st.warning("The audiorecorder package is not available. Please try the text input method instead.")
                st.info("You can upload an audio file manually if you have one:")
                
                uploaded_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"])
                if uploaded_file is not None:
                    st.audio(uploaded_file)
                    
                    if st.button("Process uploaded audio"):
                        with st.spinner("Processing audio... (this may take a moment)"):
                            response = call_huggingface_api_with_retry(API_URL_RECOGNITION, headers, data=uploaded_file.getvalue())
                            if response.status_code == 200:
                                text = response.json().get("text", "Speech recognition failed")
                                st.success(f"Recognized text: {text}")
                                
                                # Diagnosis
                                response = call_huggingface_api_with_retry(
                                    DIAGNOSTIC_MODEL_API, 
                                    headers, 
                                    json_data={"inputs": [text]}
                                )
                                
                                try:
                                    if response.status_code == 200:
                                        result = response.json()[0]['generated_text']
                                        st.success(f"🧠 Diagnosis: {result}")
                                        
                                        # Update history
                                        st.session_state.history.append({"message": text, "is_user": True})
                                        st.session_state.history.append({"message": result, "is_user": False})
                                    else:
                                        st.error(f"Diagnosis API error: {response.status_code}")
                                        st.error("The Hugging Face service is currently unavailable. Please try again later.")
                                        st.info("Response details:")
                                        st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                                except Exception as e:
                                    st.error(f"Diagnosis failed: {str(e)}")
                            else:
                                st.error(f"API error: {response.status_code}")
                                st.error("The Hugging Face service is currently unavailable. Please try again later.")
                                st.info("Response details:")
                                st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                
                return None
                
        except Exception as e:
            logger.error(f"Error in simple audio recorder: {str(e)}")
            st.error(f"Unable to use the simple audio recorder: {str(e)}")
            return None
    
    # Create a tab layout for audio recorder and text input
    tab1, tab2 = st.tabs(["🔊 Audio Recorder", "⌨️ Text Input"])
    
    with tab1:
        st.subheader("Record or upload audio for diagnosis")
        simple_audio = use_simple_audio_recorder()
        
        # Add the file uploader in its own section
        st.markdown("---")
        st.subheader("Or upload audio directly")
        st.info("You can upload a pre-recorded audio file here:")
        
        uploaded_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"], key="direct_upload")
        if uploaded_file is not None:
            st.audio(uploaded_file)
            if st.button("Analyze uploaded audio", key="process_upload"):
                with st.spinner("Processing audio... (this may take a moment)"):
                    try:
                        response = call_huggingface_api_with_retry(
                            API_URL_RECOGNITION, 
                            headers, 
                            data=uploaded_file.getvalue()
                        )
                        if response.status_code == 200:
                            text = response.json().get("text", "Speech recognition failed")
                            st.success(f"You said: `{text}`")
                            
                            # Diagnosis
                            response = call_huggingface_api_with_retry(
                                DIAGNOSTIC_MODEL_API, 
                                headers, 
                                json_data={"inputs": [text]}
                            )
                            
                            if response.status_code == 200:
                                result = response.json()[0]['generated_text']
                                st.success(f"🧠 Diagnosis: {result}")
                                
                                # Update history
                                st.session_state.history.append({"message": text, "is_user": True})
                                st.session_state.history.append({"message": result, "is_user": False})
                            else:
                                st.error(f"Diagnosis API error: {response.status_code}")
                                st.error("The Hugging Face service is currently unavailable. Please try again later.")
                                st.info("Response details:")
                                st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                        else:
                            st.error(f"Speech recognition API error: {response.status_code}")
                            st.error("The Hugging Face service is currently unavailable. Please try again later.")
                            st.info("Response details:")
                            st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
    
    with tab2:
        st.subheader("Type your symptoms")
        # Text input as alternative
        text_input = st.text_input("Describe your symptoms here:")
        
        if text_input and st.button("Analyze Text"):
            with st.spinner("Processing..."):
                response = call_huggingface_api_with_retry(
                    DIAGNOSTIC_MODEL_API, 
                    headers, 
                    json_data={"inputs": [text_input]}
                )
                
                try:
                    result = response.json()[0]['generated_text']
                    st.success(f"🧠 Diagnosis: {result}")
                    
                    # Update history
                    st.session_state.history.append({"message": text_input, "is_user": True})
                    st.session_state.history.append({"message": result, "is_user": False})
                except Exception as e:
                    st.error(f"Diagnosis failed: {str(e)}")
                
except Exception as e:
    st.error(f"⚠️ Error: {str(e)}")
    logger.error(f"Error: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    st.info("Falling back to text input mode")
    
    # Text input as fallback
    text_input = st.text_input("Type your symptoms here:")
    
    if text_input and st.button("Analyze Text"):
        with st.spinner("Processing..."):
            response = call_huggingface_api_with_retry(
                DIAGNOSTIC_MODEL_API, 
                headers, 
                json_data={"inputs": [text_input]}
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
