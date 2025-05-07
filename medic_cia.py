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
# Fixed API endpoint format
DIAGNOSTIC_MODEL_API = "https://api-inference.huggingface.co/pipeline/text-generation/shanover/medbot_godel_v3"
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

# Add a retry function for API calls with improved error handling
def call_huggingface_api_with_retry(api_url, headers, data=None, json_data=None, max_retries=5):
    """Call Hugging Face API with retry logic for handling 503 errors"""
    retry_count = 0
    backoff_time = 1  # Start with 1 second backoff
    
    while retry_count < max_retries:
        try:
            if json_data:
                response = requests.post(api_url, headers=headers, json=json_data, timeout=30)
            else:
                response = requests.post(api_url, headers=headers, data=data, timeout=30)
            
            # If success, return immediately
            if response.status_code == 200:
                return response
                
            # For 503 errors, retry with backoff
            if response.status_code == 503:
                retry_count += 1
                logger.warning(f"Received 503 error from API. Retry {retry_count}/{max_retries}")
                
                if retry_count < max_retries:
                    sleep_time = backoff_time + random.uniform(0, 1.0)
                    message = f"API temporarily unavailable. Retrying in {sleep_time:.1f} seconds... ({retry_count}/{max_retries})"
                    logger.info(message)
                    st.warning(message)
                    time.sleep(sleep_time)
                    backoff_time *= 2
                else:
                    logger.error("Max retries reached. API might be unavailable.")
                    return response
            elif response.status_code == 404:
                logger.error(f"Model not found (404): {api_url}")
                st.error(f"Model not found. Please check if the model is available at Hugging Face.")
                return response
            else:
                logger.error(f"API error: {response.status_code}")
                return response
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                sleep_time = backoff_time + random.uniform(0, 0.5)
                message = f"Network error. Retrying in {sleep_time:.1f} seconds... ({retry_count}/{max_retries})"
                logger.info(message)
                st.warning(message)
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                class ErrorResponse:
                    def __init__(self):
                        self.status_code = 0
                        self.text = str(e)
                return ErrorResponse()
    
    class FinalErrorResponse:
        def __init__(self):
            self.status_code = 500
            self.text = "Maximum retries exceeded"
    return FinalErrorResponse()

# Modified function for diagnosis API calls
def call_diagnosis_api(text, max_retries=3):
    """Call the diagnosis API with proper format and handling"""
    url = DIAGNOSTIC_MODEL_API
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        "options": {
            "use_cache": True
        }
    }
    
    retry_count = 0
    backoff_time = 1
    
    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()[0]["generated_text"]
                # Sometimes the result includes the input, remove it
                if result.startswith(text):
                    result = result[len(text):].strip()
                return {"success": True, "result": result}
            elif response.status_code == 503:
                retry_count += 1
                logger.warning(f"Model still loading (503). Retry {retry_count}/{max_retries}")
                
                if retry_count < max_retries:
                    sleep_time = backoff_time + random.uniform(0, 1.0)
                    message = f"Model is loading. Retrying in {sleep_time:.1f} seconds... ({retry_count}/{max_retries})"
                    logger.info(message)
                    st.warning(message)
                    time.sleep(sleep_time)
                    backoff_time *= 2
                else:
                    return {"success": False, "status_code": response.status_code, "error": "Model still loading after max retries."}
            else:
                logger.error(f"API error: {response.status_code}")
                return {"success": False, "status_code": response.status_code, "error": response.text}
                
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                sleep_time = backoff_time + random.uniform(0, 0.5)
                message = f"Network error. Retrying in {sleep_time:.1f} seconds... ({retry_count}/{max_retries})"
                logger.info(message)
                st.warning(message)
                time.sleep(sleep_time)
                backoff_time *= 2
            else:
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "Maximum retries exceeded"}

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
                            st.info("Step 1/2: Transcribing audio... (might take up to 30 seconds)")
                            response = call_huggingface_api_with_retry(API_URL_RECOGNITION, headers, data=audio_bytes)
                            if response.status_code == 200:
                                text = response.json().get("text", "Speech recognition failed")
                                st.success(f"You said: {text}")
                                
                                # Diagnosis using the new function
                                st.info("Step 2/2: Generating diagnosis...")
                                diagnosis_result = call_diagnosis_api(text)
                                
                                if diagnosis_result["success"]:
                                    result = diagnosis_result["result"]
                                    st.success(f"🧠 Diagnosis: {result}")
                                    
                                    # Update history
                                    st.session_state.history.append({"message": text, "is_user": True})
                                    st.session_state.history.append({"message": result, "is_user": False})
                                else:
                                    error_msg = diagnosis_result.get("error", "Unknown error")
                                    status_code = diagnosis_result.get("status_code", "")
                                    st.error(f"Diagnosis API error: {status_code}")
                                    st.error(f"The model could not process your request: {error_msg}")
                                    st.info("Please try the text input method instead.")
                            else:
                                st.error(f"API error: {response.status_code}")
                                st.error("The Hugging Face service is currently unavailable. Please try again in a few minutes.")
                                
                                if "Service Unavailable" in response.text:
                                    st.warning("This is likely a temporary issue with the Hugging Face API servers.")
                                    st.info("Alternative: Try using the text input method instead while the service is unavailable.")
                                
                                st.info("Response details:")
                                st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                return audio
            except ImportError:
                # If audiorecorder package is not available
                st.warning("The audiorecorder package is not available.")
                return None
                
        except Exception as e:
            logger.error(f"Error in simple audio recorder: {str(e)}")
            st.error(f"Unable to use the simple audio recorder: {str(e)}")
            return None
    
    # Create a tab layout for audio recorder and text input
    tab1, tab2 = st.tabs(["🔊 Audio", "⌨️ Text Input"])
    
    with tab1:
        st.subheader("Audio Processing")
        
        # First try the audio recorder if available
        audio_recorder_result = use_simple_audio_recorder()
        
        # Audio file uploader (shown regardless of recorder availability)
        st.markdown("---")
        st.subheader("Upload Audio File")
        st.write("Upload a pre-recorded audio file for diagnosis:")
        
        uploaded_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"])
        if uploaded_file is not None:
            st.audio(uploaded_file)
            if st.button("Analyze uploaded audio"):
                with st.spinner("Processing audio... (this may take a moment)"):
                    try:
                        st.info("Step 1/2: Transcribing audio... (might take up to 30 seconds)")
                        
                        # First attempt with longer timeout
                        response = call_huggingface_api_with_retry(
                            API_URL_RECOGNITION, 
                            headers, 
                            data=uploaded_file.getvalue()
                        )
                        
                        if response.status_code == 200:
                            text = response.json().get("text", "Speech recognition failed")
                            st.success(f"You said: `{text}`")
                            
                            # Diagnosis using the new function
                            st.info("Step 2/2: Generating diagnosis...")
                            diagnosis_result = call_diagnosis_api(text)
                            
                            if diagnosis_result["success"]:
                                result = diagnosis_result["result"]
                                st.success(f"🧠 Diagnosis: {result}")
                                
                                # Update history
                                st.session_state.history.append({"message": text, "is_user": True})
                                st.session_state.history.append({"message": result, "is_user": False})
                            else:
                                error_msg = diagnosis_result.get("error", "Unknown error")
                                status_code = diagnosis_result.get("status_code", "")
                                st.error(f"Diagnosis API error: {status_code}")
                                st.error(f"The model could not process your request: {error_msg}")
                                st.info("Please try the text input method instead.")
                        else:
                            st.error(f"Speech recognition API error: {response.status_code}")
                            st.error("The Hugging Face service is currently unavailable. Please try again in a few minutes.")
                            
                            if "Service Unavailable" in response.text:
                                st.warning("This is likely a temporary issue with the Hugging Face API servers.")
                                st.info("Alternative: Try using the text input method instead while the service is unavailable.")
                            
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
                # Use the new function for text diagnosis
                diagnosis_result = call_diagnosis_api(text_input)
                
                if diagnosis_result["success"]:
                    result = diagnosis_result["result"]
                    st.success(f"🧠 Diagnosis: {result}")
                    
                    # Update history
                    st.session_state.history.append({"message": text_input, "is_user": True})
                    st.session_state.history.append({"message": result, "is_user": False})
                else:
                    error_msg = diagnosis_result.get("error", "Unknown error")
                    status_code = diagnosis_result.get("status_code", "")
                    st.error(f"Diagnosis API error: {status_code}")
                    st.error(f"The model could not process your request: {error_msg}")
                    
                    if "404" in str(status_code):
                        st.error("The diagnosis model is not available. Please check if the model is accessible on Hugging Face.")
                    else:
                        st.info("Please try again later or try rephrasing your symptoms.")

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
            # Use the new function for text diagnosis
            diagnosis_result = call_diagnosis_api(text_input)
            
            if diagnosis_result["success"]:
                result = diagnosis_result["result"]
                st.success(f"🧠 Diagnosis: {result}")
                
                # Update history
                st.session_state.history.append({"message": text_input, "is_user": True})
                st.session_state.history.append({"message": result, "is_user": False})
            else:
                error_msg = diagnosis_result.get("error", "Unknown error")
                status_code = diagnosis_result.get("status_code", "")
                st.error(f"Diagnosis API error: {status_code}")
                st.error(f"The model could not process your request: {error_msg}")
                
                if "404" in str(status_code):
                    st.error("The diagnosis model is not available. Please check if the model is accessible on Hugging Face.")
                else:
                    st.info("Please try again later or try rephrasing your symptoms.")

# Display chat history
st.subheader("Consultation History")
for chat in st.session_state.history:
    st.write(f"{'👤' if chat['is_user'] else '🤖'}: {chat['message']}")

# Add an offline mode toggle in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("API Status Settings")
use_offline_mode = st.sidebar.checkbox("Enable offline/mock mode", value=False, 
                                       help="Enable this if Hugging Face APIs are down. This will use mock responses instead.")

if use_offline_mode:
    st.sidebar.warning("⚠️ Offline mode enabled - using mock responses for demo purposes")
    
    # Keep original call_huggingface_api_with_retry for speech recognition
    original_api_call = call_huggingface_api_with_retry
    
    # Override just the diagnosis function
    def call_diagnosis_api(text, max_retries=3):
        """Mock version that returns fake successful responses"""
        time.sleep(1)  # Simulate API delay
        
        if "stomachache" in text.lower() or "pain in my abdomen" in text.lower():
            return {
                "success": True, 
                "result": "Based on your symptoms of stomachache and abdominal pain, you may have gastritis or indigestion. This could be due to dietary factors, stress, or minor infections. I recommend avoiding spicy and fatty foods, taking over-the-counter antacids, and staying hydrated. If the pain is severe, persistent, or accompanied by fever or vomiting, please consult a healthcare provider immediately as it could indicate a more serious condition."
            }
        else:
            return {
                "success": True,
                "result": "Based on your symptoms, you may have a common digestive issue. I recommend rest, staying hydrated, and avoiding irritating foods. If symptoms persist for more than 48 hours or worsen, please consult a healthcare provider for proper evaluation and treatment."
            }

    # Override the API call function to return mock responses
    def call_huggingface_api_with_retry(api_url, headers, data=None, json_data=None, max_retries=5):
        """Mock version that returns fake successful responses"""
        time.sleep(1)  # Simulate API delay
        
        if API_URL_RECOGNITION in api_url:
            # Speech recognition mock
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self._json = {"text": "I have been having a persistent cough and fever for the past three days."}
                def json(self):
                    return self._json
            return MockResponse()
        elif DIAGNOSTIC_MODEL_API in api_url:
            # Diagnosis mock
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self._json = [{"generated_text": "Based on your symptoms of persistent cough and fever for three days, you may have a respiratory infection. It could be a viral infection like a cold or flu, or possibly COVID-19. I recommend rest, fluids, and over-the-counter fever reducers. Please consult a healthcare provider if symptoms worsen or you develop difficulty breathing."}]
                def json(self):
                    return self._json
            return MockResponse()
        else:
            # Default mock
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self._json = {"result": "mock response"}
                def json(self):
                    return self._json
            return MockResponse() 
