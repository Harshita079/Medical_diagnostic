import numpy as np
import streamlit as st
import requests
import os
import logging
import time
import random
from dotenv import load_dotenv

# Enhanced logging for better debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config first (must be at the top)
st.set_page_config(
    page_title="Medical Diagnostic Assistant",
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
    st.error("⚠ Please set your HUGGINGFACE_API_KEY in the Streamlit Cloud secrets.")
    st.stop()

st.title("🩺 Medical Diagnostic Assistant")
st.markdown("### Get a quick diagnosis by describing your symptoms or uploading an audio recording")

# History management
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

# Process diagnosis helper function
def process_diagnosis(text, is_text_input=False):
    """Process diagnosis for given text and handle API response appropriately"""
    # Use text_input parameter name if this was called from text input field
    input_text = text
    
    response = call_huggingface_api_with_retry(
        DIAGNOSTIC_MODEL_API, 
        headers, 
        json_data={"inputs": [input_text]}
    )
    
    try:
        # First check if response has valid status code
        if response.status_code == 200:
            # Check if response content is valid JSON
            try:
                result_json = response.json()
                if result_json and isinstance(result_json, list) and len(result_json) > 0:
                    result = result_json[0]['generated_text']
                    st.success(f"🧠 Diagnosis: {result}")
                    
                    # Update history
                    st.session_state.history.append({"message": input_text, "is_user": True})
                    st.session_state.history.append({"message": result, "is_user": False})
                    return True, result
                else:
                    st.error("Invalid response format from the diagnosis model")
                    logger.error(f"Invalid response format: {result_json}")
                    st.info("Response details:")
                    st.code(str(result_json)[:500])
                    return False, None
            except ValueError as json_err:
                st.error(f"Failed to parse API response: {str(json_err)}")
                logger.error(f"JSON parsing error: {str(json_err)}, Response: {response.text[:100]}")
                st.info("Response details:")
                st.code(response.text[:500])
                return False, None
        else:
            st.error(f"Diagnosis API error: {response.status_code}")
            st.error("The Hugging Face service is currently unavailable. Please try again later.")
            st.info("Response details:")
            st.code(response.text[:500] if hasattr(response, 'text') else "No response text available")
            return False, None
    except Exception as e:
        st.error(f"Diagnosis failed: {str(e)}")
        logger.error(f"General error in diagnosis processing: {str(e)}")
        return False, None

# Create a tab layout for text input and audio upload
tab1, tab2 = st.tabs(["⌨ Text Input", "🎤 Voice Upload"])

with tab1:
    st.subheader("Describe your symptoms")
    st.markdown("""
    Please provide details about your symptoms such as:
    - What symptoms are you experiencing?
    - When did they start?
    - Are they getting better or worse?
    """)
    
    # Text input for symptoms
    text_input = st.text_area("Type your symptoms here:", height=150)
    
    if text_input and st.button("Get Diagnosis", key="text_diagnosis"):
        with st.spinner("Analyzing your symptoms..."):
            process_diagnosis(text_input, is_text_input=True)

with tab2:
    st.subheader("Upload an audio recording")
    st.markdown("""
    ### Instructions:
    1. Record yourself describing your symptoms using any recording app
    2. Save the recording as a WAV file
    3. Upload the file below
    4. Click 'Analyze Audio' to get a diagnosis
    """)
    
    # Audio file upload
    uploaded_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Analyze Audio"):
            with st.spinner("Processing audio... (this may take a moment)"):
                try:
                    response = call_huggingface_api_with_retry(
                        API_URL_RECOGNITION, 
                        headers, 
                        data=uploaded_file.getvalue()
                    )
                    
                    if response.status_code == 200:
                        text = response.json().get("text", "Speech recognition failed")
                        st.success(f"🗣 Transcription: {text}")
                        
                        # Get diagnosis based on transcribed text
                        process_diagnosis(text)
                    else:
                        st.error(f"Speech recognition API error: {response.status_code}")
                        st.error("The Hugging Face service is currently unavailable. Please try again later.")
                        st.info("Response details:")
                        st.code(response.text[:500] if hasattr(response, 'text') else "No response text available")
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    logger.error(f"Audio processing error: {str(e)}")

# Display consultation history
if st.session_state.history:
    st.subheader("Consultation History")
    history_container = st.container()
    
    with history_container:
        for chat in st.session_state.history:
            if chat["is_user"]:
                st.markdown(f"**👤 You:** {chat['message']}")
            else:
                st.markdown(f"**🤖 AI Doctor:** {chat['message']}")
    
    # Add a button to clear history
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()
else:
    st.info("Your consultation history will appear here after you get a diagnosis.")

# Add some helpful information at the bottom
st.markdown("---")
st.markdown("""
**Disclaimer:** This application provides general information only and is not a substitute for professional medical advice. 
Always consult with a qualified healthcare provider for medical diagnosis and treatment.
""")
    
