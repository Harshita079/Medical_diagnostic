import numpy as np
import streamlit as st
import requests
import os
import logging
import time
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

# Add a session state for audio frames
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# Add recording status
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Add flag for frames received
if "frames_received" not in st.session_state:
    st.session_state.frames_received = False

try:
    # Import WebRTC components
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    import av
    import wave
    
    # Display success message if imports work
    st.success("✅ WebRTC successfully loaded!")
    
    # Audio processor class - improved for better audio capturing
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            logger.info("AudioProcessor initialized")
            # Clear session state frames at initialization
            st.session_state.audio_frames = []
            st.session_state.frames_received = False
        
        def recv(self, frame):
            try:
                # Check frame type more explicitly
                if frame is None:
                    logger.warning("Received None frame")
                    return None
                elif isinstance(frame, str):
                    # This is what causes the 'str has no attribute name' error
                    logger.warning(f"Received string instead of audio frame: {frame}")
                    # Return empty audio frame instead of None
                    empty_frame = av.AudioFrame.from_ndarray(
                        np.zeros((1, 1), dtype=np.float32),
                        layout="mono"
                    )
                    return empty_frame
                elif hasattr(frame, 'to_ndarray'):
                    # Normal case - convert frame to numpy array
                    audio = frame.to_ndarray()
                    # Store in instance and session state
                    self.frames.append(audio)
                    st.session_state.audio_frames.append(audio)
                    st.session_state.frames_received = True
                    # Visual feedback that frames are being received
                    if len(self.frames) % 10 == 0:
                        logger.info(f"Received {len(self.frames)} audio frames")
                    return av.AudioFrame.from_ndarray(audio, layout="mono")
                else:
                    # Any other unexpected type
                    logger.warning(f"Received unexpected frame type: {type(frame)}")
                    # Return the frame as is to avoid errors
                    return frame
            except Exception as e:
                logger.error(f"Error in recv: {str(e)}")
                # Return empty frame on error instead of the original frame
                empty_frame = av.AudioFrame.from_ndarray(
                    np.zeros((1, 1), dtype=np.float32),
                    layout="mono"
                )
                return empty_frame
        
    def save_audio_frames(frames, path="audio.wav"):
        """Helper function to save audio frames to a WAV file"""
        try:
            if not frames:
                logger.warning("No audio frames to save")
                return False
            
            # Log the shapes for debugging
            shapes = [frame.shape for frame in frames]
            logger.info(f"Audio frame shapes: {shapes[:5]}... (total: {len(shapes)})")
            
            # Concatenate and save
            audio_data = np.concatenate(frames)
            wf = wave.open(path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_data.tobytes())
            wf.close()
            logger.info(f"Audio saved to {path} (size: {os.path.getsize(path)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            return False
    
    # Create a tab layout for WebRTC and text input
    tab1, tab2 = st.tabs(["🎤 Voice Recognition", "⌨️ Text Input"])
    
    with tab1:
        st.subheader("Talk to get a diagnosis")
        st.write("1. Click START to begin recording")
        st.write("2. Speak clearly about your symptoms")
        st.write("3. Click STOP when finished")
        st.write("4. Click ANALYZE to get a diagnosis")
        
        # Status indicators
        col1, col2 = st.columns(2)
        with col1:
            status_placeholder = st.empty()
        with col2:
            frames_info = st.empty()
        
        # Define the WebRTC streamer with proper configuration for remote connections
        webrtc_ctx = webrtc_streamer(
            key="audio_only",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=AudioProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True,
        )
        
        # Update recording status based on webrtc_ctx state
        if webrtc_ctx.state.playing:
            status_placeholder.success("🔴 Recording...")
            st.session_state.is_recording = True
            # Show frames being received
            if st.session_state.frames_received:
                frames_info.info(f"Received {len(st.session_state.audio_frames)} audio frames")
            else:
                frames_info.warning("No audio frames detected yet. Please speak louder or check your microphone.")
        else:
            if st.session_state.is_recording:  # Was recording but now stopped
                status_placeholder.info("✅ Recording stopped. Click ANALYZE to process.")
                st.session_state.is_recording = False
            else:
                status_placeholder.info("Click START to begin recording")
        
        # Process audio when user clicks Analyze
        if st.button("📝 ANALYZE", key="webrtc_analyze"):
            with st.spinner("Processing audio..."):
                # Check if we have audio frames in the session state
                if len(st.session_state.audio_frames) > 0:
                    # Save the audio from session state
                    saved = save_audio_frames(st.session_state.audio_frames)
                    
                    if saved:
                        st.success(f"✅ Audio recorded successfully ({len(st.session_state.audio_frames)} frames).")
                        
                        # Speech recognition
                        with open("audio.wav", "rb") as f:
                            data = f.read()
                        
                        # Show audio size for debugging
                        st.info(f"Audio file size: {len(data) / 1024:.2f} KB")
                        
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
                        st.warning("⚠️ Failed to save audio.")
                else:
                    st.warning("⚠️ No audio captured yet. Try speaking into the mic.")
    
    with tab2:
        st.subheader("Type your symptoms")
        # Text input as alternative
        text_input = st.text_input("Describe your symptoms here:")
        
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
