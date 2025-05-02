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
st.sidebar.warning("Note: WebRTC may have limited support on some cloud platforms. If voice recognition doesn't work, try the Simple Recorder or text input.")

# Browser check warning
import platform
user_agent = st.experimental_get_query_params().get('user_agent', [''])[0]
if 'Safari' in user_agent and 'Chrome' not in user_agent:
    st.warning("⚠️ Safari has known issues with WebRTC. Please try Chrome or Firefox for better results.")

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

# Add a session state for audio frames
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# Add recording status
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Add flag for frames received
if "frames_received" not in st.session_state:
    st.session_state.frames_received = False
    
# Add frame counter session state
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0

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
    # Import WebRTC components
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration, MediaStreamConstraints
    import av
    import wave
    
    # Display success message if imports work
    st.success("✅ WebRTC successfully loaded!")
    logger.info("WebRTC components successfully imported")
    
    # Audio processor class - improved for better audio capturing and debugging
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            logger.info("AudioProcessor initialized")
            # Clear session state frames at initialization
            st.session_state.audio_frames = []
            st.session_state.frames_received = False
            st.session_state.frame_counter = 0
            
            # Print debug info
            logger.debug(f"Audio processor created at {time.time()}")
        
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
                    
                    # Debug audio levels to detect very quiet audio
                    audio_level = np.abs(audio).mean()
                    logger.debug(f"Audio level: {audio_level}, shape: {audio.shape}")
                    
                    # Store in instance and session state
                    self.frames.append(audio)
                    st.session_state.audio_frames.append(audio)
                    st.session_state.frames_received = True
                    st.session_state.frame_counter += 1
                    
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
                import traceback
                logger.error(traceback.format_exc())
                # Return empty frame on error instead of the original frame
                empty_frame = av.AudioFrame.from_ndarray(
                    np.zeros((1, 1), dtype=np.float32),
                    layout="mono"
                )
                return empty_frame
    
    # Create a simple audio recorder option to fall back to simple audio recording
    def use_simple_audio_recorder():
        try:
            # Try to import audiorecorder, but provide a more graceful fallback if it fails
            try:
                from audiorecorder import AudioRecorder
                st.subheader("Alternative Audio Recorder")
                st.write("If WebRTC isn't working, try this simple recorder:")
                
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
    
    def save_audio_frames(frames, path="audio.wav"):
        """Helper function to save audio frames to a WAV file"""
        try:
            if not frames:
                logger.warning("No audio frames to save")
                return False
            
            # Log the shapes for debugging
            shapes = [frame.shape for frame in frames]
            logger.info(f"Audio frame shapes: {shapes[:5]}... (total: {len(shapes)})")
            
            # Check if frames contain actual audio (not just silence)
            audio_levels = [np.abs(frame).mean() for frame in frames]
            avg_level = sum(audio_levels) / len(audio_levels)
            logger.info(f"Average audio level: {avg_level}")
            
            if avg_level < 0.001:
                logger.warning(f"Very low audio levels detected: {avg_level}")
                st.warning("Very quiet audio detected. Please speak louder or check your microphone.")
            
            # Concatenate and save
            audio_data = np.concatenate(frames)
            wf = wave.open(path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_data.tobytes())
            wf.close()
            
            file_size = os.path.getsize(path)
            logger.info(f"Audio saved to {path} (size: {file_size} bytes)")
            
            # Extra validation check
            if file_size < 1000:  # Less than 1KB, probably empty or corrupted
                logger.warning(f"Audio file too small ({file_size} bytes), might be empty")
                if file_size == 0:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # Create a tab layout for WebRTC and text input
    tab1, tab2, tab3 = st.tabs(["🎤 Voice Recognition", "🔊 Simple Recorder", "⌨️ Text Input"])
    
    with tab1:
        st.subheader("Talk to get a diagnosis")
        st.markdown("""
        ### Instructions:
        1. Click **START** to begin recording
        2. Speak clearly about your symptoms
        3. Click **STOP** when finished
        4. Click **ANALYZE** to get a diagnosis
        
        _If nothing happens, try the Simple Recorder in the next tab._
        """)
        
        # Status indicators
        col1, col2 = st.columns(2)
        with col1:
            status_placeholder = st.empty()
        with col2:
            frames_info = st.empty()
        
        # Show microphone test option
        st.markdown("---")
        mic_test_expander = st.expander("Test your microphone")
        with mic_test_expander:
            st.write("This will help check if your browser can access your microphone:")
            if st.button("Test Microphone Access"):
                st.components.v1.html("""
                <script>
                    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                        navigator.mediaDevices.getUserMedia({ audio: true })
                            .then(function(stream) {
                                document.getElementById('mic-status').innerHTML = 
                                    '✅ Microphone access granted! Browser permissions are working correctly.';
                                stream.getTracks().forEach(track => track.stop());
                            })
                            .catch(function(error) {
                                document.getElementById('mic-status').innerHTML = 
                                    '❌ Error accessing microphone: ' + error.message;
                            });
                    } else {
                        document.getElementById('mic-status').innerHTML = 
                            '❌ getUserMedia not supported in this browser. Try Chrome or Firefox.';
                    }
                </script>
                <div id="mic-status">Testing microphone access...</div>
                """, height=100)
        
        # Enhanced WebRTC configuration for cloud deployment
        # Use comprehensive STUN/TURN servers for better NAT traversal
        # Include public TURN servers to help with restrictive networks
        rtc_configuration = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                    # Add public TURN servers that don't require credentials
                    {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"},
                    {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"},
                    {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"},
                ],
                "iceTransportPolicy": "all",
                "bundlePolicy": "max-bundle",
                "rtcpMuxPolicy": "require",
                "sdpSemantics": "unified-plan"
            }
        )
        
        # Define the WebRTC streamer with cloud-optimized configuration
        webrtc_ctx = webrtc_streamer(
            key="audio_only",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={
                "audio": {
                    "echoCancellation": {"ideal": True},
                    "noiseSuppression": {"ideal": True},
                    "autoGainControl": {"ideal": True}
                },
                "video": False
            },
            audio_processor_factory=AudioProcessor,
            rtc_configuration=rtc_configuration,
            async_processing=True,
        )
        
        # Update recording status based on webrtc_ctx state
        if webrtc_ctx.state and webrtc_ctx.state.playing:
            status_placeholder.success("🔴 Recording...")
            st.session_state.is_recording = True
            # Show frames being received
            current_frames = len(st.session_state.audio_frames)
            if st.session_state.frames_received and current_frames > 0:
                frames_info.info(f"Received {current_frames} audio frames")
                # Update debug info
                debug_info.info(f"""
                - Frame count: {current_frames}
                - Recording state: Active
                - WebRTC state: {webrtc_ctx.state}
                - Last frame time: {time.time()}
                """)
            else:
                frames_info.warning("""
                No audio frames detected yet. Please:
                1. Speak louder
                2. Check if your microphone is working
                3. Try a different browser (Chrome recommended)
                """)
                
                # Try toggling the advanced settings if no frames after a few seconds
                if time.time() % 10 < 5:  # Show this message half the time
                    st.error("""
                    If you're still not seeing any audio frames:
                    1. Click STOP
                    2. Try the Simple Recorder tab instead
                    3. Or use the text input option
                    """)
        else:
            if st.session_state.is_recording:  # Was recording but now stopped
                status_placeholder.info("✅ Recording stopped. Click ANALYZE to process.")
                st.session_state.is_recording = False
                # Update debug info with final frame count
                debug_info.info(f"""
                - Total frames captured: {len(st.session_state.audio_frames)}
                - Recording state: Stopped
                - WebRTC state: {webrtc_ctx.state if webrtc_ctx.state else 'Inactive'}
                """)
            else:
                status_placeholder.info("Click START to begin recording")
        
        # Add a manual file uploader as an alternative
        st.markdown("---")
        st.markdown("### Or upload audio directly")
        uploaded_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"], key="webrtc_upload")
        if uploaded_file is not None:
            st.audio(uploaded_file)
            if st.button("Analyze uploaded audio", key="process_webrtc_upload"):
                with st.spinner("Processing audio... (this may take a moment)"):
                    try:
                        response = call_huggingface_api_with_retry(
                            API_URL_RECOGNITION, 
                            headers, 
                            data=uploaded_file.getvalue()
                        )
                        if response.status_code == 200:
                            text = response.json().get("text", "Speech recognition failed")
                            st.success(f"🗣 You said: `{text}`")
                            
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
        
        # Process audio when user clicks Analyze
        if st.button("📝 ANALYZE", key="webrtc_analyze"):
            with st.spinner("Processing audio..."):
                # Check if we have audio frames in the session state
                if len(st.session_state.audio_frames) > 0:
                    # Log frame information for debugging
                    logger.info(f"Analyzing {len(st.session_state.audio_frames)} audio frames")
                    logger.debug(f"First few frames shapes: {[f.shape for f in st.session_state.audio_frames[:3]]}")
                    
                    # Save the audio from session state
                    saved = save_audio_frames(st.session_state.audio_frames)
                    
                    if saved:
                        st.success(f"✅ Audio recorded successfully ({len(st.session_state.audio_frames)} frames).")
                        
                        # Speech recognition
                        with open("audio.wav", "rb") as f:
                            data = f.read()
                        
                        # Show audio size for debugging
                        file_size = len(data) / 1024
                        st.info(f"Audio file size: {file_size:.2f} KB")
                        
                        # Extra validation for very small files
                        if file_size < 1.0:
                            st.warning(f"Audio file very small ({file_size:.2f} KB), may contain no speech")
                        
                        response = call_huggingface_api_with_retry(API_URL_RECOGNITION, headers, data=data)
                        if response.status_code != 200:
                            st.error(f"API error: {response.status_code}")
                            st.error("The Hugging Face service is currently unavailable. Please try again later.")
                            st.info("Response details:")
                            st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                            text = "Speech recognition failed"
                        else:
                            result_json = response.json()
                            logger.debug(f"API response: {result_json}")
                            text = result_json.get("text", "Speech recognition failed")
                        
                        st.write(f"🗣 You said: `{text}`")
                        
                        # Diagnosis
                        response = call_huggingface_api_with_retry(
                            DIAGNOSTIC_MODEL_API, 
                            headers, 
                            json_data={"inputs": [text]}
                        )
                        
                        try:
                            if response.status_code == 200:
                                result = response.json()[0]['generated_text']
                            else:
                                st.error(f"Diagnosis API error: {response.status_code}")
                                st.error("The Hugging Face service is currently unavailable. Please try again later.")
                                st.info("Response details:")
                                st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                                result = "Diagnosis failed due to service unavailability."
                        except Exception as e:
                            logger.error(f"Error in diagnosis: {str(e)}")
                            result = "Diagnosis failed."
                        
                        st.success(f"🧠 Diagnosis: {result}")
                        
                        # Update history
                        st.session_state.history.append({"message": text, "is_user": True})
                        st.session_state.history.append({"message": result, "is_user": False})
                    else:
                        st.warning("⚠️ Failed to save audio.")
                        # Let's examine what went wrong by checking the frames
                        if len(st.session_state.audio_frames) > 0:
                            num_frames = len(st.session_state.audio_frames)
                            audio_stats = {
                                "count": num_frames,
                                "shapes": [str(f.shape) for f in st.session_state.audio_frames[:3]],
                                "means": [float(np.mean(f)) for f in st.session_state.audio_frames[:3]],
                                "max_values": [float(np.max(f)) for f in st.session_state.audio_frames[:3]]
                            }
                            logger.debug(f"Audio stats: {audio_stats}")
                            st.error(f"""
                            Audio saving failed despite having {num_frames} frames.
                            First few frames shapes: {audio_stats['shapes']}
                            Audio levels: {audio_stats['means']}
                            """)
                        else:
                            st.error("No audio frames were present despite counter indicating otherwise.")
                else:
                    st.warning("⚠️ No audio captured yet. Try speaking into the mic.")
                    # Show debug info to help troubleshoot
                    st.error(f"""
                    Troubleshooting info:
                    - WebRTC state: {webrtc_ctx.state if webrtc_ctx.state else 'Not initialized'}
                    - Frame counter: {st.session_state.frame_counter}
                    - Frames received flag: {st.session_state.frames_received}
                    
                    Please try:
                    1. Refreshing the page
                    2. Using Chrome browser 
                    3. Checking your microphone in system settings
                    4. Trying the Simple Recorder tab
                    """)
        
        # Add a manual reset button
        if st.button("Reset Audio Capture"):
            st.session_state.audio_frames = []
            st.session_state.frame_counter = 0
            st.session_state.frames_received = False
            st.session_state.is_recording = False
            st.success("Audio capture has been reset. Try recording again.")
    
    with tab2:
        st.subheader("Simple Audio Recorder (Alternative)")
        st.write("If the WebRTC recorder isn't working, try this simpler recorder:")
        simple_audio = use_simple_audio_recorder()
    
    with tab3:
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
    st.error(f"⚠️ WebRTC error: {str(e)}")
    logger.error(f"WebRTC error: {str(e)}")
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
