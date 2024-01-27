import requests
from pydub import AudioSegment
import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import json
import time

# Updated API details
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"

# New model API details
NEW_MODEL_API_URL = "https://api-inference.huggingface.co/models/shanover/medbot_godel_v3"
NEW_MODEL_INFO = {"name": "New Model", "api_url": NEW_MODEL_API_URL}
DIAGNOSTIC_MODELS = [NEW_MODEL_INFO]

headers = {"Authorization": "Bearer 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"}


def recognize_speech(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()

    response = requests.post(API_URL_RECOGNITION, headers=headers, data=data)

    if response.status_code == 503:  # HTTP 503 Service Unavailable
        estimated_time = response.json().get('estimated_time', 20.0)
        st.warning(
            f"Model is currently loading. Please wait for approximately {estimated_time:.2f} seconds and try again.")
        time.sleep(20)
        return recognize_speech(audio_file)  # Retry after waiting

    if response.status_code != 200:
        st.error(f"Speech recognition API error: {response.content}")
        return "Speech recognition failed"

    output = response.json()
    final_output = output.get('text', 'Speech recognition failed')
    return final_output


def diagnostic_medic(voice_text):
    model_results = []

    for model_info in DIAGNOSTIC_MODELS:
        payload = {"inputs": [voice_text]}
        response = requests.post(model_info["api_url"], headers=headers, json=payload)

        try:
            generated_text = response.json()[0]['generated_text']
            model_results.append({"name": model_info["name"], "results": [{'label': generated_text, 'score': 1.0}]})
        except (KeyError, IndexError):
            st.warning(f'Diagnostic information not available for {model_info["name"]}')

    if not model_results:
        return 'No diagnostic information available'

    # Compare results based on confidentiality score and choose the model with the highest score
    best_model_result = max(model_results, key=lambda x: max([result['score'] for result in x['results']], default=0.0))

    return format_diagnostic_results(best_model_result["results"], best_model_result["name"])


def format_diagnostic_results(results, model_name):
    # Sort the results based on the score in descending order
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Extract the names and scores of the top results
    top_results = sorted_results[:2]
    formatted_results = [(result['label'], result['score']) for result in top_results]

    if not formatted_results:
        return 'No diagnostic information available'

    # Create a string with disease names and confidence scores
    formatted_results_str = ', '.join([f'{label} ({score:.2%})' for label, score in formatted_results])

    return f'Top Diseases or Symptoms from {model_name}:\n{formatted_results_str}'


def generate_answer(audio_recording):
    st.spinner("Consultation in progress...")

    # To save audio to a file:
    audio_recording.export("audio.wav", format="wav")

    # Voice recognition model
    st.write("Audio file saved. Starting speech recognition...")
    text = recognize_speech("audio.wav")

    if "recognition failed" in text.lower():
        st.error("Voice recognition failed. Please try again.")
        return

    st.write(f"Speech recognition result: {text}")

    # Disease Prediction Model
    st.write("Calling diagnostic models...")
    diagnostic = diagnostic_medic(text)
    st.write(f"Diagnostic result:\n{diagnostic}")

    # Add the statement for more detailed symptoms
    st.write("Please provide more detailed symptoms for precise recognition.")

    # Save conversation
    st.session_state.history.append({"message": text, "is_user": True})
    st.session_state.history.append({"message": diagnostic, "is_user": False})

    st.success("Medical consultation done")


if __name__ == "__main__":
    # Remove the hamburger in the upper right-hand corner and the Made with Streamlit footer
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("./logo_.png", width=200)

    with col3:
        st.write(' ')

    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("Medical Diagnostic Assistant")

    # Show Input
    audio = audiorecorder("Start recording", "Recording in progress...")

    if audio:
        generate_answer(audio)

        for i, chat in enumerate(st.session_state.history):  # Show historical consultation
            st_message(**chat, key=str(i))
