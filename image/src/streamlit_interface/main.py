import streamlit as st
import sys
import os
from dotenv import load_dotenv
from audio.Speech_To_Text import AudioRecorder
from rag.Rag_model import get_final_response

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and description
st.title("💬 Langraph AI Agent: Adaptative RAG 🤖")
st.info("Explore advanced retrieval-augmented generation with our custom RAG model", icon="📚")

# Sidebar for additional info
st.sidebar.title("About the RAG Chatbot")
st.sidebar.markdown("""
### How it Works
- Retrieves relevant documents.
- Make Web search.
- Generates contextually-aware responses.
""")

# Initialize session state variables
if "recorder" not in st.session_state:
    st.session_state.recorder = AudioRecorder(sample_rate=16000, channels=1)
    st.session_state.is_recording = False
    st.session_state.audio_file = None
    st.session_state.transcription = None  # Store transcription here
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about AI, ML, or my knowledge base!"}
    ]

recorder = st.session_state.recorder

# Sidebar for recording
st.sidebar.markdown("### Enregistrement vocal")
if not st.session_state.is_recording and st.sidebar.button("🎤 Démarrer l'enregistrement"):
    st.session_state.is_recording = True
    recorder.start_recording()
    st.sidebar.success("Enregistrement démarré!")

if st.session_state.is_recording and st.sidebar.button("⏹️ Arrêter l'enregistrement"):
    st.session_state.is_recording = False
    audio_file = recorder.stop_recording()
    st.session_state.audio_file = audio_file
    st.sidebar.success(f"Enregistrement arrêté et sauvegardé: {audio_file}")

# Process recorded audio
if st.session_state.audio_file:
    st.sidebar.write("Audio enregistré: ", st.session_state.audio_file)
    if st.sidebar.button("Transcrire l'audio"):
        transcription = recorder.transcribe_audio(st.session_state.audio_file, model_size="medium.en", device="cuda")
        if transcription:
            st.session_state.transcription = transcription
            st.sidebar.success("Transcription réussie et insérée comme requête.")
        else:
            st.sidebar.error("Erreur lors de la transcription de l'audio.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input field for new questions
user_input = st.chat_input("Posez une question :")

# If transcription exists, pre-fill the input field and clear transcription
if st.session_state.transcription and not user_input:
    user_input = st.session_state.transcription
    st.session_state.transcription = None  # Clear transcription after use

# Process user input and generate response
if user_input:
    # Add user message to chat history immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message instantly in the chat
    with st.chat_message("user"):
        st.write(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate response using the RAG model
            response = get_final_response(user_input)
            st.write(response.response_text)
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
