import base64
import numpy as np
import requests
import soundfile as sf
import spacy
import streamlit as st
import subprocess
import sys
import tempfile
import uuid
from audiorecorder import audiorecorder
# from faster_whisper import WhisperModel
from IPython.display import display
from io import BytesIO
from kokoro import KPipeline
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = "https://ttygmhmwmgpblfstbang.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0eWdtaG13bWdwYmxmc3RiYW5nIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjY3Mjk4OCwiZXhwIjoyMDYyMjQ4OTg4fQ.U2zgReY_71RZY_BTRSgGgbd8IrVXCCLHef_vFGaQSYA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Webhook URL (replace with your n8n webhook URL)
# WEBHOOK_URL = "https://gcg-big.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"
# WEBHOOK_URL = "http://localhost:8000/chat"
WEBHOOK_URL = "https://big-2.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"


st.markdown("""
<style>one 
.chat-container { 
    display: flex;
    margin-bottom: 10px;
    padding: 8px;
    font-family: 'Inter', sans-serif !important;
}
.user-message {
    margin-left: auto;
    background-color: #455a64;
    color: #ffffff;
    padding: 16px;
    border-radius: 18px;
    max-width: 75%;
}
.assistant-message {
    margin-right: auto;
    background-color: #343D46;
    color: #ffffff;
    padding: 20px;
    border-radius: 18px;
    max-width: 75%;
}
</style>
""", unsafe_allow_html=True)

def init_kokoro():
    pipeline = KPipeline(lang_code='a')
    return pipeline

# def init_whisper():
#     model = WhisperModel("large-v3", device="cpu", compute_type="int8")
#     return model

def login(email: str, password: str):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return res
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def signup(email: str, password: str):
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return res
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return None

def generate_session_id():
    return str(uuid.uuid4())

def init_session_state():
    if "auth" not in st.session_state:
        st.session_state.auth = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat():
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-container'><div class='user-message'>{message['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-container'><div class='assistant-message'>{message['content']}</div></div>", unsafe_allow_html=True)

def handle_logout():
    st.session_state.auth = None
    st.session_state.session_id = None
    st.session_state.messages = []
    st.rerun()

def auth_ui():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            auth = login(email, password)
            if auth:
                st.session_state.auth = auth
                st.session_state.session_id = generate_session_id()
                st.rerun()

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            auth = signup(email, password)
            if auth:
                st.success("Sign up successful! Please log in.")

def main():
    init_session_state()
    
    kokoro_pipeline = init_kokoro()
    # model = init_whisper()

    if st.session_state.auth is None:
        auth_ui()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.auth.user.email}")
        st.sidebar.info(f"Session ID: {st.session_state.session_id}")

        if st.sidebar.button("Logout"):
            handle_logout()

        display_chat()

        enable_audio = st.checkbox("🔊 Enable AI voice response", value=False)

        if prompt := st.chat_input("What is your message?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f"<div class='chat-container'><div class='user-message'>{prompt}</div></div>", unsafe_allow_html=True)

            payload = {
                "chatInput": prompt,
                "sessionId": st.session_state.session_id
            }

            access_token = st.session_state.auth.session.access_token
            headers = {"Authorization": f"Bearer {access_token}"}
            
            try:
                with st.spinner("AI is thinking..."):
                    response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=60)

                if response.status_code == 200:
                    ai_message = response.json().get("output", "Sorry, I couldn't generate a response.")
                    st.session_state.messages.append({"role": "assistant", "content": ai_message})
                    st.markdown(f"<div class='chat-container'><div class='assistant-message'>{ai_message}</div></div>", unsafe_allow_html=True)
            

                # with st.spinner("AI is thinking..."):
                #     response = requests.post(WEBHOOK_URL, json=payload, headers=headers)

                # if response.status_code == 200:
                #     ai_message = response.json().get("output", "Sorry, I couldn't generate a response.")
                #     st.session_state.messages.append({"role": "assistant", "content": ai_message})
                #     st.markdown(f"<div class='chat-container'><div class='assistant-message'>{ai_message}</div></div>", unsafe_allow_html=True)

                    if enable_audio:                    
                        with st.spinner("Generating audio..."):
                            generator = kokoro_pipeline(
                                ai_message,
                                voice='af_heart',
                                speed=1,
                                split_pattern=r'\n+'
                            )
                            
                            # Collect all audio arrays
                            audio_segments = []
                            for _, _, audio in generator:
                                audio_segments.append(audio)

                            # Concatenate all audio arrays
                            full_audio = np.concatenate(audio_segments)

                            # Convert to WAV in memory
                            buffer = BytesIO()
                            sf.write(buffer, full_audio, samplerate=24000, format='WAV')
                            buffer.seek(0)

                            # Convert to base64
                            b64 = base64.b64encode(buffer.read()).decode()  

                            # Autoplay a single audio tag
                            audio_html = f"""
                            <audio autoplay>
                                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                                Your browser does not support the audio element.
                            </audio>
                            """
                            st.markdown(audio_html, unsafe_allow_html=True)
                            
                # else:
                #     st.error(f"Error: {response.status_code} - {response.text}")
                else:
                    # Friendly fallback if backend fails
                    st.session_state.messages.append({"role": "assistant", "content": "I'm having trouble processing that. The answer might be too long. Try rephrasing or asking a shorter question."})
                    st.markdown(f"<div class='chat-container'><div class='assistant-message'>I'm having trouble processing that. The answer might be too long. Try rephrasing or asking a shorter question.</div></div>", unsafe_allow_html=True)

            except requests.exceptions.RequestException as e:
                # Handle network/timeout issues
                st.session_state.messages.append({"role": "assistant", "content": "Network error. Please try again later."})
                st.markdown(f"<div class='chat-container'><div class='assistant-message'>Network error. Please try again later.</div></div>", unsafe_allow_html=True)
                
        

if __name__ == "__main__":
    main()