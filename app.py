from flask import Flask, render_template, request, redirect, url_for, session
import uuid
import requests
import base64
import numpy as np
import soundfile as sf
from io import BytesIO
from kokoro import KPipeline
from supabase import create_client, Client

# --- Supabase config ---
SUPABASE_URL = "https://ttygmhmwmgpblfstbang.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WEBHOOK_URL = "https://big-2.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"

app = Flask(__name__)
# app.secret_key = "supersecretkey" 

# --- Kokoro Init ---
def init_kokoro():
    return KPipeline(lang_code='a')

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    messages = session.get("messages", [])
    audio_html = None

    if request.method == "POST":
        prompt = request.form.get("prompt")
        enable_audio = request.form.get("enable_audio") == "on"

        messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {session['access_token']}"}
        payload = {
            "chatInput": prompt,
            "sessionId": session["session_id"]
        }

        try:
            res = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=60)
            if res.status_code == 200:
                output = res.json().get("output", "Sorry, no output.")
                messages.append({"role": "assistant", "content": output})

                if enable_audio:
                    kokoro = init_kokoro()
                    audio_segments = [audio for _, _, audio in kokoro(output, voice='af_heart', speed=1, split_pattern=r'\n+')]
                    full_audio = np.concatenate(audio_segments)
                    buf = BytesIO()
                    sf.write(buf, full_audio, samplerate=24000, format='WAV')
                    b64_audio = base64.b64encode(buf.getvalue()).decode()
                    audio_html = f"""
                    <audio autoplay><source src='data:audio/wav;base64,{b64_audio}' type='audio/wav'></audio>
                    """

            else:
                messages.append({"role": "assistant", "content": "Error: backend failed."})
        except Exception:
            messages.append({"role": "assistant", "content": "Network error."})

        session["messages"] = messages

    return render_template("chat.html", messages=messages, audio_html=audio_html)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            session["user"] = res.user.email
            session["access_token"] = res.session.access_token
            session["session_id"] = str(uuid.uuid4())
            session["messages"] = []
            return redirect(url_for("index"))
        except Exception as e:
            return render_template("login.html", error="Login failed.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == '__main__':
    app.run(debug=True)
