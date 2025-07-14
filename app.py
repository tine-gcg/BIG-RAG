from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
import uuid
import requests
import base64
import numpy as np
import os
import soundfile as sf
from io import BytesIO
from kokoro import KPipeline
from supabase import create_client, Client

# --- Load environment variables ---
load_dotenv()

# --- Supabase config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# --- Kokoro Init ---
def init_kokoro():
    return KPipeline(lang_code='a')

# --- Routes ---
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if "user" not in session:
#         return redirect(url_for("login"))

#     messages = session.get("messages", [])
#     audio_html = None

#     if request.method == "POST":
#         prompt = request.form.get("prompt")
#         enable_audio = request.form.get("enable_audio") == "on"

#         messages.append({"role": "user", "content": prompt})

#         headers = {"Authorization": f"Bearer {session['access_token']}"}
#         payload = {
#             "chatInput": prompt,
#             "sessionId": session["session_id"]
#         }

#         try:
#             res = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=60)
#             print("n8n response status:", res.status_code)
#             print("n8n response headers:", res.headers)
#             print("n8n response text:", res.text)

#             if res.status_code == 200:
#                 try:
#                     data = res.json()
#                     print("n8n response JSON:", data)
#                     output = data.get("output", "Missing 'output' in response.")
#                 except Exception as e:
#                     output = f"Failed to parse response: {str(e)}"

#                 messages.append({"role": "assistant", "content": output})

#                 if enable_audio:
#                     kokoro = init_kokoro()
#                     audio_segments = [audio for _, _, audio in kokoro(output, voice='af_heart', speed=1, split_pattern=r'\n+')]
#                     full_audio = np.concatenate(audio_segments)
#                     buf = BytesIO()
#                     sf.write(buf, full_audio, samplerate=24000, format='WAV')
#                     b64_audio = base64.b64encode(buf.getvalue()).decode()
#                     audio_html = f"""
#                         <audio autoplay><source src='data:audio/wav;base64,{b64_audio}' type='audio/wav'></audio>
#                     """
#             else:
#                 messages.append({"role": "assistant", "content": f"Error from n8n: {res.status_code} - {res.text}"})
#         except Exception as e:
#             messages.append({"role": "assistant", "content": f"Network error: {str(e)}"})

#         session["messages"] = messages
#         print("Final messages before render:", session.get("messages"))
        


#     return render_template("chat.html", messages=messages, audio_html=audio_html)

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
            print("n8n response status:", res.status_code)
            print("n8n response headers:", res.headers)
            print("n8n response text:", res.text)

            if res.status_code == 200:
                try:
                    data = res.json()
                    print("n8n response JSON:", data)
                    output = data.get("output", "Missing 'output' in response.")
                except Exception as e:
                    output = f"Failed to parse response: {str(e)}"

                messages.append({"role": "assistant", "content": output})

                if enable_audio:
                    kokoro = init_kokoro()
                    audio_segments = [audio for _, _, audio in kokoro(output, voice='af_heart', speed=1, split_pattern=r'\n+')]
                    full_audio = np.concatenate(audio_segments)
                    buf = BytesIO()
                    sf.write(buf, full_audio, samplerate=24000, format='WAV')
                    b64_audio = base64.b64encode(buf.getvalue()).decode()
                    session["audio_html"] = f"""
                        <audio autoplay><source src='data:audio/wav;base64,{b64_audio}' type='audio/wav'></audio>
                    """
            else:
                messages.append({"role": "assistant", "content": f"Error from n8n: {res.status_code} - {res.text}"})
        except Exception as e:
            messages.append({"role": "assistant", "content": f"Network error: {str(e)}"})

        session["messages"] = messages

        # 👇 Redirect to prevent form resubmission
        return redirect(url_for("index"))

    # For GET request: get audio if any
    audio_html = session.pop("audio_html", None)
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
            print("Supabase login error:", e)
            return render_template("login.html", error=str(e))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == '__main__':
    app.run(debug=True)
