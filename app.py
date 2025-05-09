# import streamlit as st
# import requests
# import uuid

# # Constants
# WEBHOOK_URL = "https://gcg-big.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"
# BEARER_TOKEN = "{123}"

# def generate_session_id():
#     return str(uuid.uuid4())

# def send_message_to_llm(session_id, message):
#     headers = {
#         "Authorization": f"Bearer {BEARER_TOKEN}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "sessionId": session_id,
#         "chatInput": message
#     }
#     response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
#     if response.status_code == 200:
#         return response.json()["output"]
#     else:
#         return f"Error: {response.status_code} - {response.text}"

# def main():
#     st.title("Chat with LLM")

#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = generate_session_id()

#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # User input
#     user_input = st.chat_input("Type your message here...")

#     if user_input:
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.write(user_input)

#         # Get LLM response
#         llm_response = send_message_to_llm(st.session_state.session_id, user_input)

#         # Add LLM response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": llm_response})
#         with st.chat_message("assistant"):
#             st.write(llm_response)

# if __name__ == "__main__":
#     main()








import streamlit as st
import requests
import uuid
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = "YOUR_SUPABASE_PROJECT_URL_HERE"
SUPABASE_KEY = "YOUR_SUPABASE_ANONYMOUS_API_KEY_HERE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Webhook URL (replace with your n8n webhook URL)
WEBHOOK_URL = "https://gcg-big.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"

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
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
    st.title("AI Chat Interface")
    init_session_state()

    if st.session_state.auth is None:
        auth_ui()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.auth.user.email}")
        st.sidebar.info(f"Session ID: {st.session_state.session_id}")

        if st.sidebar.button("Logout"):
            handle_logout()

        display_chat()

        if prompt := st.chat_input("What is your message?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare the payload
            payload = {
                "chatInput": prompt,
                "sessionId": st.session_state.session_id
            }
            
            # Get the access token from the session
            access_token = st.session_state.auth.session.access_token
            
            # Send request to webhook
            headers = {
                "Authorization": f"Bearer {access_token}"
            }
            with st.spinner("AI is thinking..."):
                response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                ai_message = response.json().get("output", "Sorry, I couldn't generate a response.")
                st.session_state.messages.append({"role": "assistant", "content": ai_message})
                with st.chat_message("assistant"):
                    st.markdown(ai_message)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()