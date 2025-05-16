# import pathlib
# import requests
# import streamlit as st
# import streamlit.components.v1 as components
# import uuid

# WEBHOOK_URL = "https://gcg-big.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"
# BEARER_TOKEN = "1aaae10683651e62de041b9cf32f796f9ff2facc7f07f6045c16fc6c563cb8ac"

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

#     print("Status Code:", response.status_code)
#     print("Raw Response Text:", response.text)

#     try:
#         return response.json()["output"]
#     except Exception as e:
#         return f"Failed to decode JSON: {str(e)}"

# def main():    
#     st.markdown("""
#     <style>   
#     .chat-container{
#         display: flex;
#         margin-bottom: 10px;
#         padding: 8px;
#         font-family: 'Inter', sans-serif !important;
#     }
#     .user-message {
#         margin-left: auto;
#         background-color: #455a64;
#         color: #ffffff;	;
#         padding: 16px;
#         border-radius: 18px;
#         max-width: 75%;
#     }
#     .assistant-message {
#         margin-right: auto;
#         background-color: #343D46;
#         color: #ffffff;	;
#         padding: 20px;
#         border-radius: 18px;
#         max-width: 75%;
#     }
#         </style>
#     """, unsafe_allow_html=True)

#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = generate_session_id()

#     # Display chat messages
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             st.markdown(f"<div class='chat-container'><div class='user-message'>{message['content']}</div></div>", unsafe_allow_html=True)
            
#         else:
#             st.markdown(f"<div class='chat-container'><div class='assistant-message'>{message['content']}</div></div>", unsafe_allow_html=True)

#     # User input
#     user_input = st.chat_input("Type your message here...")

#     if user_input:
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         st.markdown(f"<div class='chat-container'><div class='user-message'>{user_input}</div></div>", unsafe_allow_html=True)

#         # Get LLM response
#         llm_response = send_message_to_llm(st.session_state.session_id, user_input)

#         # Add LLM response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": llm_response})
#         st.markdown(f"<div class='chat-container'><div class='assistant-message'>{llm_response}</div></div>", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


import streamlit as st
import requests
import uuid
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = "https://ttygmhmwmgpblfstbang.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0eWdtaG13bWdwYmxmc3RiYW5nIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjY3Mjk4OCwiZXhwIjoyMDYyMjQ4OTg4fQ.U2zgReY_71RZY_BTRSgGgbd8IrVXCCLHef_vFGaQSYA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Webhook URL (replace with your n8n webhook URL)
WEBHOOK_URL = "https://gcg-big.app.n8n.cloud/webhook-test/bf4dd093-bb02-472c-9454-7ab9af97bd1d"

st.markdown("""
<style>
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

# def display_chat():
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

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
            st.markdown(f"<div class='chat-container'><div class='user-message'>{prompt}</div></div>", unsafe_allow_html=True)

            # Prepare payload
            payload = {
                "chatInput": prompt,
                "sessionId": st.session_state.session_id
            }
        
            access_token = st.session_state.auth.session.access_token

            headers = {
                "Authorization": f"Bearer {access_token}"
            }

            with st.spinner("AI is thinking..."):
                response = requests.post(WEBHOOK_URL, json=payload, headers=headers)

            if response.status_code == 200:
                ai_message = response.json().get("output", "Sorry, I couldn't generate a response.")
                st.session_state.messages.append({"role": "assistant", "content": ai_message})
                st.markdown(f"<div class='chat-container'><div class='assistant-message'>{ai_message}</div></div>", unsafe_allow_html=True)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")


        # if prompt := st.chat_input("What is your message?"):
        #     st.session_state.messages.append({"role": "user", "content": prompt})
        #     with st.chat_message("user"):
        #         st.markdown(prompt)

        #     # Prepare the payload
        #     payload = {
        #         "chatInput": prompt,
        #         "sessionId": st.session_state.session_id
        #     }
            
        #     # Get the access token from the session
        #     access_token = st.session_state.auth.session.access_token
            
        #     # Send request to webhook
        #     headers = {
        #         "Authorization": f"Bearer {access_token}"
        #     }
        #     with st.spinner("AI is thinking..."):
        #         response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
            
        #     if response.status_code == 200:
        #         ai_message = response.json().get("output", "Sorry, I couldn't generate a response.")
        #         st.session_state.messages.append({"role": "assistant", "content": ai_message})
        #         with st.chat_message("assistant"):
        #             st.markdown(ai_message)
        #     else:
        #         st.error(f"Error: {response.status_code} - {response.text}")
                

if __name__ == "__main__":
    main()