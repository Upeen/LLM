import streamlit as st
from llm import load_llm, MODEL_LIST

st.set_page_config(
    page_title="Free Streamlit LLM",
    layout="wide"
)

st.title("⚡ Free & Lightweight LLM on Streamlit")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    list(MODEL_LIST.keys())
)

temperature = st.sidebar.slider(
    "Creativity",
    0.1, 1.0, 0.7
)

# Cache model
@st.cache_resource
def get_llm(model_name):
    return load_llm(model_name)

llm = get_llm(MODEL_LIST[model_choice])

# Session chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            prompt = f"User: {user_input}\nAssistant:"
            response = llm(prompt)
            st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# Extra features
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
