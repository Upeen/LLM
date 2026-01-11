import streamlit as st
from llm import load_model, MODEL_LIST

st.set_page_config(
    page_title="Free Streamlit LLM",
    layout="wide"
)

st.title("⚡ Free Lightweight LLM (Streamlit)")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_LIST.keys())
)

temperature = st.sidebar.slider(
    "Creativity",
    0.1, 1.0, 0.7
)

@st.cache_resource
def get_model(model_name):
    return load_model(model_name)

generator = get_model(MODEL_LIST[model_choice])

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            output = generator(
                user_input,
                temperature=temperature
            )
            reply = output[0]["generated_text"]
            st.write(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

# Extra
st.sidebar.divider()
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
