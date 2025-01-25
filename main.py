import streamlit as st
from  utilities import embed_files, getResponse

max_history = 6

file_names = []

if "prev_count" not in st.session_state:
    st.session_state.prev_count = 0

with st.sidebar:
    st.title("QA Chatbot")
    st.image("chatbot_img.jpeg", width=100)
    uploaded_files = st.file_uploader(label="Please upload your files by clicking 'Browse files' ", accept_multiple_files=True)
    if uploaded_files:
        current_count = len(uploaded_files)
        
        if st.session_state.prev_count < current_count:
            embed_files(uploaded_files)
        st.session_state.prev_count = current_count

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:= st.chat_input("Please ask you query here:"):
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
                       
    response = getResponse(prompt, st.session_state.messages)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

    if len(st.session_state.messages) > max_history:
        st.session_state.messages = st.session_state.messages[-max_history:]
