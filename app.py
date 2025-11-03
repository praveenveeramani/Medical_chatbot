# app.py
import streamlit as st
from medical_chatbot import get_response

st.title("🧠 Medical Chatbot (BioMistral-7B)")
st.write("Ask medical questions. The model runs locally using LlamaCpp.")

query = st.text_input("Enter your query:")
if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = get_response(query)
        st.success("Response:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
