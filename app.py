import streamlit as st
from transformers import pipeline
import torch

# Load summarization model (cached for performance)
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="t5-small",  # light model, can switch to "facebook/bart-large-cnn"
        device=0 if torch.cuda.is_available() else -1
    )

summarizer = load_model()

# ---- UI ----
st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.title("📝 AI Text Summarizer")
st.write("Paste any text below and get a concise 2–3 line summary.")

# Input box
input_text = st.text_area("Enter your text:", height=200, placeholder="Paste paragraph or article here...")

# Button
if st.button("Summarize"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            summary = summarizer(input_text, max_length=60, min_length=20, do_sample=False)
        st.subheader("Summary:")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("⚠️ Please enter some text first.")
