import streamlit as st
import tiktoken
import re
from collections import Counter

# ---------------- TOKEN COUNTER ---------------- #
def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# ---------------- SIMPLE SENTENCE SPLITTER ---------------- #
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# ---------------- CLEAN TEXT ---------------- #
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- SIMPLE FREQUENCY SUMMARIZER ---------------- #
def summarize_text(text, compression_ratio=0.6):
    sentences = split_sentences(text)
    words = re.findall(r'\w+', text.lower())
    freq = Counter(words)

    sentence_scores = {}
    for sentence in sentences:
        score = 0
        sentence_words = re.findall(r'\w+', sentence.lower())
        for word in sentence_words:
            score += freq[word]
        sentence_scores[sentence] = score

    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    select_len = max(1, int(len(sentences) * compression_ratio))
    selected = ranked[:select_len]
    return " ".join(selected)

# ---------------- TOKEN FITTING ---------------- #
def fit_to_token_limit(text, target_tokens, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= target_tokens:
        return text
    trimmed = tokens[:target_tokens]
    return encoding.decode(trimmed)

# ---------------- MAIN OPTIMIZER ---------------- #
def optimize_prompt(text, target_tokens):
    text = clean_text(text)
    summarized = summarize_text(text, compression_ratio=0.7)
    optimized = fit_to_token_limit(summarized, target_tokens)
    return optimized

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(
    page_title="PromptOptimizerX 🌈🤖", 
    layout="wide", 
    page_icon="🧠"
)

# ---------------- CUSTOM CSS FOR COLORFUL AI EFFECT ---------------- #
st.markdown(
    """
    <style>
    /* Light background gradient */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb, #ebedee, #ffffff);
        color: #111;  /* Dark text for readability */
    }
    /* Title glow */
    h2 {
        text-shadow: 0 0 8px #00f0ff, 0 0 16px #ff00ff, 0 0 24px #00ff99;
        color: #222;
    }
    /* TextArea styling */
    textarea {
        background: #f0f4f8;
        color: #111;
        border: 2px solid #00f0ff;
        border-radius: 8px;
        padding: 10px;
    }
    /* Button styling */
    div.stButton>button {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        color: #000;
        font-weight: bold;
        border-radius: 12px;
        height: 50px;
        width: 200px;
    }
    div.stButton>button:hover {
        box-shadow: 0 0 20px #ff00ff, 0 0 30px #00ffff;
        transform: scale(1.05);
    }
    /* Metrics cards */
    .stMetric {
        background: rgba(0,0,0,0.05);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("## 🤖 PromptOptimizerX: AI-Powered Token Optimizer")

user_input = st.text_area("📝 Enter your prompt:", height=250)
target_tokens = st.number_input("🎯 Target Token Limit", min_value=10, value=100)

if st.button("⚡ Optimize Prompt"):
    if user_input:
        original_tokens = count_tokens(user_input)
        optimized_text = optimize_prompt(user_input, target_tokens)
        optimized_tokens = count_tokens(optimized_text)
        reduction = original_tokens - optimized_tokens
        percent = (reduction / original_tokens) * 100 if original_tokens > 0 else 0

        st.subheader("✨ Optimized Prompt")
        st.text_area("Result:", optimized_text, height=200)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Tokens 🔹", original_tokens)
        col2.metric("Optimized Tokens 🔹", optimized_tokens)
        col3.metric("Reduction % 🔹", f"{percent:.2f}%")
    else:
        st.warning("⚠️ Please enter a prompt.")