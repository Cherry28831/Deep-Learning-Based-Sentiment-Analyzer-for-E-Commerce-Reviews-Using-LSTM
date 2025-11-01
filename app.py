import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Setup (run once)
nltk.download(["punkt", "stopwords", "wordnet"], quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def predict_sentiment(text, tokenizer, model, max_len=100):
    proc = preprocess(text)
    seq = tokenizer.texts_to_sequences([proc])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad, verbose=0)[0]
    label = np.argmax(pred)
    confidence = np.max(pred)
    sentiments = ["negative", "neutral", "positive"]
    return sentiments[label], confidence


# Load model and tokenizer
@st.cache_resource
def load_all():
    import joblib
    import os
    
    # Debug: Show current directory and files
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir("."))
    
    if os.path.exists("models"):
        st.write("Files in models directory:", os.listdir("models"))
    else:
        st.error("Models directory not found!")
        st.write("Available directories:", [d for d in os.listdir(".") if os.path.isdir(d)])
        st.stop()
    
    try:
        model = load_model("models/lstm_sentiment.h5")
        tokenizer = joblib.load("models/tokenizer.pkl")
        st.success("Models loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()


model, tokenizer = load_all()

st.title("LSTM Sentiment Analyzer")
text = st.text_area("Enter Review:", "This is the best product ever!")

# Store user inputs for word cloud
if 'user_texts' not in st.session_state:
    st.session_state.user_texts = {'negative': [], 'neutral': [], 'positive': []}

if st.button("Analyze"):
    label, conf = predict_sentiment(text, tokenizer, model)
    st.success(f"Sentiment: **{label}** (Confidence: {conf:.2f})")
    
    # Add to session state for word cloud
    processed_text = preprocess(text)
    st.session_state.user_texts[label].append(processed_text)

uploaded = st.file_uploader("Upload CSV for Batch")
if uploaded:
    df_up = pd.read_csv(uploaded)
    df_up["Processed"] = df_up["Text"].apply(preprocess)  # Assume 'Text' col
    results = []
    for i, proc in enumerate(df_up["Processed"]):
        seq = tokenizer.texts_to_sequences([proc])
        pad = pad_sequences(seq, maxlen=100, padding='post')
        pred = model.predict(pad, verbose=0)[0]
        label_idx = np.argmax(pred)
        label = ["negative", "neutral", "positive"][label_idx]
        results.append({"Text": df_up["Text"].iloc[i], "Sentiment": label})
        
        # Add to word cloud data
        st.session_state.user_texts[label].append(proc)
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    st.success(f"Processed {len(results)} reviews and added to word clouds!")

# Generate word clouds from user inputs
st.subheader("Word Clouds from Your Inputs")

for sentiment in ['negative', 'neutral', 'positive']:
    if st.session_state.user_texts[sentiment]:
        combined_text = " ".join(st.session_state.user_texts[sentiment])
        if len(combined_text.strip()) > 0:
            try:
                wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"{sentiment.title()} Words from Your Reviews")
                st.pyplot(fig)
                plt.close(fig)
            except:
                st.write(f"Not enough {sentiment} words yet to generate word cloud")
    else:
        st.write(f"No {sentiment} reviews entered yet")

if st.button("Clear Word Clouds"):
    st.session_state.user_texts = {'negative': [], 'neutral': [], 'positive': []}
    st.rerun()
