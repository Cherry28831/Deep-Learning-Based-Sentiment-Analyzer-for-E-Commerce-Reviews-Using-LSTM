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
from io import BytesIO
import base64

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
    model = load_model("lstm_sentiment.h5")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer


model, tokenizer = load_all()

st.title("Simple LSTM Sentiment Analyzer")
text = st.text_area("Enter Review:", "This is the best product ever!")
if st.button("Analyze"):
    label, conf = predict_sentiment(text, tokenizer, model)
    st.success(f"Sentiment: **{label}** (Confidence: {conf:.2f})")

uploaded = st.file_uploader("Upload CSV for Batch")
if uploaded:
    df_up = pd.read_csv(uploaded)
    df_up["Processed"] = df_up["Text"].apply(preprocess)  # Assume 'Text' col
    results = []
    for proc in df_up["Processed"]:
        seq = tokenizer.texts_to_sequences([proc])
        pad = pad_sequences(seq, maxlen=100, padding='post')
        pred = model.predict(pad, verbose=0)[0]
        label = np.argmax(pred)
        results.append({"Sentiment": ["negative", "neutral", "positive"][label]})
    st.dataframe(pd.DataFrame(results))

# Word Cloud Demo (Static for pos)
st.subheader("Positive Word Cloud Example")
pos_wc = WordCloud(width=800, height=400).generate(
    "amazing great love best delicious"
)  # Dummy; load from df
fig, ax = plt.subplots()
ax.imshow(pos_wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
