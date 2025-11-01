import streamlit as st
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Simple stopwords list
stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

def preprocess(text):
    # Simple preprocessing without NLTK
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def predict_sentiment_demo(text):
    """Demo sentiment prediction using simple keyword matching"""
    processed = preprocess(text)
    
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic', 'wonderful', 'perfect']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor', 'trash', 'useless']
    
    pos_score = sum(1 for word in positive_words if word in processed)
    neg_score = sum(1 for word in negative_words if word in processed)
    
    if pos_score > neg_score:
        return "positive", 0.75 + (pos_score * 0.05)
    elif neg_score > pos_score:
        return "negative", 0.75 + (neg_score * 0.05)
    else:
        return "neutral", 0.60

st.title("LSTM Sentiment Analyzer (Demo)")
st.info("🚧 Demo Mode: Using keyword-based prediction while model files are being deployed")

text = st.text_area("Enter Review:", "This is the best product ever!")

# Store user inputs for word cloud
if 'user_texts' not in st.session_state:
    st.session_state.user_texts = {'negative': [], 'neutral': [], 'positive': []}

if st.button("Analyze"):
    label, conf = predict_sentiment_demo(text)
    st.success(f"Sentiment: **{label}** (Confidence: {conf:.2f})")
    
    # Add to session state for word cloud
    processed_text = preprocess(text)
    st.session_state.user_texts[label].append(processed_text)

uploaded = st.file_uploader("Upload CSV for Batch")
if uploaded:
    df_up = pd.read_csv(uploaded)
    results = []
    for i, row_text in enumerate(df_up["Text"]):
        label, conf = predict_sentiment_demo(row_text)
        results.append({"Text": row_text, "Sentiment": label, "Confidence": f"{conf:.2f}"})
        
        # Add to word cloud data
        processed_text = preprocess(row_text)
        st.session_state.user_texts[label].append(processed_text)
    
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

st.markdown("---")
st.markdown("**Note:** This is a demo version using keyword-based sentiment analysis. The full LSTM model will be deployed soon.")