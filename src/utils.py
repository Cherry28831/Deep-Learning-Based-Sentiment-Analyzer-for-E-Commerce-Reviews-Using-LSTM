import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import joblib  # For saving tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Setup NLTK (run once)
nltk.download(["punkt", "punkt_tab", "stopwords", "wordnet"], quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    """
    Clean and preprocess a single review text.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def prepare_data(
    df,
    text_col="Text",
    label_col="Score",
    max_samples=20000,
    max_len=200,
    vocab_size=10000,
):
    """
    Load, subsample, preprocess, and prepare data for training.
    Returns: X (padded sequences), y (one-hot), tokenizer, splits.
    """
    df = df[[text_col, label_col]].dropna().sample(max_samples, random_state=42)
    df["Sentiment"] = pd.cut(
        df[label_col], bins=[0, 2.5, 3.5, 5], labels=["negative", "neutral", "positive"]
    )
    df["Sentiment"] = (
        df["Sentiment"].astype("category").cat.codes
    )  # 0=neg, 1=neu, 2=pos

    df["Processed"] = df[text_col].apply(preprocess)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["Processed"])
    sequences = tokenizer.texts_to_sequences(df["Processed"])
    padded = pad_sequences(sequences, maxlen=max_len, padding="post")

    X = padded
    y = pd.get_dummies(df["Sentiment"]).values  # One-hot

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Save tokenizer
    joblib.dump(tokenizer, "tokenizer.pkl")

    return (X_train, X_val, X_test), (y_train, y_val, y_test), tokenizer


def predict_sentiment(text, tokenizer, model, max_len=200):
    """
    Predict sentiment for a single text.
    Returns: (label, confidence)
    """
    proc = preprocess(text)
    seq = tokenizer.texts_to_sequences([proc])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad, verbose=0)[0]
    label_idx = np.argmax(pred)
    confidence = np.max(pred)
    sentiments = ["negative", "neutral", "positive"]
    return sentiments[label_idx], confidence


def evaluate_model(model, X_test, y_test):
    """
    Evaluate and print metrics + confusion matrix.
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print(
        classification_report(
            y_test_classes,
            y_pred_classes,
            target_names=["negative", "neutral", "positive"],
        )
    )

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(
        "F1-Score (macro):", f1_score(y_test_classes, y_pred_classes, average="macro")
    )

    return cm


def generate_wordcloud(df, sentiment_class=2, title="Positive Words"):
    """
    Generate word cloud for a sentiment class.
    """
    pos_df = df[df["Sentiment"] == sentiment_class]
    text = " ".join(pos_df["Processed"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# Example usage (for testing)
if __name__ == "__main__":
    # Dummy test
    sample_text = "This is amazing!"
    print(preprocess(sample_text))  # Output: cleaned tokens
