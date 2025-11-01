import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import sys
from datetime import datetime

# Setup logging to capture all output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        pass

sys.stdout = Logger("training_log.txt")
print(f"Training started at: {datetime.now()}")
print("=" * 50)

# Setup NLTK
nltk.download(["punkt", "punkt_tab", "stopwords", "wordnet"], quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Clean and preprocess text"""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# Load Data
print("Loading data...")
df = pd.read_csv("data/Reviews.csv")
df = df[["Text", "Score"]].dropna()

# Create sentiment labels
df["Sentiment"] = pd.cut(df["Score"], bins=[0, 2.5, 3.5, 5], labels=["negative", "neutral", "positive"])
df["Sentiment"] = df["Sentiment"].astype("category").cat.codes  # 0=neg, 1=neu, 2=pos

print(f"Original dataset shape: {df.shape}")
print("Original sentiment distribution:")
print(df["Sentiment"].value_counts())

# Sample 50% from each sentiment class
df = df.groupby('Sentiment').apply(lambda x: x.sample(frac=0.5, random_state=42)).reset_index(drop=True)

print(f"Balanced dataset shape: {df.shape}")
print("Balanced sentiment distribution:")
print(df["Sentiment"].value_counts())

# Preprocess text
print("Preprocessing text...")
df["Processed"] = df["Text"].apply(preprocess)
print("Sample processed text:", df["Processed"].iloc[0])

# Tokenization & Padding
print("Tokenizing and padding...")
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["Processed"])
sequences = tokenizer.texts_to_sequences(df["Processed"])
max_len = 100
padded = pad_sequences(sequences, maxlen=max_len, padding="post")

X = padded
y = to_categorical(df["Sentiment"], num_classes=3)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Build Model
print("Building model...")
vocab_size = 5000
embedding_dim = 50

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=True, dropout=0.3),
    LSTM(32, dropout=0.3),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train Model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# Save model and tokenizer
model.save("lstm_sentiment.h5")
joblib.dump(tokenizer, "tokenizer.pkl")
print("Model saved as 'lstm_sentiment.h5'")
print("Tokenizer saved as 'tokenizer.pkl'")

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print(classification_report(y_test_classes, y_pred_classes, target_names=["negative", "neutral", "positive"]))

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
           xticklabels=["neg", "neu", "pos"], 
           yticklabels=["neg", "neu", "pos"])
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Test prediction function
def predict_sentiment(text, tokenizer, model, max_len=100):
    proc = preprocess(text)
    seq = tokenizer.texts_to_sequences([proc])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad, verbose=0)[0]
    label = np.argmax(pred)
    confidence = np.max(pred)
    sentiments = ["negative", "neutral", "positive"]
    return sentiments[label], confidence

# Test examples
test_texts = [
    "This product is amazing!",
    "Terrible quality, waste of money",
    "It's okay, nothing special"
]

print("\nTest predictions:")
for text in test_texts:
    result = predict_sentiment(text, tokenizer, model)
    print(f"'{text}' -> {result}")

print("\nTraining complete!")