# Deep Learning Based Sentiment Analyzer for E-Commerce Reviews Using LSTM

A comprehensive sentiment analysis system that classifies product reviews into positive, negative, or neutral sentiments using LSTM neural networks.

## 🚀 Live Demo

**Try the app:** [https://cherry28831-sentiment-analysis-for-product-reviews-u-app-vppgkg.streamlit.app/](https://cherry28831-sentiment-analysis-for-product-reviews-u-app-vppgkg.streamlit.app/)

## 📋 Features

- **Real-time Sentiment Analysis**: Analyze individual reviews instantly
- **Batch Processing**: Upload CSV files for bulk sentiment analysis
- **Interactive Word Clouds**: Visualize sentiment-specific words from your inputs
- **High Accuracy**: 84% accuracy on test data with balanced dataset
- **Web Interface**: User-friendly Streamlit application

## 🏗️ Architecture

- **Model**: LSTM with embedding layer (291,955 parameters)
- **Dataset**: Amazon Fine Food Reviews (~284k balanced samples)
- **Preprocessing**: NLTK tokenization, lemmatization, stopword removal
- **Classes**: Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars)

## 📊 Model Performance

```
              precision    recall  f1-score   support
    negative       0.79      0.73      0.76      8204
     neutral       0.65      0.58      0.61      4264
    positive       0.87      0.91      0.89     44377

    accuracy                           0.84     56845
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Cherry28831/Deep-Learning-Based-Sentiment-Analyzer-for-E-Commerce-Reviews-Using-LSTM.git
cd Deep-Learning-Based-Sentiment-Analyzer-for-E-Commerce-Reviews-Using-LSTM
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── models/              # Trained LSTM model and tokenizer
│   ├── lstm_sentiment.h5
│   └── tokenizer.pkl
├── notebooks/           # Training scripts
│   └── train.py
├── src/                 # Utility functions
│   └── utils.py
├── logs/                # Training logs and visualizations
├── sample_data/         # Test CSV files
├── app.py              # Main Streamlit application
└── requirements.txt    # Dependencies
```

## 🔧 Usage

### Web Interface
1. Enter a review in the text area
2. Click "Analyze" to get sentiment prediction
3. Upload CSV files for batch processing
4. View dynamic word clouds based on your inputs

### CSV Format
For batch processing, use CSV with a "Text" column:
```csv
Text
"This product is amazing!"
"Poor quality, disappointed"
"Average product, nothing special"
```

### Programmatic Usage
```python
from src.utils import predict_sentiment
import joblib
from tensorflow.keras.models import load_model

# Load model and tokenizer
model = load_model("models/lstm_sentiment.h5")
tokenizer = joblib.load("models/tokenizer.pkl")

# Predict sentiment
result = predict_sentiment("Great product!", tokenizer, model)
print(result)  # ('positive', 0.92)
```

## 🧠 Model Details

- **Architecture**: Embedding → LSTM(64) → LSTM(32) → Dense(3)
- **Training**: 15 epochs, Adam optimizer, categorical crossentropy
- **Vocabulary**: 5,000 most frequent words
- **Sequence Length**: 100 tokens
- **Dataset Split**: 60% train, 20% validation, 20% test

## 📈 Training Process

The model was trained on 284k balanced samples with the following progression:
- Epoch 1: 64% accuracy → Epoch 15: 85% accuracy
- Validation accuracy: 81%
- F1-Score (macro): 0.75

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Amazon Fine Food Reviews dataset from Kaggle
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework
- NLTK for natural language processing

---

**Built with ❤️ using Python, TensorFlow, and Streamlit**
