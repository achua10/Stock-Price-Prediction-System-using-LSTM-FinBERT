from flask import Flask, request, jsonify
from keras.models import load_model
from transformers import pipeline
import yfinance as yf
import numpy as np

app = Flask(__name__)

# Load your model
model = load_model('models/keras_model1.keras')
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json['ticker']
    df = yf.download(ticker, start='2010-01-01')
    price = float(df['Close'].iloc[-1])
    predicted_price = price * 1.02  # mock logic for testing
    return jsonify({'ticker': ticker, 'predictedPrice': round(predicted_price, 2)})

@app.route('/sentiment', methods=['POST'])
def sentiment():
    ticker = request.json['ticker']
    sentiment = finbert(f"{ticker} stock outlook")[0]
    return jsonify({'ticker': ticker, 'sentiment': sentiment['label']})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
