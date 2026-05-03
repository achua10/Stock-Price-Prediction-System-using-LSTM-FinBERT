from flask import Flask, request, jsonify
from keras.models import load_model
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and FinBERT once at startup
model = load_model('models/keras_model1.keras')
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def get_prediction(ticker: str) -> dict:
    """
    Downloads historical data for a ticker, runs it through the LSTM model,
    and returns the last predicted price alongside the last actual price.
    """
    df = yf.download(ticker, start='2010-01-01', auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Flatten multi-index columns (newer yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = pd.DataFrame(df['Close'])

    # Train/test split — same 70/30 as training
    data_training = pd.DataFrame(close[0:int(len(close) * 0.70)])
    data_testing = pd.DataFrame(close[int(len(close) * 0.70):])

    # Scale on training data only — no leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_training)

    # Prepare input — last 100 days of training + test set
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    # Build sequences
    x_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
    x_test = np.array(x_test)

    # Predict and inverse scale
    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor

    last_predicted = float(round(y_predicted[-1][0], 2))
    last_actual = float(round(df['Close'].iloc[-1], 2))

    return {
        "ticker": ticker.upper(),
        "lastActualPrice": last_actual,
        "lastPredictedPrice": last_predicted,
        "direction": "UP" if last_predicted > last_actual else "DOWN"
    }


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'ticker' not in data:
        return jsonify({"error": "Missing 'ticker' in request body"}), 400

    ticker = data['ticker'].strip().upper()

    try:
        result = get_prediction(ticker)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()
    if not data or 'ticker' not in data:
        return jsonify({"error": "Missing 'ticker' in request body"}), 400

    ticker = data['ticker'].strip().upper()

    try:
        # Fetch company name for a more meaningful FinBERT input
        info = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)

        # Analyse a few meaningful phrases rather than one vague string
        texts = [
            f"{company_name} stock performance",
            f"{company_name} earnings outlook",
            f"{ticker} investment outlook",
        ]
        results = finbert(texts)

        # Aggregate scores
        score = 0.0
        for r in results:
            lbl = r["label"].lower()
            s = r["score"]
            if "positive" in lbl:
                score += s
            elif "negative" in lbl:
                score -= s
        avg_score = round(score / len(results), 4)

        if avg_score >= 0.15:
            signal = "BUY"
        elif avg_score <= -0.15:
            signal = "SELL"
        else:
            signal = "HOLD"

        return jsonify({
            "ticker": ticker,
            "company": company_name,
            "sentimentScore": avg_score,
            "signal": signal
        })

    except Exception as e:
        return jsonify({"error": f"Sentiment analysis failed: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "keras_model1.keras", "finbert": "ProsusAI/finbert"})


if __name__ == '__main__':
    app.run(port=5000, debug=True)