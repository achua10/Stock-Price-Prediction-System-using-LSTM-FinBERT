import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import datetime
import requests

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping

from newsapi import NewsApiClient
from transformers import pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- API Keys ---
news_api_key = st.secrets["NEWS_API_KEY"]
marketaux_key = st.secrets["MARKETAUX_KEY"]

# --- Page config ---
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("Stock Trend Prediction")

# --- Constants ---
WINDOW      = 60    # lookback window in days (was 100 — shorter = more responsive)
TRAIN_SPLIT = 0.70  # 70% train, 30% test

# --- User input ---
end = datetime.datetime.now().strftime('%Y-%m-%d')
user_input = st.text_input('Enter Stock Ticker', 'AAPL').strip().upper()


# ---------------------------------------------------------------------------
# Model architecture — reduced dropout so predictions aren't over-smoothed
# ---------------------------------------------------------------------------
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, activation='relu', return_sequences=True),
        Dropout(0.1),                                          # was 0.2
        LSTM(60, activation='relu', return_sequences=True),
        Dropout(0.2),                                          # was 0.3
        LSTM(80, activation='relu', return_sequences=True),
        Dropout(0.2),                                          # was 0.3
        LSTM(120, activation='relu'),
        Dropout(0.3),                                          # was 0.5
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ---------------------------------------------------------------------------
# Train on returns — cached per ticker
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model_for_ticker(ticker: str):
    """
    Downloads data, converts close prices to daily % returns,
    trains LSTM on returns, and returns everything needed for inference.
    """
    df = yf.download(ticker, start='2010-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'), auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # --- Convert to daily returns ---
    # pct_change() gives (today - yesterday) / yesterday
    # dropna() removes the first NaN row
    close_prices = df['Close'].copy()
    returns = close_prices.pct_change().dropna()

    # Keep the actual prices aligned to returns index for reconstruction later
    aligned_prices = close_prices.loc[returns.index]

    # --- Train/test split on returns ---
    split_idx = int(len(returns) * TRAIN_SPLIT)
    train_returns = returns.iloc[:split_idx].values.reshape(-1, 1)
    test_returns  = returns.iloc[split_idx:].values.reshape(-1, 1)

    # --- Scale returns (they're already small but scaling helps LSTM) ---
    scaler = MinMaxScaler(feature_range=(-1, 1))   # symmetric around 0 for returns
    train_scaled = scaler.fit_transform(train_returns)

    # --- Build sequences ---
    x_train, y_train = [], []
    for i in range(WINDOW, len(train_scaled)):
        x_train.append(train_scaled[i - WINDOW:i])
        y_train.append(train_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # --- Train ---
    model = build_model((x_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=50, batch_size=32,
              callbacks=[early_stop], verbose=0)

    return model, scaler, df, returns, aligned_prices, split_idx


# ---------------------------------------------------------------------------
# Reconstruct price from predicted returns
# ---------------------------------------------------------------------------
def reconstruct_prices(actual_returns, predicted_returns_scaled, scaler, start_price):
    """
    Inverse-scale predicted returns, then compound them starting from
    the last known price before the test window.
    """
    pred_returns = scaler.inverse_transform(predicted_returns_scaled.reshape(-1, 1)).flatten()
    actual_ret   = actual_returns.flatten()

    # Compound from start_price
    pred_prices   = [start_price]
    actual_prices = [start_price]

    for r_pred, r_actual in zip(pred_returns, actual_ret):
        pred_prices.append(pred_prices[-1] * (1 + r_pred))
        actual_prices.append(actual_prices[-1] * (1 + r_actual))

    return np.array(actual_prices[1:]), np.array(pred_prices[1:])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
if user_input:
    with st.spinner(f"Training model for **{user_input}**... ~60 seconds on first load, instant after."):
        try:
            model, scaler, df, returns, aligned_prices, split_idx = train_model_for_ticker(user_input)
        except Exception as e:
            st.error(f"Could not load data for **{user_input}**: {e}")
            st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # --- Data summary ---
    st.subheader(f"Data from 2010 to {datetime.datetime.now().year}")
    st.write(df.describe())

    # --- Chart 1: Closing Price ---
    st.subheader("Closing Price vs Time Chart")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.Close, label="Close Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # --- Chart 2: 100MA ---
    st.subheader("Closing Price vs Time Chart with 100MA")
    ma100 = df.Close.rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df.Close, label="Close Price")
    ax2.plot(ma100, 'r', label="100 MA")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Chart 3: 100MA & 200MA ---
    st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
    ma200 = df.Close.rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df.Close, label="Close Price")
    ax3.plot(ma100, 'r', label="100 MA")
    ax3.plot(ma200, 'g', label="200 MA")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    # --- Prepare test sequences ---
    test_returns_raw = returns.iloc[split_idx:].values.reshape(-1, 1)

    # Need last WINDOW returns from training to seed the first test sequence
    all_returns = returns.values.reshape(-1, 1)
    all_scaled  = scaler.transform(all_returns)

    x_test, y_test_scaled = [], []
    for i in range(split_idx, len(all_scaled)):
        if i - WINDOW < 0:
            continue
        x_test.append(all_scaled[i - WINDOW:i])
        y_test_scaled.append(all_scaled[i, 0])

    x_test         = np.array(x_test)
    y_test_scaled  = np.array(y_test_scaled)

    # --- Predict ---
    y_pred_scaled = model.predict(x_test)

    # --- Reconstruct prices from returns ---
    # Start price = last closing price of training set
    start_price = float(aligned_prices.iloc[split_idx - 1])
    actual_test_returns = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    actual_prices, pred_prices = reconstruct_prices(
        actual_test_returns, y_pred_scaled, scaler, start_price
    )

    # --- Prediction Chart ---
    st.subheader("Prediction vs Original")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(actual_prices, 'b', label="Original Price")
    ax4.plot(pred_prices,   'r', label="Predicted Price")
    ax4.set_xlabel("Trading Days (Test Period)")
    ax4.set_ylabel("Price")
    ax4.legend()
    st.pyplot(fig4)
    plt.close(fig4)

    # --- Metrics (on reconstructed prices) ---
    min_len = min(len(actual_prices), len(pred_prices))
    actual_prices = actual_prices[:min_len]
    pred_prices   = pred_prices[:min_len]

    mse  = mean_squared_error(actual_prices, pred_prices)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actual_prices, pred_prices)
    r2   = r2_score(actual_prices, pred_prices)

    st.write("**Model Evaluation Metrics:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE",  f"{mae:.2f}")
    col3.metric("R² Score", f"{r2:.3f}")

    # ---------------------------------------------------------------------------
    # News Sentiment
    # ---------------------------------------------------------------------------
    st.subheader("News Sentiment Analysis — Recommendation")

    company_name = user_input
    try:
        ticker_info  = yf.Ticker(user_input).info
        company_name = ticker_info.get("longName", user_input)
        st.write(f"Detected company: **{company_name}**")
    except Exception as e:
        st.warning(f"Could not fetch company name: {e}")

    def fetch_marketaux_news(api_key, query, limit=25):
        url    = "https://api.marketaux.com/v1/news/all"
        params = {"api_token": api_key, "symbols": query.upper(), "language": "en", "limit": limit}
        try:
            response = requests.get(url, params=params)
            return response.json().get("data", [])
        except Exception as e:
            st.error(f"MarketAux error: {e}")
            return []

    @st.cache_resource(show_spinner=False)
    def get_finbert_pipeline(device_id=-1):
        model_name = "ProsusAI/finbert"
        return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device_id)

    device_id = 0 if torch.cuda.is_available() else -1
    finbert   = get_finbert_pipeline(device_id=device_id)

    def fetch_all_news(napi_key, maux_key, query, limit=25):
        all_articles = []
        try:
            newsapi = NewsApiClient(api_key=napi_key)
            data    = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=limit)
            for art in data.get("articles", []):
                all_articles.append({
                    "title":       art["title"],
                    "description": art.get("description", ""),
                    "source":      art["source"]["name"],
                    "url":         art["url"]
                })
        except Exception as e:
            st.warning(f"NewsAPI fetch error: {e}")

        try:
            for art in fetch_marketaux_news(maux_key, query, limit):
                all_articles.append({
                    "title":       art["title"],
                    "description": art.get("description", ""),
                    "source":      art.get("source", "MarketAux"),
                    "url":         art["url"]
                })
        except Exception as e:
            st.warning(f"MarketAux fetch error: {e}")

        seen, unique = set(), []
        for a in all_articles:
            if a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)
        return unique

    def aggregate_sentiment(preds):
        score = 0.0
        for p in preds:
            lbl = p["label"].lower()
            s   = p["score"]
            if "positive" in lbl:
                score += s
            elif "negative" in lbl:
                score -= s
        return score / len(preds) if preds else 0

    def sentiment_to_action(s, buy=0.15, sell=-0.15):
        if s >= buy:   return "BUY"
        if s <= sell:  return "SELL"
        return "HOLD"

    if st.button("Analyze News Sentiment"):
        try:
            with st.spinner(f"Fetching latest news about {company_name}..."):
                articles = fetch_all_news(news_api_key, marketaux_key, company_name)

            if not articles:
                st.warning("No news found from either NewsAPI or MarketAux.")
            else:
                st.success(f"Retrieved {len(articles)} unique news articles.")
                texts = [
                    (a.get("title", "") or "") + ". " + (a.get("description", "") or "")
                    for a in articles if isinstance(a, dict)
                ]
                texts = [t for t in texts if t.strip()]

                if not texts:
                    st.warning("No valid article text to analyze.")
                else:
                    with st.spinner("Analyzing sentiment with FinBERT..."):
                        results = finbert(texts)

                    avg_sentiment = aggregate_sentiment(results)
                    action        = sentiment_to_action(avg_sentiment)

                    st.write(f"### Overall Sentiment: `{avg_sentiment:.3f}` → **{action}**")
                    st.markdown("---")

                    for art, res in zip(articles[:5], results[:5]):
                        if not isinstance(art, dict):
                            continue
                        title  = art.get("title", "No title")
                        source = art.get("source", "Unknown")
                        if isinstance(source, dict):
                            source = source.get("name", "Unknown")
                        score_label = res.get("label", "Neutral")
                        score_value = res.get("score", 0.0)
                        st.write(f"**{score_label} ({score_value:.2f})** — {title} — *{source}*")
                        if art.get("url"):
                            st.markdown(f"[Read more]({art['url']})")
                        st.markdown("---")

        except Exception as e:
            st.error(f"Error fetching or analyzing news: {e}")