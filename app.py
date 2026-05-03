import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime
import requests

from newsapi import NewsApiClient
from transformers import pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- API Keys (stored in .streamlit/secrets.toml) ---
news_api_key = st.secrets["NEWS_API_KEY"]
marketaux_key = st.secrets["MARKETAUX_KEY"]

# --- Page config ---
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title('Stock Trend Prediction')

# --- User input ---
start = '2010-01-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end, auto_adjust=True)

# Flatten multi-index columns (newer yfinance versions return multi-index)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

if df.empty:
    st.error("No data found for this ticker. Please check the symbol and try again.")
    st.stop()

# --- Describing Data ---
st.subheader(f'Data from 2010 to {datetime.datetime.now().year}')
st.write(df.describe())

# --- Chart 1: Closing Price ---
st.subheader('Closing Price vs Time Chart')
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.Close, label='Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)
plt.close(fig1)

# --- Chart 2: With 100MA ---
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.Close, label='Close Price')
ax2.plot(ma100, 'r', label='100 MA')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

# --- Chart 3: With 100MA & 200MA ---
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df.Close, label='Close Price')
ax3.plot(ma100, 'r', label='100 MA')
ax3.plot(ma200, 'g', label='200 MA')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)
plt.close(fig3)

# --- Train/Test Split ---
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# --- Scale on training data only ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_training)  # fit ONLY on training — no leakage

# --- Load model ---
model = load_model('models/keras_model1.keras')

# --- Prepare test input ---
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)  # transform only — scaler already fit on training

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# --- Predict ---
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]  # dynamic — works for any stock
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# --- Prediction Chart ---
st.subheader('Prediction vs Original')
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(y_test, 'b', label='Original Price')
ax4.plot(y_predicted, 'r', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
plt.close(fig4)

# --- Metrics ---
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

st.write("**Model Evaluation Metrics:**")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

# --- News Sentiment ---
st.subheader("News Sentiment Analysis — Recommendation")

company_name = user_input
if user_input:
    try:
        ticker_info = yf.Ticker(user_input).info
        company_name = ticker_info.get("longName", user_input)
        st.write(f"Detected company: **{company_name}**")
    except Exception as e:
        st.warning(f"Could not fetch company name: {e}")


def fetch_marketaux_news(api_key, query, limit=25):
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": api_key,
        "symbols": query.upper(),
        "language": "en",
        "limit": limit,
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        st.error(f"MarketAux error: {e}")
        return []


@st.cache_resource(show_spinner=False)
def get_finbert_pipeline(device_id=-1):
    model_name = "ProsusAI/finbert"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device_id)


device_id = 0 if torch.cuda.is_available() else -1
finbert = get_finbert_pipeline(device_id=device_id)


def fetch_all_news(news_api_key, marketaux_key, query, limit=25):
    all_articles = []

    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        data = newsapi.get_everything(
            q=query, language="en", sort_by="relevancy", page_size=limit
        )
        for art in data.get("articles", []):
            all_articles.append({
                "title": art["title"],
                "description": art.get("description", ""),
                "source": art["source"]["name"],
                "url": art["url"]
            })
    except Exception as e:
        st.warning(f"NewsAPI fetch error: {e}")

    try:
        for art in fetch_marketaux_news(marketaux_key, query, limit):
            all_articles.append({
                "title": art["title"],
                "description": art.get("description", ""),
                "source": art.get("source", "MarketAux"),
                "url": art["url"]
            })
    except Exception as e:
        st.warning(f"MarketAux fetch error: {e}")

    seen = set()
    unique_articles = []
    for a in all_articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique_articles.append(a)

    return unique_articles


def aggregate_sentiment(preds):
    score = 0.0
    for p in preds:
        lbl = p["label"].lower()
        s = p["score"]
        if "positive" in lbl:
            score += s
        elif "negative" in lbl:
            score -= s
    return score / len(preds) if preds else 0


def sentiment_to_action(s, buy=0.15, sell=-0.15):
    if s >= buy:
        return "BUY"
    elif s <= sell:
        return "SELL"
    else:
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
                action = sentiment_to_action(avg_sentiment)

                st.write(f"### Overall Sentiment: `{avg_sentiment:.3f}` → **{action}**")
                st.markdown("---")

                for art, res in zip(articles[:5], results[:5]):
                    if not isinstance(art, dict):
                        continue
                    title = art.get("title", "No title")
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