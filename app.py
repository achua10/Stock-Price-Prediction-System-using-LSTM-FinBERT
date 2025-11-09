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


news_api_key = "70b04dd005444746a85107949248b837" # NewsAPI key

marketaux_key="za1OVrPQEQaAC213DjPBfHXCAv03ZXd0KlD1A23h"




st.title('Stock Trend Prediction')

start = '2010-01-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')
user_input=st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start, end=end, auto_adjust=True)

#Describing Data

st.subheader('Data from 2010-2025')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#SPLITTINGG THE DATA INTO TRAINING AND TESTING
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])  #splitting done hereee
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#Load my model
model=load_model('keras_model1.keras')

#testing part
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df) 
# BUT first ensure data_testing values are within training range
input_data = np.clip(input_data, 0, 1)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

st.write(f"📊 **Model Evaluation Metrics:**")
st.write(f"- RMSE (Root Mean Squared Error): {rmse:.2f}")
st.write(f"- MAE (Mean Absolute Error): {mae:.2f}")
st.write(f"- R² Score: {r2:.3f}")




st.subheader("📰 News Sentiment Analysis : RECOMMENDATION")

# Automatically infer company name from ticker
if user_input:  # user_input is your ticker, e.g. 'AAPL'
    try:
        ticker_info = yf.Ticker(user_input).info
        company_name = ticker_info.get("longName", user_input)
        st.write(f"Detected company: **{company_name}**")
    except Exception as e:
        st.warning(f"Could not fetch company name: {e}")
        company_name = user_input
else:
    company_name = "Apple Inc"



#markettttaux news fetching

def fetch_marketaux_news(api_key, query, limit=25):
    """Fetch financial news from MarketAux API."""
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
        if "data" in data:
            return data["data"]
        else:
            return []
    except Exception as e:
        st.error(f"MarketAux error: {e}")
        return []


# Load FinBERT once
@st.cache_resource(show_spinner=False)
def get_finbert_pipeline(device_id=-1):
    model_name = "ProsusAI/finbert"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device_id)

device_id = 0 if torch.cuda.is_available() else -1
finbert = get_finbert_pipeline(device_id=device_id)

# Fetch latest financial news via NewsAPI
def fetch_all_news(news_api_key, marketaux_key, query, limit=25):
    """Fetch and combine news from both NewsAPI and MarketAux."""
    all_articles = []

    # --- From NewsAPI ---
    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        data = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=limit
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

    # --- From MarketAux ---
    try:
        marketaux_data = fetch_marketaux_news(marketaux_key, query, limit)
        for art in marketaux_data:
            all_articles.append({
                "title": art["title"],
                "description": art.get("description", ""),
                "source": art.get("source", "MarketAux"),
                "url": art["url"]
            })
    except Exception as e:
        st.warning(f"MarketAux fetch error: {e}")

    # --- Deduplicate by title ---
    seen = set()
    unique_articles = []
    for a in all_articles:
        key = a["title"]
        if key not in seen:
            seen.add(key)
            unique_articles.append(a)

    return unique_articles


# Sentiment aggregation helpers
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

# Run FinBERT sentiment when button pressed
# --- 🔍 Analyze News Sentiment Section ---
if st.button("Analyze News Sentiment"):
    try:
        # ✅ Use pre-set API keys
        if not news_api_key or not marketaux_key:
            st.error("❌ Please make sure both NewsAPI and MarketAux API keys are set.")
        else:
            with st.spinner(f"📰 Fetching latest news about {company_name}..."):
                articles = fetch_all_news(news_api_key, marketaux_key, company_name)

            if not articles:
                st.warning("⚠️ No news found from either NewsAPI or MarketAux.")
            else:
                st.success(f"✅ Retrieved {len(articles)} unique news articles.")
                texts = [
                    (a.get("title", "") or "") + ". " + (a.get("description", "") or "")
                    for a in articles if isinstance(a, dict)
                ]
                texts = [t for t in texts if t.strip()]  # Remove blanks

                if not texts:
                    st.warning("No valid article text to analyze.")
                else:
                    with st.spinner("🤖 Analyzing sentiment with FinBERT..."):
                        results = finbert(texts)

                    # --- Sentiment Aggregation ---
                    avg_sentiment = aggregate_sentiment(results)
                    action = sentiment_to_action(avg_sentiment)

                    st.write(f"### 🧠 Overall Sentiment: `{avg_sentiment:.3f}` → **{action}**")
                    st.markdown("---")

                    # --- Display Top 5 Analyzed News ---
                    for art, res in zip(articles[:5], results[:5]):
                        if not isinstance(art, dict):
                            continue
                        title = art.get("title", "No title")
                        desc = art.get("description", "")
                        source = art.get("source", "Unknown source")
                        # Handle dict or string source
                        if isinstance(source, dict):
                            source = source.get("name", "Unknown")

                        score_label = res.get("label", "Neutral")
                        score_value = res.get("score", 0.0)
                        st.write(
                            f"**{score_label} ({score_value:.2f})** — {title} — *{source}*"
                        )
                        if art.get("url"):
                            st.markdown(f"[Read more]({art['url']})")
                        st.markdown("---")

    except Exception as e:
        st.error(f"🚨 Error fetching or analyzing news: {e}")
