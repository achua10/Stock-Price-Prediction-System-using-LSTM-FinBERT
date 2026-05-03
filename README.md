# Stock Price Prediction System using LSTM & FinBERT

A deep learning application that predicts stock price trends using a stacked LSTM neural network and performs real-time news sentiment analysis using FinBERT to generate BUY / HOLD / SELL recommendations.

Built with Streamlit for an interactive web interface, pulling live market data via yfinance and financial news from NewsAPI and MarketAux.

---

## Features

- **LSTM Price Prediction** — 4-layer stacked LSTM model trained on 15 years of historical closing price data, predicting future price direction against actual values
- **Moving Average Charts** — 100-day and 200-day moving averages plotted alongside closing price for trend context
- **Model Evaluation Metrics** — RMSE, MAE, and R² score displayed after each prediction run
- **News Sentiment Analysis** — Aggregates recent financial news from two sources (NewsAPI + MarketAux), runs each article through FinBERT, and produces an overall BUY / HOLD / SELL signal
- **Any Ticker** — Works with any valid stock symbol (AAPL, TSLA, MSFT, NVDA, etc.)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Deep Learning | Keras / TensorFlow |
| NLP / Sentiment | FinBERT (ProsusAI/finbert) via HuggingFace |
| Market Data | yfinance |
| News Data | NewsAPI, MarketAux |
| Data Processing | NumPy, Pandas, scikit-learn |
| Visualisation | Matplotlib |

---

## Project Structure

```
Stock-Price-Prediction-System-using-LSTM-FinBERT/
├── app.py                  # Main Streamlit application
├── ml_api.py               # Flask REST API (model + sentiment endpoints)
├── test_.py                # Offline model evaluation script
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── .gitattributes          # Git LFS config for model files
├── models/
│   └── keras_model1.keras  # Trained LSTM model (Git LFS)
└── notebooks/
    ├── train_model.ipynb   # Full training pipeline
    └── eda_scratch.ipynb   # Exploratory data analysis
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- Git LFS (for downloading the model file)

### 1. Clone the repository

```bash
git lfs install
git clone https://github.com/achua10/Stock-Price-Prediction-System-using-LSTM-FinBERT.git
cd Stock-Price-Prediction-System-using-LSTM-FinBERT
```

> Git LFS is required to pull the `.keras` model file. Without `git lfs install` first, the model file will download as a pointer file and the app will fail to load.

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.streamlit/secrets.toml` file in the project root:

```toml
NEWS_API_KEY = "your_newsapi_key_here"
MARKETAUX_KEY = "your_marketaux_key_here"
```

- Get a free NewsAPI key at [newsapi.org](https://newsapi.org)
- Get a free MarketAux key at [marketaux.com](https://marketaux.com)

> This file is in `.gitignore` and will never be committed.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Model Architecture

The LSTM model is a 4-layer stacked network trained on 70% of historical closing price data (2010 – present) with a 100-day sliding window.

| Layer | Units | Dropout |
|---|---|---|
| LSTM 1 | 50 | 0.2 |
| LSTM 2 | 60 | 0.3 |
| LSTM 3 | 80 | 0.3 |
| LSTM 4 | 120 | 0.5 |
| Dense output | 1 | — |

- Optimizer: Adam
- Loss: Mean Squared Error
- Epochs: 50
- Input: 100-day sliding window of normalised closing prices

To retrain the model on new data, open `notebooks/train_model.ipynb` and run all cells.

---

## Offline Model Evaluation

To evaluate model performance without the Streamlit UI:

```bash
python test_.py
```

This downloads fresh data, runs predictions, and prints RMSE, MAE, R² score, and directional accuracy (% of up/down moves predicted correctly).

---

## REST API (Optional)

A Flask API is included in `ml_api.py` for integrating predictions into other applications.

```bash
python ml_api.py
```

Endpoints:

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| POST | `/predict` | `{"ticker": "AAPL"}` | Predicted next price |
| POST | `/sentiment` | `{"ticker": "AAPL"}` | Sentiment label |

---

## Screenshots


<img width="497" height="566" alt="image" src="https://github.com/user-attachments/assets/fdf5df99-b930-48a9-8db9-14fd20553638" />

<img width="470" height="436" alt="image" src="https://github.com/user-attachments/assets/38d0140f-222d-4670-972b-867752e9af27" />

<img width="467" height="483" alt="image" src="https://github.com/user-attachments/assets/2f39aa91-b068-4f8c-8cb6-a69e006cc8bf" />


---

## Limitations

- The LSTM model predicts price **trend direction**, not exact future prices — it should not be used as sole basis for financial decisions
- NewsAPI free tier limits requests to articles from the past 30 days
- Model was trained on TSLA data — accuracy will vary across different stocks and market conditions
- Sentiment analysis reflects news tone, not market fundamentals

---

## License

This project is for educational purposes. Not financial advice.


