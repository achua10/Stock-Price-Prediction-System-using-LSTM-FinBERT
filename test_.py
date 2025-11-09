import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import matplotlib.pyplot as plt

#load data
start = '2010-01-01'
end = datetime.date.today().strftime('%Y-%m-%d')
ticker = 'AAPL'  # <-- change stock symbol here

print(f"Downloading {ticker} data from {start} to {end}...")
df = yf.download(ticker, start=start, end=end, auto_adjust=True)

#split into t n t
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

#scale n prep
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load trained model
model = load_model('keras_model1.keras')

# Prepare test input
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# ----------------------------------------
# 4. Predict
# ----------------------------------------
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#eval perf
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

print("\n📊 Model Evaluation (on data up to 2025):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Directional Accuracy
real_movement = np.sign(np.diff(y_test))
predicted_movement = np.sign(np.diff(y_predicted))
direction_accuracy = np.mean(real_movement == predicted_movement) * 100
print(f"Directional (Up/Down) Accuracy: {direction_accuracy:.2f}%")

#resultt
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{ticker} Price Prediction vs Original (Up to {end})')
plt.legend()
plt.show()
