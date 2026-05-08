import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
# 1. Load Stock Data
# ---------------------------
stock = "AAPL"  # Apple stock (you can change to TSLA, MSFT, etc.)

df = yf.download(stock, start="2020-01-01", end="2025-01-01")

print("\n--- DATA SAMPLE ---")
print(df.head())

# ---------------------------
# 2. Feature Engineering
# ---------------------------
df = df[["Open", "High", "Low", "Volume", "Close"]]

# Predict next day's closing price
df["Target"] = df["Close"].shift(-1)

df.dropna(inplace=True)

X = df[["Open", "High", "Low", "Volume"]]
y = df["Target"]

# ---------------------------
# 3. Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------------------
# 4. Random Forest Model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 5. Predictions
# ---------------------------
predictions = model.predict(X_test)

# ---------------------------
# 6. Evaluation
# ---------------------------
mse = mean_squared_error(y_test, predictions)
print("\nMean Squared Error:", mse)

# ---------------------------
# 7. Plot Results
# ---------------------------
plt.figure(figsize=(10, 5))

plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title(f"{stock} Stock Price Prediction (Random Forest)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

# ---------------------------
# 8. Save Output
# ---------------------------
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "rf_stock_prediction.png"))
plt.show()

print("\nTask 2 (Random Forest) Completed Successfully!")