import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Stock Price Prediction App")

stock = st.text_input("Enter Stock Symbol", "AAPL")

days = st.slider("Days to Predict", 1, 30, 7)

data = yf.download(stock, start="2015-01-01")

if data.empty:
    st.error("Invalid stock symbol")
    st.stop()

data = data[['Close']]

data['Prediction'] = data[['Close']].shift(-days)

x = np.array(data.drop(['Prediction'], axis=1))[:-days]

y = np.array(data['Prediction'])[:-days]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(x_train, y_train)

future = np.array(data.drop(['Prediction'], axis=1))[-days:]

predictions = model.predict(future)

st.subheader("Predicted Prices")

for i in range(days):
    st.write(f"Day {i+1}: ${predictions[i]:.2f}")

st.subheader("Stock Closing Price Graph")

fig = plt.figure(figsize=(12,6))

plt.plot(data['Close'])

plt.xlabel("Days")

plt.ylabel("Price")

plt.title(f"{stock} Closing Prices")

st.pyplot(fig)