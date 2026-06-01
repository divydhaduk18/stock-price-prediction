import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

stock = input("Enter stock ticker: ")

future_days = int(
    input("Enter number of future days to predict: ")
)

data = yf.download(
    stock,
    start="2015-01-01",
    end="2026-01-01"
)

print(data.head())

data.to_csv(f"{stock}.csv")

close_data = data['Close']

dataset = close_data.to_numpy()

dataset = dataset.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)

training_data_len = int(len(scaled_data) * 0.8)

train_data = scaled_data[0:training_data_len]

x_train = []
y_train = []

for i in range(60, len(train_data)):

    x_train.append(train_data[i-60:i, 0])

    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(
    x_train,
    (
        x_train.shape[0],
        x_train.shape[1],
        1
    )
)

model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=50,
        return_sequences=False
    )
)

model.add(Dropout(0.2))

model.add(Dense(units=25))

model.add(Dense(units=1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10
)

test_data = scaled_data[training_data_len - 60:]

x_test = []

y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(
    x_test,
    (
        x_test.shape[0],
        x_test.shape[1],
        1
    )
)

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

train = data[:training_data_len]

valid = data[training_data_len:].copy()

valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))

plt.title(f'{stock} Stock Price Prediction')

plt.xlabel('Date')

plt.ylabel('Close Price')

plt.plot(train['Close'])

plt.plot(valid[['Close']])

plt.plot(valid[['Predictions']])

plt.legend([
    'Train',
    'Real Price',
    'Predicted Price'
])

plt.show()

print(valid[['Close', 'Predictions']])

last_60_days = scaled_data[-60:]

future_input = last_60_days.reshape(1, 60, 1)

future_predictions = []

for i in range(future_days):

    pred = model.predict(future_input)

    future_predictions.append(pred[0][0])

    future_input = np.append(
        future_input[:, 1:, :],
        [[[pred[0][0]]]],
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

print("\nFuture Predictions:\n")

for i, price in enumerate(future_predictions):

    print(f"Day {i+1}: ${price[0]:.2f}")

plt.figure(figsize=(12, 6))

plt.plot(
    range(1, future_days + 1),
    future_predictions
)

plt.title(f'{stock} Future Predictions')

plt.xlabel('Future Days')

plt.ylabel('Predicted Price')

plt.show()