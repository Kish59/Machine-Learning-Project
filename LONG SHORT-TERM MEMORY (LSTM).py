import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv('airline_passengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Passengers']])

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 10
x, y = create_sequences(scaled_data, seq_length)

x = x.reshape((x.shape[0], x.shape[1], 1))

model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, verbose=0)

predicted = model.predict(x)
predicted_rescaled = scaler.inverse_transform(predicted)
actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(df.index[seq_length:], actual_rescaled, label='Actual')
plt.plot(df.index[seq_length:], predicted_rescaled, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('LSTM - Actual vs Predicted Passengers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
