import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fetch live data
symbol = 'TSLA'
data = yf.download(symbol, start='2019-01-01', end='2024-01-01')

# Calculate technical indicators
data['SMA'] = data['Close'].rolling(window=20).mean()
data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
data['BB_upper'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
data['BB_lower'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
delta = data['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Drop rows with NaN values resulting from indicator calculations
data.dropna(inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'SMA', 'EMA', 'BB_upper', 'BB_lower', 'RSI']])

# Prepare the training data
X_train, y_train = [], []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, :])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Build the LSTM model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model with validation split
history = regressor.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
regressor.save('my_model.keras')
# Evaluation on the last part of training data used as validation set
validation_data = X_train[int(0.8 * len(X_train)):]  # Assume last 20% is validation
validation_targets = y_train[int(0.8 * len(y_train)):].reshape(-1, 1)  # Ensure proper shape for scaling

predicted_validation = regressor.predict(validation_data)

# Inverse transform the predicted prices and targets to get the original scale
inverse_predicted_validation = scaler.inverse_transform(np.hstack((predicted_validation, np.zeros((predicted_validation.shape[0], scaled_data.shape[1]-1)))))
inverse_validation_targets = scaler.inverse_transform(np.hstack((validation_targets, np.zeros((validation_targets.shape[0], scaled_data.shape[1]-1)))))

# Calculate evaluation metrics
mse = mean_squared_error(inverse_validation_targets, inverse_predicted_validation)
mae = mean_absolute_error(inverse_validation_targets, inverse_predicted_validation)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((inverse_validation_targets - inverse_predicted_validation) / inverse_validation_targets)) * 100  # MAPE calculation
r_squared = r2_score(inverse_validation_targets, inverse_predicted_validation)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
print(f'R^2 Score: {r_squared}')
# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for '+ symbol)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
