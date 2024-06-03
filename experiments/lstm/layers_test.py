import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import l2

# Load data
data = pd.read_csv('daily_average_prices.csv', parse_dates=['Date'], index_col='Date')

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Split data into training, validation, and testing sets (70%, 10%, 20%)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.1)
test_size = len(data) - train_size - val_size

train_data = scaled_data[:train_size]
val_data = scaled_data[train_size:train_size + val_size]
test_data = scaled_data[train_size + val_size:]

# Define time steps
time_steps = 10

# Function to create sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Prepare input sequences
X_train, y_train = create_sequences(train_data, time_steps)
X_val, y_val = create_sequences(val_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Print X_test content
for i, sequence in enumerate(X_test):
    print(f"Sequence {i+1}:")
    print(sequence)

# Build LSTM model with 3 layers, Tanh activation, and L2 regularization
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1), activation='tanh', kernel_regularizer=l2(0.01)))
model.add(LSTM(50, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(LSTM(50, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Predict on test data
predictions = model.predict(X_test)
predictions_inverse = scaler.inverse_transform(predictions)

# Inverse transform of y_test for correct comparison
y_test_inverse = scaler.inverse_transform(y_test)

# Calculate Mean Absolute Error (MAE) and Mean Relative Error (MRE)
mae = mean_absolute_error(y_test_inverse, predictions_inverse)
rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions_inverse))
mre = np.mean(np.abs((y_test_inverse - predictions_inverse) / y_test_inverse))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'MRE: {mre}')

# Prepare full prediction array
full_predictions = np.full(shape=(len(data),), fill_value=np.nan)
full_predictions[train_size + val_size + time_steps:] = predictions_inverse.flatten()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Real Average Price')
plt.plot(data.index, full_predictions, label='Predicted Average Price', color='orange')
plt.title('Real Estate Price Prediction with 3 LSTM Layers, L2 Regularization, and Tanh')
plt.xlabel('Time')
plt.ylabel('Average Price')
plt.legend()
plt.savefig("lstm_prediction_3_layers_l2_tanh.png", format='png')
plt.close()


'''
MAE: 24.98118722098213
RMSE: 29.409862956418074
MRE: 0.010884385525357903
'''
