import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import l2

data = pd.read_csv('daily_average_prices.csv', parse_dates=['Date'], index_col='Date')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.1)
test_size = len(data) - train_size - val_size

train_data = scaled_data[:train_size]
val_data = scaled_data[train_size:train_size + val_size]
test_data = scaled_data[train_size + val_size:]

time_steps = 10
units_count = 50
batch_size = 8
epochs = 50

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, time_steps)
X_val, y_val = create_sequences(val_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)


model = Sequential()
model.add(LSTM(units_count, input_shape=(time_steps, 1), activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

predictions = model.predict(X_test)
predictions_inverse = scaler.inverse_transform(predictions)

y_test_inverse = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inverse, predictions_inverse)
rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions_inverse))
mre = np.mean(np.abs((y_test_inverse - predictions_inverse) / y_test_inverse))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'MRE: {mre}')

full_predictions = np.full(shape=(len(data),), fill_value=np.nan)
full_predictions[train_size + val_size + time_steps:] = predictions_inverse.flatten()

plt.figure(figsize=(10, 6))
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Real Average Price')
plt.plot(data.index, full_predictions, label='Predicted Average Price', color='orange')
plt.title('Real Estate Price Prediction with Single LSTM Layer, L2 Regularization, and Tanh')
plt.xlabel('Time')
plt.ylabel('Average Price')
plt.legend()
plt.savefig("lstm_prediction_batch_8_70y30.png", format='png')
plt.close()


'''
50/50
8
MAE: 73.36728948279273
RMSE: 86.56794549996792
MRE: 0.03181367121906437

16
MAE: 56.715968651107595
RMSE: 67.72628120993785
MRE: 0.024624082761255453

32
MAE: 38.13223459750792
RMSE: 46.21829946412806
MRE: 0.016602001692107904

64
MAE: 41.42919612836235
RMSE: 49.6524983211791
MRE: 0.01804073994050181

128
MAE: 37.95991149129746
RMSE: 43.02010487783007
MRE: 0.016394088429609658

70/30

8
MAE: 14.966657366071416
RMSE: 18.58588999208024
MRE: 0.006472457260162155

16
MAE: 14.30189034598213
RMSE: 17.693936052156054
MRE: 0.006217515441805159

32
MAE: 14.279108537946415
RMSE: 18.008302135022934
MRE: 0.006191490714795061

64
MAE: 13.830573381696416
RMSE: 16.98097514655113
MRE: 0.0060159195852811116

128
MAE: 13.139111328124986
RMSE: 16.2796828815131
MRE: 0.005704738986652635

'''
