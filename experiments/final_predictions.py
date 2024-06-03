import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def lstm():
	data = pd.read_csv('daily_average_prices.csv', parse_dates=['Date'], index_col='Date')

	scaler = MinMaxScaler()
	scaled_data = scaler.fit_transform(data)

	train_size = int(len(data) * 0.5)
	val_size = int(len(data) * 0.1)
	test_size = len(data) - train_size - val_size

	train_data = scaled_data[:train_size]
	val_data = scaled_data[train_size:train_size + val_size]
	test_data = scaled_data[train_size + val_size:]

	time_steps = 5
	units_count = 50
	batch_size = 128
	epochs = 50
    learning_rate = 0.001
	loss_function = 'mean_absolute_error'

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

	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss=loss_function)

	#model.compile(optimizer='adam', loss=loss_function)

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

	return data.index, full_predictions, scaler.inverse_transform(scaled_data)

def sarima_test():
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df = pd.read_csv("daily_average_prices.csv", parse_dates=['Date'], index_col='Date')

    df.rename(columns={'Average Price': 'average_price'}, inplace=True)

    train = df.iloc[:135]
    test = df.iloc[135:221]

    model = SARIMAX(train['average_price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=86)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    mre = np.mean(np.abs((test['average_price'] - mean_forecast) / test['average_price']))
    mae = mean_absolute_error(test['average_price'], mean_forecast)
    rmse = np.sqrt(mean_squared_error(test['average_price'], mean_forecast))

    print(f"Mean Relative Error: {mre * 100:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    return mean_forecast.index, mean_forecast

def arima_test():
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    df = pd.read_csv("daily_average_prices.csv", parse_dates=['Date'], index_col='Date')

    df.rename(columns={'Average Price': 'average_price'}, inplace=True)

    train = df.iloc[:135]
    test = df.iloc[135:221]

    model = ARIMA(train['average_price'], order=(2, 1, 1))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=86)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    mre = np.mean(np.abs((test['average_price'] - mean_forecast) / test['average_price']))
    mae = mean_absolute_error(test['average_price'], mean_forecast)
    rmse = np.sqrt(mean_squared_error(test['average_price'], mean_forecast))

    print(f"Mean Relative Error: {mre * 100:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    return df['average_price'], mean_forecast.index, mean_forecast

date_lstm, lstm_full_predictions, lstm_scaled_data = lstm()
combined_index_sarima, combined_data_sarima = sarima_test()
average_price_arima, forecast_index_arima, mean_forecast_arime = arima_test()


plt.figure(figsize=(10, 5))
plt.plot(average_price_arima, label='Realūs duomenys', color='blue')
plt.plot(forecast_index_arima, mean_forecast_arime, color='red', label='ARIMA modelio prognozė')
plt.plot(date_lstm, lstm_full_predictions, color='orange', label='LSTM modelio prognozė')
plt.plot(combined_index_sarima, combined_data_sarima, color='green', label='SARIMA modelio prognozė')

plt.title('Vidutinės kainos prognozė naudojant skirtingus modelius')
plt.xlabel('Data')
plt.ylabel('Vidutinė buto kaina už kvadratinį metrą Kaune.')
plt.legend()
plt.savefig(f"customBLA.png", format='png')
plt.close()
