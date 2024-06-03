def arima_test(section, csv_file):
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    df = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')

    df.rename(columns={'Average Price': 'average_price'}, inplace=True)

    train = df.iloc[:211]
    test = df.iloc[10:]

    model = ARIMA(train['average_price'], order=(1, 1, 1))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=10)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    mre = np.mean(np.abs((test['average_price'] - mean_forecast) / test['average_price']))
    print(f"Mean Relative Error: {mre * 100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(df['average_price'], label='Observed (Complete Data)', color='blue')

    combined_index = train.index.union(mean_forecast.index)
    combined_data = pd.concat([train['average_price'], mean_forecast])

    plt.plot(combined_index, combined_data, color='red', label='Forecast')
    plt.fill_between(mean_forecast.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('Average Price Forecast with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.legend()
    plt.savefig(f"custom.png", format='png')
    plt.close()

arima_test('test_section', 'daily_average_prices.csv')


