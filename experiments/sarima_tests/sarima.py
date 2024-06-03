def sarima_test():
    
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv("daily_average_prices.csv", parse_dates=['Date'], index_col='Date')

    df.rename(columns={'Average Price': 'average_price'}, inplace=True)

    train = df.iloc[:176]
    test = df.iloc[45:]

    model = SARIMAX(train['average_price'], order=(1, 1, 0), seasonal_order=(1, 1, 0, 7))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=45)
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
    plt.title('Average Price Forecast with SARIMA (Weekly Seasonality)')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.legend()

    plot_filename = f"sarima_forecast_80y20_1101107.png"

    plt.savefig(plot_filename, format='png')
    plt.close()

sarima_test()
