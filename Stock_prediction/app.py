from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

app = Flask(__name__)

valid_stock_key = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN']

def grid_search_arima(data, p_values, d_values, q_values):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_model = model_fit
        except Exception as e:
            print(f"Error fitting ARIMA({p},{d},{q}): {e}")
            continue
    return best_order, best_model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        if stock_symbol not in valid_stock_key:
            return render_template('index.html', error="Invalid stock key!")

        stock_data = yf.Ticker(stock_symbol)
        historical_prices = stock_data.history(period="1d", start="2010-01-01", end="2010-08-19")
        historical_prices = historical_prices[['Close']].dropna()
        historical_prices.index = pd.to_datetime(historical_prices.index)
        historical_prices = historical_prices.asfreq('B').ffill()  # Updated to use ffill()

        train_size = int(len(historical_prices) * 0.8)
        train, test = historical_prices[:train_size], historical_prices[train_size:]

        p_values = range(0, 6)
        d_values = range(0, 3)
        q_values = range(0, 6)

        best_order, best_model = grid_search_arima(train, p_values, d_values, q_values)

        if best_model is None:
            return render_template('index.html', error="Failed to fit ARIMA model. Please try again.")

        predictions = best_model.forecast(steps=len(test))
        test['Predictions'] = predictions.values

        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train['Close'], label='Training Data')
        plt.plot(test.index, test['Close'], label='Actual Prices')
        plt.plot(test.index, test['Predictions'], label='Predicted Prices', color='red')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.title(f'{stock_symbol} Stock Price Prediction using ARIMA')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()  # Close the plot to free up memory

        return render_template('result.html', plot_url=plot_url, best_order=best_order, summary=best_model.summary().as_text())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
