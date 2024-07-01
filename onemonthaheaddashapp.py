import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import callback_context
import csv
from datetime import datetime

# Load the LSTM model
model = load_model('my_model.keras')

# Define Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define layout for the Prediction feature
prediction_layout = html.Div([
    html.H1('Stock Prediction Dashboard - Prediction'),
    dcc.Input(id='stock_symbol', placeholder='Enter stock symbol...', type='text'),
    dcc.DatePickerRange(id='date_range', display_format='YYYY-MM-DD'),
    html.Button('Predict', id='predict_button'),
    html.Div(id='prediction_output')
])

# Define layout for the Watchlist feature
watchlist_layout = html.Div([
    html.H1('Stock Prediction Dashboard - Watchlist'),
    dcc.Input(id='watchlist_stock_symbol', placeholder='Enter stock symbols separated by commas...', type='text'),
    html.Button('Add Stock to Watchlist', id='add_watchlist_button'),
    html.Div(id='watchlist_container', children=[])  # Container for displaying watchlist items
])

# Define layout
app.layout = html.Div([
    html.H1('Stock Prediction Dashboard'),
    html.Div([
        dbc.Button('Prediction', id='prediction_button', style={'margin-right': '10px'}),
        dbc.Button('Watchlist', id='watchlist_button')
    ], style={'text-align': 'center', 'margin-top': '50px'}),
    html.Div(id='content'),
    dcc.Interval(id='interval-component', interval=30 * 1000, n_intervals=0),  # Update every 30 seconds
], id='main-layout')

@app.callback(
    Output('content', 'children'),
    [Input('prediction_button', 'n_clicks'),
     Input('watchlist_button', 'n_clicks')],
    [State('content', 'children')]
)
def switch_content(prediction_clicks, watchlist_clicks, current_layout):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'prediction_button':
        return prediction_layout
    elif (triggered_id == 'watchlist_button') and (watchlist_layout is not None):
        return watchlist_layout
    return current_layout  # This line can help retain the current layout if no buttons have been clicked

@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('stock_symbol', 'value'),
     State('date_range', 'start_date'),
     State('date_range', 'end_date')]
)
def predict_stock_price(n_clicks, stock_symbols, start_date, end_date):
    if not all([stock_symbols, start_date, end_date]):
        return "Please fill in all fields."

    symbols = stock_symbols.split(',')
    if len(symbols) > 2:
        return "Please enter one or two stock symbols separated by a comma."

    graph_data = []
    prediction_details = []
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler inside the function

    for index, symbol in enumerate(symbols):
        symbol = symbol.strip()
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return f"No data available for {symbol}"
        
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
        data.dropna(inplace=True)

        if data.empty:
            return f"Insufficient non-NaN data after indicator computation for {symbol}"

        # Prepare data for model input
        inputs = scaler.fit_transform(data[['Close', 'SMA', 'EMA', 'BB_upper', 'BB_lower', 'RSI']])

        # Make one month ahead predictions
        X_test = np.array([inputs[-60:]])  # Last 60 days to predict the next month
        predicted_prices = []
        for _ in range(22):  # Assuming 22 trading days in a month
            next_day_prediction = model.predict(X_test)
            predicted_prices.append(next_day_prediction[0, 0])
            next_day_input = np.roll(X_test, -1, axis=1)
            next_day_input[0, -1, 0] = next_day_prediction
            X_test = next_day_input

        # Inverse transform the predicted prices
        predicted_prices = scaler.inverse_transform(
            np.hstack((np.array(predicted_prices).reshape(-1, 1), np.zeros((22, inputs.shape[1] - 1)))))[:, 0]

        # Combine actual and predicted data
        predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=22, freq='B')  # B for business days
        combined_dates = data.index.append(predicted_dates)
        combined_prices = np.append(data['Close'].values, predicted_prices)

        # Plot the results
        graph_data.append(dcc.Graph(
            id=f'stock_prices_{symbol}',
            figure={
                'data': [
                    {'x': combined_dates, 'y': combined_prices, 'type': 'line', 'name': 'Predicted Price', 'line': {'color': 'green'}},
                    {'x': data.index, 'y': data['Close'], 'type': 'line', 'name': 'Actual Price', 'line': {'color': 'blue'}}
                ],
                'layout': {
                    'title': f'{symbol} Stock Predictions',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Price'}
                }
            }
        ))

        # Calculate percentage changes
        last_close = data['Close'].iloc[-1]
        percentage_changes = [(predicted_prices[i] - last_close) / last_close * 100 for i in range(len(predicted_prices))]

        # Display predicted prices and percentage changes
        for date, price, change in zip(predicted_dates, predicted_prices, percentage_changes):
            prediction_details.append(html.P(f"Predicted close for {symbol} on {date.date()}: ${price:.2f} ({change:+.2f}%)"))

    return graph_data + prediction_details

@app.callback(
    Output('watchlist_container', 'children'),
    [Input('add_watchlist_button', 'n_clicks')],
    [State('watchlist_stock_symbol', 'value'),
     State('watchlist_container', 'children')]
)
def add_stock_to_watchlist(n_clicks, stock_symbols, current_watchlist):
    if not stock_symbols:
        return "Please enter stock symbols."
    stock_symbols = [symbol.strip().upper() for symbol in stock_symbols.split(',')]  # Split and clean input
    updated_watchlist = list(current_watchlist) if isinstance(current_watchlist, str) else current_watchlist

    if n_clicks:
        for stock_symbol in stock_symbols:
            if stock_symbol:  # Ensure non-empty strings
                try:
                    data = yf.Ticker(stock_symbol).history(period='1mo')  # Fetch data for the last month
                    latest_price = data['Close'].iloc[-1]
                    graph = dcc.Graph(
                        id={'type': 'watchlist-graph', 'index': stock_symbol},
                        figure={
                            'data': [{'x': data.index, 'y': data['Close'], 'type': 'line', 'name': 'Actual Stock Price'}],
                            'layout': {'title': f'{stock_symbol} Actual Stock Price'}
                        }
                    )
                    updated_watchlist.append(html.Div([html.Div(f'{stock_symbol}: ${latest_price:.2f}'), graph]))
                except Exception as e:
                    updated_watchlist.append(html.Div(f"Error retrieving data for {stock_symbol}: {str(e)}"))

    return updated_watchlist

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
