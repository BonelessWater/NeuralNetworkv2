import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from nn import Network  # Import the Network class from nn.py
import random
import math

# Fetch the list of S&P 500 companies
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Define the period for which we need data
start_date = '2020-01-01'
end_date = '2022-12-31'

# Initialize an empty DataFrame to store all the stock data
all_data = pd.DataFrame()

# Connect to the SQLite3 database
db_name = 'sp500_stock_data.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

for ticker in sp500_tickers:
    try:
        print(f"Processing {ticker}...")
        
        # Query data for the current ticker from the database
        query = f"""
        SELECT date, open, high, low, close, adjusted_close, volume
        FROM stock_data
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """
        data = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date), parse_dates=['date'])
        
        if data.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        data.set_index('date', inplace=True)  # Set date as the index

        # Calculate percentage change between open and close prices
        data['PercentChange'] = (data['close'] - data['open']) / data['open']
        
        # Calculate RSI
        def calculate_RSI(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        data['RSI'] = calculate_RSI(data['close'])

        # Calculate Bollinger Bands
        def calculate_bollinger_bands(prices, window=20):
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            return upper_band, lower_band

        data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data['close'])

        # Calculate MACD
        def calculate_MACD(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_diff = macd - macd_signal
            return macd, macd_signal, macd_diff

        data['MACD'], data['MACD_Signal'], data['MACD_Diff'] = calculate_MACD(data['close'])

        # Calculate Stochastic Oscillator
        def calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
            highest_high = high.rolling(window=k_period).max()
            lowest_low = low.rolling(window=k_period).min()
            K = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            D = K.rolling(window=d_period).mean()
            return K, D

        data['%K'], data['%D'] = calculate_stochastic_oscillator(data['high'], data['low'], data['close'])

        # Calculate On-Balance Volume (OBV)
        data['OBV'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()

        # Calculate Average True Range (ATR)
        def calculate_ATR(high, low, close, window=14):
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr

        data['ATR'] = calculate_ATR(data['high'], data['low'], data['close'])

        # Calculate Williams %R
        def calculate_williams_r(high, low, close, period=14):
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            return williams_r

        data['Williams %R'] = calculate_williams_r(data['high'], data['low'], data['close'])

        # Calculate Commodity Channel Index (CCI)
        def calculate_CCI(high, low, close, window=20):
            tp = (high + low + close) / 3
            sma = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
            cci = (tp - sma) / (0.015 * mad)
            return cci

        data['CCI'] = calculate_CCI(data['high'], data['low'], data['close'])

        # Normalize and prepare the data
        data['RSI'] = data['RSI'].apply(lambda x: (x - 50) / 50)  # Normalized to [-1, 1]
        data['Bollinger'] = np.where(data['close'] > data['Upper Band'], (data['close'] - data['Upper Band']) / data['close'], 
                                     np.where(data['close'] < data['Lower Band'], (data['Lower Band'] - data['close']) / data['close'], 0))
        data['MACD'] = data['MACD_Diff'] / data['MACD_Diff'].abs().max()  # Normalizing to [-1, 1]
        data['Stochastic'] = (data['%K'] - data['%D']) / 100
        data['OBV'] = data['OBV'] / data['OBV'].abs().max()  # Normalize OBV to [-1, 1]
        data['ATR'] = data['ATR'] / data['close']  # Normalized ATR to relative value
        data['Williams %R'] = data['Williams %R'] / 100  # Normalizing to [-1, 1]
        data['CCI'] = data['CCI'] / 100  # Normalizing to [-1, 1]

        # Fill inconclusive values with 0
        data['RSI'] = data['RSI'].apply(lambda x: x if abs(x) > 0.2 else 0)
        data['Bollinger'] = data['Bollinger'].apply(lambda x: x if abs(x) > 0.2 else 0)
        data['MACD'] = data['MACD'].apply(lambda x: x if abs(x) > 0.1 else 0)
        data['Stochastic'] = data['Stochastic'].apply(lambda x: x if abs(x) > 0.1 else 0)
        data['OBV'] = data['OBV'].apply(lambda x: x if abs(x) > 0.1 else 0)
        data['ATR'] = data['ATR'].apply(lambda x: x if abs(x) > 0.01 else 0)
        data['Williams %R'] = data['Williams %R'].apply(lambda x: x if abs(x) > 0.2 else 0)
        data['CCI'] = data['CCI'].apply(lambda x: x if abs(x) > 0.2 else 0)

        # Combine the current stock's data with the aggregate DataFrame
        all_data = pd.concat([all_data, data[['RSI', 'Bollinger', 'MACD', 'Stochastic', 'OBV', 'ATR', 'Williams %R', 'CCI', 'PercentChange']].dropna()], axis=0)
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Close the database connection
conn.close()

# Check if all_data is empty before proceeding
if all_data.empty:
    print("No data available for training or prediction.")
else:
    print(all_data.tail())  # Check the aggregated data

    # Define the network structure
    network_structure = [8, 9, 9, 1]  # Assuming 8 input features

    # Initialize the neural network
    best_network = Network(network_structure)

    # Set the number of epochs
    num_epochs = 10

    # Training Phase
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for _ in range(200):  # Train with 200 random samples per epoch
            sample = all_data.sample()
            indicators = sample[['RSI', 'Bollinger', 'MACD', 'Stochastic', 'OBV', 'ATR', 'Williams %R', 'CCI']].values.flatten().tolist()
            expected_output = sample['PercentChange'].values[0]

            # Train the network
            best_network.backpropagate(indicators, learning_rate=0.05)

    # Prediction Phase
    correct_predictions = 0
    total_predictions = 50
    for _ in range(total_predictions):
        sample = all_data.sample()
        indicators = sample[['RSI', 'Bollinger', 'MACD', 'Stochastic', 'OBV', 'ATR', 'Williams %R', 'CCI']].values.flatten().tolist()
        expected_output = sample['PercentChange'].values[0]

        # Evaluate the network
        result = best_network.evaluate(indicators)
        print(f"Expected: {expected_output}, Predicted: {result}")

        # Simple accuracy measure: consider prediction correct if the sign of percent change is predicted correctly
        if (expected_output > 0 and result > 0) or (expected_output <= 0 and result <= 0):
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Output the multipliers (weights) of the first four nodes of the first hidden layer
    first_layer_weights = []
    num_nodes_to_print = best_network.structure[0]

    input_labels = ['RSI', 'Bollinger', 'MACD', 'Stochastic', 'OBV', 'ATR', 'Williams %R', 'CCI']

    for i in range(num_nodes_to_print):
        if i < len(best_network.network[1]):  # Ensure we don't exceed the number of nodes in the first hidden layer
            weights = best_network.network[1][i][1]  # Extracting weights of the first four nodes
            first_layer_weights.append(weights)
            # Calculate the norm (L2 norm) for each node
            norm = math.sqrt(sum(w ** 2 for w in weights))
            print(f"Node {i + 1} weights (connected to {input_labels[i]}): Norm: {norm}")
