import sqlite3
from datetime import datetime
import pandas as pd
import yfinance as yf 

# Create or connect to the SQLite3 database
db_name = 'sp500_stock_data.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Define the table schema
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, date)
    )
''')

# Fetch the list of S&P 500 companies
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Define the period for which we need data
start_date = '2020-01-01'
end_date = '2022-12-31'

# Loop through each S&P 500 stock ticker to fetch and store data
for ticker in sp500_tickers:
    try:
        print(f"Processing {ticker}...")
        
        # Check if the data for this ticker and date range is already in the database
        cursor.execute("SELECT date FROM stock_data WHERE ticker = ? AND date >= ? AND date <= ?", (ticker, start_date, end_date))
        existing_dates = set(row[0] for row in cursor.fetchall())
        
        # Fetch the historical data for the current ticker from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        # Insert new data into the database
        for index, row in data.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            if date_str not in existing_dates:
                cursor.execute('''
                    INSERT INTO stock_data (ticker, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (ticker, date_str, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']))

        conn.commit()  # Commit changes to the database

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Close the database connection
conn.close()
print("Data logging completed.")