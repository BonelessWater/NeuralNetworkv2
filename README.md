# S&P 500 Stock Indicator Analysis and Neural Network Prediction

This project is designed to help analyze the importance of various stock market indicators and predict buy or sell signals using a neural network. It leverages historical stock data from the S&P 500 companies, calculates various technical indicators, and uses a neural network to identify which indicators are more crucial in predicting market movements.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Technical Indicators](#technical-indicators)
- [Neural Network](#neural-network)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Stock market analysis often involves examining various technical indicators to forecast future price movements. This project aggregates historical stock data for S&P 500 companies, calculates key technical indicators, and uses a neural network to predict whether the stock price will go up or down. By training and evaluating the neural network, the project aims to identify the most influential indicators for making trading decisions.

## Features

- Fetches historical stock data for S&P 500 companies.
- Calculates key technical indicators such as RSI, Bollinger Bands, MACD, Stochastic Oscillator, OBV, ATR, Williams %R, and CCI.
- Stores and retrieves data using SQLite database for efficient access.
- Trains a neural network to predict price movements based on the calculated indicators.
- Evaluates the importance of each indicator in predicting buy or sell signals.

## Setup

### Prerequisites

- Python 3.x
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `yfinance`, `sqlite3`
- A SQLite database named `sp500_stock_data.db` for storing and accessing historical stock data.

### Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/BonelessWater/NeuralNetworkv2.git
    ```

2. Navigate to the project directory:

    ```bash
    cd NeuralNetworkv2
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the SQLite database by running the script to fetch and store historical data (replace `main.py` with the actual script that initializes the database if it exists separately):

    ```bash
    python main.py
    ```

## Usage

1. **Run the main analysis script:**

    ```bash
    python main.py
    ```

   This script will fetch historical stock data, calculate technical indicators, train a neural network, and evaluate its performance.

2. **Check the output:**

   The script prints accuracy and the importance of various indicators based on neural network weights.

3. **Analyze the results:**

   The output will show how well the neural network predicts buy or sell signals and which indicators are more influential.

## Technical Indicators

The following indicators are calculated and used for training the neural network:

1. **Relative Strength Index (RSI)**
2. **Bollinger Bands**
3. **Moving Average Convergence Divergence (MACD)**
4. **Stochastic Oscillator**
5. **On-Balance Volume (OBV)**
6. **Average True Range (ATR)**
7. **Williams %R**
8. **Commodity Channel Index (CCI)**

These indicators are commonly used in technical analysis to gauge market momentum, volatility, and potential reversal points.

## Neural Network

The neural network used in this project is a custom implementation designed to:

- Take the above indicators as input features.
- Predict whether a stock's price will go up or down (buy/sell signal).
- Adjust its weights during training to emphasize the most important indicators.

### Network Structure

- **Input Layer:** 8 nodes (corresponding to the 8 technical indicators)
- **Hidden Layers:** Two hidden layers with 9 nodes each
- **Output Layer:** 1 node (prediction of price movement)

### Training

The network is trained using historical data, and its accuracy is evaluated using random samples. The goal is to achieve high accuracy in predicting the sign of the price change.

## Results

- The network's accuracy in predicting buy or sell signals is printed at the end of the script execution.
- The importance of each indicator is assessed based on the magnitude of the input weights.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests for any improvements or additional features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
