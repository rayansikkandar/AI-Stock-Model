# AI-Stock-Model


This repository contains a Python-based application for predicting stock market signals and training AI models using historical stock data. It utilizes machine learning techniques to provide trading signals such as Buy, Sell, and Hold, based on technical indicators. You can utilize the pre-trained model "trained_model.pkl" or train your own model using "stock_trainer.py". Our pre-trained model has been trained using AMZN, META, MSFT, TSLA, NVDA, TSM, AMD, and BABA. 

**1. Training Module (stock_trainer.py)**

Fetches historical stock data from Yahoo Finance.
Adds technical indicators to the data.
Trains an AI model using XGBoost for classification of trading signals.
Provides insights into model performance with an interactive GUI built using Tkinter.
Exports the trained model to trained_model.pkl for use in the prediction module.

**2. Prediction Module (predict_stocks.py)**

Predicts stock market signals using the pre-trained AI model - trained_model.pkl
Calculates technical indicators like:
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
VMA (Volume Moving Average)
Designed to predict real-time signals for predefined stock tickers.

# Prerequisites

Required Libraries:
Install the following dependencies using pip

```bash
pip install pandas yfinance plotly scikit-learn xgboost joblib openai tkinter
```

# Usage

**1. Training a New Model**

Run the stock_trainer.py script to train a new AI model:
```bash
python stock_trainer.py
```
Use the GUI to input a stock ticker and start the training process. The script will fetch data, train the model, and save it to "models/trained_model.pkl"

**2. Predicting Stock Signals**

First edit the predict_stocks.py and add the ticker that you want to generate a trading signal for into the ticker = ' ' area. Then run predict_stocks.py:
```bash
python predict_stocks.py
```
Ensure that trained_model.pkl is present in the models directory.

# Technical Indicators
1. RSI: Detects overbought or oversold conditions.

2. MACD: Measures momentum and trend direction.

3. VMA: Tracks average volume over a defined period.

# File Descriptions
predict_stocks.py: Script for real-time signal prediction.

stock_trainer.py: Script for fetching data, training the AI model, and providing insights.

trained_model.pkl: Pre-trained model used for predictions (binary file).


