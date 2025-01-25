import joblib
import yfinance as yf
import pandas as pd

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load the trained model with error handling
try:
    model = joblib.load('models/trained_model.pkl')
    if model is None:
        raise ValueError("Loaded model is None. Please check the model file.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)  
    
# Define the stock tickers
tickers = [' ']

# Fetch the latest stock data
data = yf.download(tickers, period='1d', interval='1m')  # Adjust as needed

# Print the column names to inspect them
print("Column names after downloading data:")
print(data.columns)

# Preprocess the data
for ticker in tickers:
    # Calculate VMA for each ticker
    data[('VMA_20', ticker)] = data[('Volume', ticker)].rolling(window=20).mean()
    data[('MACD', ticker)] = data[('Close', ticker)].ewm(span=12, adjust=False).mean() - data[('Close', ticker)].ewm(span=26, adjust=False).mean()
    data[('MACD_Signal', ticker)] = data[('MACD', ticker)].ewm(span=9, adjust=False).mean()
    data[('RSI', ticker)] = calculate_rsi(data[('Close', ticker)])  # Calculate RSI

# Drop rows with NaN values for each ticker
for ticker in tickers:
    data = data.dropna(subset=[('Close', ticker), ('MACD', ticker), ('RSI', ticker), ('VMA_20', ticker)])

# Prepare features for prediction
predictions = {}
for ticker in tickers:
    # Select the last row of features for prediction
    last_row = data[[('Close', ticker), ('MACD', ticker), ('RSI', ticker), ('VMA_20', ticker)]].iloc[-1].values
    last_row = last_row.reshape(1, -1)  # Reshape to (1, 4) for prediction
    predictions[ticker] = model.predict(last_row)  # Make prediction

# Assuming your model predicts classes (e.g., 0 for 'Hold', 1 for 'Buy', 2 for 'Sell')
signal_mapping = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
predicted_signals = {ticker: signal_mapping[pred[0]] for ticker, pred in predictions.items()}

# Output the predictions
for ticker, signal in predicted_signals.items():
    print(f"{ticker}: {signal}")
