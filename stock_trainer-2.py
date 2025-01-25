import os
import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import openai
import joblib


# Function to fetch stock data
def fetch_stock_data():
    stock_symbol = stock_entry.get().strip()
    time_period = "1y"  
    
    if not stock_symbol:
        messagebox.showerror("Error", "Please enter a stock ticker!")
        return

    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period=time_period)

        if data.empty:
            messagebox.showinfo("No Data", f"No data found for ticker: {stock_symbol}")
        else:
            # Add technical indicators
            data['VMA_20'] = data['Volume'].rolling(window=20).mean()
            data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['RSI'] = calculate_rsi(data['Close'])

            # Generate signals for simplicity
            data['Signal'] = generate_signals(data)

            data = data.dropna(subset=['Close', 'MACD', 'RSI', 'VMA_20', 'Signal'])

            X = data[['Close', 'MACD', 'RSI', 'VMA_20']].values
            y = pd.Categorical(data['Signal']).codes  # Convert to numerical labels

            # Check lengths
            print(f"Length of X: {len(X)}, Length of y: {len(y)}")

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Plot data
            plot_data(data)

            # Train the model
            model = train_model(data)
            
            os.makedirs('models', exist_ok=True)
            
            joblib.dump(model, 'models/trained_model.pkl')

            messagebox.showinfo("Success", f"Data fetched and model trained for {stock_symbol}!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to calculate RSI
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to generate basic trading signals
def generate_signals(data):
    signals = []
    for i in range(len(data)):
        if data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and data['RSI'].iloc[i] < 30:
            signals.append('Buy')
        elif data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and data['RSI'].iloc[i] > 70:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

# Function to plot data
def plot_data(data):
    fig = go.Figure()

    # Plot Close price
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

    # Plot VMA
    fig.add_trace(go.Scatter(x=data.index, y=data['VMA_20'], mode='lines', name='VMA_20', line=dict(color='green')))

    # Plot MACD and MACD Signal
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))

    # Add hover functionality
    fig.update_traces(marker=dict(size=5), hovertemplate='%{x}<br>Value: %{y}<extra></extra>')

    # Layout
    fig.update_layout(
        title="Stock Price and Indicators",
        xaxis_title="Date",
        yaxis_title="Price and Indicators",
        hovermode="closest",
        template="plotly_dark",
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            autorange=True,
            fixedrange=False,
            title="Price",
            rangemode="normal",
        ),
        dragmode="zoom",
        autosize=True,
    )

    fig.show()

# Function to train AI model
def train_model(data):
    # Prepare data for training
    X = data[['Close', 'MACD', 'RSI', 'VMA_20']].dropna().values
    y = pd.Categorical(data['Signal'].dropna()).codes  # Convert to numerical labels

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # OpenAI Insights
    insights = generate_openai_insights(accuracy, classification_report(y_test, y_pred))
    print("AI Insights:\n", insights)

    return model  # Return the trained model

# Function to generate insights using OpenAI
def generate_openai_insights(accuracy, report):
    try:
        prompt = (
            f"The AI model has been trained on stock data. Here are the results:\n"
            f"Accuracy: {accuracy}\n"
            f"Classification Report:\n{report}\n"
            f"Based on these results, provide insights about how the model performed and suggest improvements."
        )

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating insights: {e}"

# Set up the GUI
root = tk.Tk()
root.title("Stock Data Fetcher and AI Trainer")

# Input label and entry
label = tk.Label(root, text="Enter Stock Ticker:")
label.pack(pady=5)

stock_entry = tk.Entry(root, width=20)
stock_entry.pack(pady=5)

# Fetch button
fetch_button = tk.Button(root, text="Fetch Data & Train AI", command=fetch_stock_data)
fetch_button.pack(pady=10)

# Run the GUI loop
root.mainloop()
