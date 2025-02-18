import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Download stock data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Prepare dataset for LSTM
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close', 'Volume']])
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict and plot
def predict_stock(model, data, scaler):
    predictions = model.predict(data)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:,0]
    return predictions

# Main function
def main():
    ticker = 'AAPL'  # Apple stock
    start, end = '2015-01-01', '2023-01-01'
    df = get_stock_data(ticker, start, end)
    
    # Train-test split
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size - 60:]
    
    # Prepare data
    X_train, y_train, scaler = prepare_data(train_data)
    X_test, y_test, _ = prepare_data(test_data)
    
    # Train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    
    # Make predictions
    predictions = predict_stock(model, X_test, scaler)
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(df.index[train_size:], scaler.inverse_transform(np.expand_dims(y_test, axis=1)), label='Actual Prices')
    plt.plot(df.index[train_size:], predictions, label='Predicted Prices', linestyle='dashed')
    plt.legend()
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.show()

if __name__ == "__main__":
    main()
