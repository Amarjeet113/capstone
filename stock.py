import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from test import classify_sentiment
from report import generate_pdf_report

# Function to fetch and process stock data
def fetch_nse_stock_data(stock_symbol):
    stock_symbol = stock_symbol.upper() + ".NS"
    print(f"[INFO] Fetching data for {stock_symbol}...")
    stock = yf.Ticker(stock_symbol)
    hist_data = stock.history(period="2y")
    
    if hist_data.empty:
        print(f"[ERROR] No data found for {stock_symbol}. Please check the stock symbol.")
        return None
    
    print(f"[INFO] Calculating technical indicators (RSI, EMA, MACD)...")
    hist_data['RSI'] = ta.rsi(hist_data['Close'], length=14)
    hist_data['EMA_20'] = ta.ema(hist_data['Close'], length=20)
    macd_values = ta.macd(hist_data['Close'])
    hist_data['MACD'] = macd_values['MACD_12_26_9']
    hist_data['MACD_Signal'] = macd_values['MACDs_12_26_9']
    hist_data['MACD_Hist'] = macd_values['MACDh_12_26_9']

    # Fill NaN values
    print(f"[INFO] Handling NaN values...")
    hist_data.fillna(method='ffill', inplace=True)
    hist_data.fillna(method='bfill', inplace=True)
    
    # Select necessary columns
    data = hist_data[['Open', 'High', 'Low', 'Close', 'RSI', 'EMA_20', 'MACD', 'MACD_Signal', 'MACD_Hist']]
    print(f"[INFO] Data fetching and processing complete.")
    
    # Save data for future use
    file_name = f"{stock_symbol}_processed_data.csv"
    data.to_csv(file_name)
    print(f"[INFO] Data saved to {file_name}.")
    
    return data

# Function to create sequences for LSTM
def create_sequences(data, look_back=60):
    X, y_price, y_direction, y_volatility = [], [], [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y_price.append(data[i + look_back, 3])  # Close price
        y_direction.append(1 if data[i + look_back, 3] > data[i + look_back - 1, 3] else 0)
        y_volatility.append(data[i + look_back, 1] - data[i + look_back, 2])  # High - Low
    
    print(f"[INFO] Created {len(X)} sequences from the data.")
    return np.array(X), np.array(y_price), np.array(y_direction), np.array(y_volatility)

# Function to build and train the LSTM model
def build_and_train_lstm(X_train, y_train):
    print("[INFO] Building LSTM model...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=3))  # For price, direction, and volatility

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("[INFO] Training the LSTM model...")
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    print("[INFO] Model training complete.")
    return model







def main():
    stock_symbol = input("Enter NSE stock symbol (e.g., RELIANCE): ").upper()
    data = fetch_nse_stock_data(stock_symbol)
    if data is None:
        return

    print("[INFO] Scaling the data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Check if scaled data is large enough to split
    if len(scaled_data) < 120:  # Need enough data for look_back and splitting
        print("[ERROR] Not enough data to proceed with the split and training.")
        return

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    print(f"[INFO] Data split into {len(train_data)} training samples and {len(test_data)} test samples.")

    # Ensure there is enough data for creating sequences
    look_back = 60
    if len(train_data) < look_back or len(test_data) < look_back:
        print("[ERROR] Not enough data for creating sequences after the split.")
        return

    X_train, y_train_price, y_train_direction, y_train_volatility = create_sequences(train_data, look_back)
    X_test, y_test_price, y_test_direction, y_test_volatility = create_sequences(test_data, look_back)

    # Stack the target variables for training
    y_train = np.column_stack([y_train_price, y_train_direction, y_train_volatility])

    # Proceed if there are enough samples to train
    if X_train.size == 0 or X_test.size == 0:
        print("[ERROR] Not enough sequences available to train or predict.")
        return

    model = build_and_train_lstm(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)

    # Extract predicted values
    predicted_price = predictions[:, 0]
    predicted_direction = (predictions[:, 1] > 0.5).astype(int)
    predicted_volatility = predictions[:, 2]

    # Reshape data for inverse scaling
    y_test_price = y_test_price.reshape(-1, 1)
    predicted_price = predicted_price.reshape(-1, 1)

    # Inverse transform to get actual prices
    # Prepare dummy arrays to match the scaler's expected input shape
    dummy_array = np.zeros((len(y_test_price), scaled_data.shape[1]))
    dummy_array[:, 3] = y_test_price[:, 0]  # Set the Close price column
    y_test_price_rescaled = scaler.inverse_transform(dummy_array)[:, 3]

    dummy_array[:, 3] = predicted_price[:, 0]
    predicted_price_rescaled = scaler.inverse_transform(dummy_array)[:, 3]

    print(f"\nLogging Predictions for {stock_symbol}:")
    print(f"Actual Price: {y_test_price_rescaled[:5]}")
    print(f"Predicted Price: {predicted_price_rescaled[:5]}")
    print(f"Actual Direction: {y_test_direction[:5]}")
    print(f"Predicted Direction: {predicted_direction[:5]}")
    print(f"Actual Volatility: {y_test_volatility[:5]}")
    print(f"Predicted Volatility: {predicted_volatility[:5]}")

    sentiment = classify_sentiment(
        y_test_price_rescaled, 
        y_test_direction, 
        y_test_volatility, 
    )

    # Display the sentiment results
    print("\nSentiment Analysis Results:")
    print("sentiment", sentiment)
    print(f"Price Trend: {sentiment['price_trend']}")
    print(f"Direction: {sentiment['direction']}")
    print(f"Volatility: {sentiment['volatility']}")
    print(f"Overall Sentiment: {sentiment['overall']}")

    generate_pdf_report(stock_symbol, y_test_price_rescaled, predicted_price_rescaled, 
                    y_test_direction, predicted_direction, 
                    y_test_volatility, predicted_volatility, sentiment)




    # Plotting the results
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.plot(y_test_price_rescaled, label="Actual Price")
    plt.plot(predicted_price_rescaled, label="Predicted Price")
    plt.title(f"{stock_symbol} - Actual vs Predicted Stock Price")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(y_test_direction, label="Actual Direction (Up=1/Down=0)")
    plt.plot(predicted_direction, label="Predicted Direction (Up=1/Down=0)")
    plt.title(f"{stock_symbol} - Actual vs Predicted Price Direction")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(y_test_volatility, label="Actual Volatility")
    plt.plot(predicted_volatility, label="Predicted Volatility")
    plt.title(f"{stock_symbol} - Actual vs Predicted Volatility")
    plt.legend()

    plt.tight_layout()
    plt.show()



    # After the predictions are done, we apply sentiment analysis
    # Assuming you have already made predictions
    


if __name__ == "__main__":
    main()
