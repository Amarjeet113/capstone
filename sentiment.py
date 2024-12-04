import numpy as np
import pandas as pd

# Example data for actual and predicted prices, direction, and volatility
actual_prices = np.array([4397.1, 4283.05, 4155.05, 4171.2, 4200.45])
predicted_prices = np.array([4285.07, 4297.83, 4304.06, 4298.87, 4285.19])


actual_direction = np.array([1, 0, 0, 1, 1])  # 1 means positive, 0 means negative
predicted_direction = np.array([0, 0, 0, 0, 0])

actual_volatility = np.array([-0.0162, 0.0278, 0.0275, 0.0090, -0.0107])
predicted_volatility = np.array([0.0385, 0.0386, 0.0374, 0.0342, 0.0303])

# Function to classify price movement sentiment
def classify_sentiment(actual_prices, predicted_prices, actual_direction, predicted_direction, actual_volatility, predicted_volatility):
    sentiment = {"price_trend": "", "direction": "", "volatility": "", "overall": ""}
    
    # 1. Classifying price trend (change between first and last price)
    price_change = actual_prices[-1] - actual_prices[0]
    if price_change > 0:
        sentiment['price_trend'] = "positive"
    elif price_change < 0:
        sentiment['price_trend'] = "negative"
    else:
        sentiment['price_trend'] = "neutral"

    # 2. Classifying direction
    # If majority of the actual directions are 1, it's positive
    if np.mean(actual_direction) > 0.5:
        sentiment['direction'] = "positive"
    elif np.mean(actual_direction) < 0.5:
        sentiment['direction'] = "negative"
    else:
        sentiment['direction'] = "neutral"

    # 3. Classifying volatility
    # If volatility is high and positive, we assume it as neutral or uncertain; negative volatility can indicate positive market (stabilizing)
    avg_actual_volatility = np.mean(actual_volatility)
    if avg_actual_volatility > 0:
        sentiment['volatility'] = "neutral"
    else:
        sentiment['volatility'] = "positive" if avg_actual_volatility < 0 else "neutral"
    
    # 4. Overall sentiment - based on majority rule
    if sentiment['price_trend'] == "positive" and sentiment['direction'] == "positive":
        sentiment['overall'] = "positive"
    elif sentiment['price_trend'] == "negative" or sentiment['direction'] == "negative":
        sentiment['overall'] = "negative"
    else:
        sentiment['overall'] = "neutral"

    return sentiment

# Classifying sentiment
sentiment = classify_sentiment(actual_prices, predicted_prices, actual_direction, predicted_direction, actual_volatility, predicted_volatility)

# Print sentiment results
print("Sentiment Analysis Results:")
print(f"Price Trend: {sentiment['price_trend']}")
print(f"Direction: {sentiment['direction']}")
print(f"Volatility: {sentiment['volatility']}")
print(f"Overall Sentiment: {sentiment['overall']}")
