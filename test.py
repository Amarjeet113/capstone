# sentiment_analysis.py

import numpy as np

def classify_sentiment(actual_prices, actual_direction,  actual_volatility ):
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
    if np.mean(actual_direction) > 0.5:
        sentiment['direction'] = "positive"
    elif np.mean(actual_direction) < 0.5:
        sentiment['direction'] = "negative"
    else:
        sentiment['direction'] = "neutral"

    # 3. Classifying volatility
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
