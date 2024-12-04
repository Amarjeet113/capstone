import pandas as pd
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import ta

def fetch_latest_news(stock_symbol):
    news = yf.Ticker(stock_symbol).news
    return news



def fetch_technical_analysis(stock_symbol):
    try:
        # Fetch stock data
        stock = yf.Ticker(stock_symbol)

        # Get historical data for the last year
        historical_data = stock.history(period="1y")

        # Check if historical data is empty
        if historical_data.empty:
            raise ValueError(f"No historical data found for ticker '{stock_symbol}'.")

        # Calculate technical indicators
        historical_data['SMA_50'] = ta.trend.sma_indicator(historical_data['Close'], window=50)
        historical_data['SMA_200'] = ta.trend.sma_indicator(historical_data['Close'], window=200)
        historical_data['RSI'] = ta.momentum.rsi(historical_data['Close'], window=14)

        # Calculate latest values
        latest_data = historical_data.iloc[-1]

        # Identify support and resistance levels
        support_level = historical_data['Close'].min()
        resistance_level = historical_data['Close'].max()

        # Compile the technical analysis data
        technical_analysis = {
            'moving_average_50': latest_data['SMA_50'],
            'moving_average_200': latest_data['SMA_200'],
            'rsi': latest_data['RSI'],
            'support_level': support_level,
            'resistance_level': resistance_level
        }

        return technical_analysis

    except Exception as e:
        print(f"[ERROR] An error occurred while fetching technical analysis for '{stock_symbol}': {str(e)}")
        return {
            'moving_average_50': None,
            'moving_average_200': None,
            'rsi': None,
            'support_level': None,
            'resistance_level': None
        }


# def compute_rsi(series, period=14):
#     delta = series.diff(1)
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    # print("\n--- Stock Information ---")
    # for attribute in dir(stock):
    #     if not attribute.startswith('_'):  # Ignore private attributes
    #         try:
    #             print(f"{attribute}: {getattr(stock, attribute)}")
    #         except Exception as e:
    #             print(f"{attribute}: Error retrieving attribute - {e}")

    return {
       'name': stock.info.get('longName', 'N/A'),
        'industry': stock.info.get('industry', 'N/A'),
        'sector': stock.info.get('sector', 'N/A'),
        'current_price': stock.info.get('currentPrice', 'N/A'),
        'market_cap': stock.info.get('marketCap', 'N/A'),
        'eps': stock.info.get('trailingEps', 'N/A'),
        'pe_ratio': stock.info.get('trailingPE', 'N/A'),
        'dividend_yield': stock.info.get('dividendYield', 'N/A'),
        'revenue': stock.info.get('totalRevenue', 'N/A'),
        'net_income': stock.info.get('netIncome', 'N/A'),
    }

def analyze_decision(predicted_price, actual_price, sentiment):
    decision = ""
    potential_profit = 0
    potential_loss = 0

    if predicted_price[-1] > actual_price[-1]:
        decision = "Buy"
        potential_profit = predicted_price[-1] - actual_price[-1]
    elif predicted_price[-1] < actual_price[-1]:
        decision = "Sell"
        potential_loss = actual_price[-1] - predicted_price[-1]
    else:
        decision = "Hold"

    return decision, potential_profit, potential_loss

def generate_stock_chart(stock_symbol, actual_price, predicted_price):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_price, label='Actual Price', color='blue')
    plt.plot(predicted_price, label='Predicted Price', color='orange')
    plt.title(f'Stock Price Chart for {stock_symbol}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    chart_filename = f"{stock_symbol}_price_chart.png"
    plt.savefig(chart_filename)
    plt.close()
    return chart_filename

def generate_pdf_report(stock_symbol, actual_price, predicted_price, actual_direction, predicted_direction, 
                         actual_volatility, predicted_volatility, sentiment):
    # Fetch latest news and stock data
    latest_news = fetch_latest_news(stock_symbol)
    stock_data = fetch_stock_data(stock_symbol)
    technical_data = fetch_technical_analysis(stock_symbol)


    # Analyze decision based on predictions and sentiment
    decision, potential_profit, potential_loss = analyze_decision(predicted_price, actual_price, sentiment)

    # Generate stock price chart
    chart_filename = generate_stock_chart(stock_symbol, actual_price, predicted_price)

    # Create PDF document
    pdf_filename = f"{stock_symbol}_report.pdf"
    document = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(f"<u>Stock Prediction Report for {stock_data['name']} ({stock_symbol})</u>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Stock Overview
    elements.append(Paragraph("<b>Stock Overview:</b>", styles['Heading2']))
    stock_overview = [
        ['Company Name:', stock_data['name']],
        ['Ticker Symbol:', stock_symbol],
        ['Industry:', stock_data['industry']],
        ['Sector:', stock_data['sector']],
        ['Current Price:', f"${float(stock_data['current_price']):.2f}" if stock_data['current_price'] != 'N/A' else 'N/A'],
        ['Market Capitalization:', f"${float(stock_data['market_cap']):.2f}" if stock_data['market_cap'] != 'N/A' else 'N/A'],
        ['Earnings per Share (EPS):', f"${float(stock_data['eps']):.2f}" if stock_data['eps'] != 'N/A' else 'N/A'],
        ['P/E Ratio:', f"{float(stock_data['pe_ratio']):.2f}" if stock_data['pe_ratio'] != 'N/A' else 'N/A'],
        ['Dividend Yield:', f"{float(stock_data['dividend_yield']):.2%}" if stock_data['dividend_yield'] != 'N/A' else 'N/A'],
        ['Revenue:', f"${float(stock_data['revenue']):.2f}" if stock_data['revenue'] != 'N/A' else 'N/A'],
        ['Net Income:', f"${float(stock_data['net_income']):.2f}" if stock_data['net_income'] != 'N/A' else 'N/A']
    ]
    overview_table = Table(stock_overview)
    overview_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                         ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                         ('GRID', (0, 0), (-1, -1), 1, colors.black),]))
    elements.append(overview_table)
    elements.append(Spacer(1, 12))

    # Technical Analysis
    elements.append(Paragraph("<b>Technical Analysis:</b>", styles['Heading2']))
    technical_analysis = [
      ['Moving Average (50-day):', f"${technical_data['moving_average_50']:.2f}" if technical_data['moving_average_50'] is not None else 'N/A'],
      ['Moving Average (200-day):', f"${technical_data['moving_average_200']:.2f}" if technical_data['moving_average_200'] is not None else 'N/A'],
      ['Relative Strength Index (RSI):', f"{technical_data['rsi']:.2f}" if technical_data['rsi'] is not None else 'N/A'],
      ['Support Level:', f"${technical_data['support_level']:.2f}" if technical_data['support_level'] is not None else 'N/A'],
      ['Resistance Level:', f"${technical_data['resistance_level']:.2f}" if technical_data['resistance_level'] is not None else 'N/A']
  ]
  
   
    technical_table = Table(technical_analysis)
    technical_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                          ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                          ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                          ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                          ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                          ('GRID', (0, 0), (-1, -1), 1, colors.black),]))
    elements.append(technical_table)
    elements.append(Spacer(1, 12))

    # Decision Analysis
    elements.append(Paragraph("<b>Decision Analysis:</b>", styles['Heading2']))
    decision_analysis = [
        ['Decision:', decision],
        ['Potential Profit:', f"${potential_profit:.2f}"],
        ['Potential Loss:', f"${potential_loss:.2f}"]
    ]
    decision_table = Table(decision_analysis)
    decision_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                         ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                         ('GRID', (0, 0), (-1, -1), 1, colors.black),]))
    elements.append(decision_table)
    elements.append(Spacer(1, 12))

    # Sentiment Analysis
    elements.append(Paragraph("<b>Sentiment Analysis:</b>", styles['Heading2']))
    sentiment_results = {
        'Sentiment Metric': ['Price Trend', 'Direction', 'Volatility', 'Overall Sentiment'],
        'Sentiment Values': [sentiment['price_trend'], sentiment['direction'], 
                             sentiment['volatility'], sentiment['overall']]
    }
    
    sentiment_df = pd.DataFrame(sentiment_results)
    sentiment_table_data = [['Sentiment Metric', 'Sentiment Values']] + sentiment_df.values.tolist()
    sentiment_table = Table(sentiment_table_data)
    sentiment_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                          ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                          ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                          ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                          ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                          ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                          ('GRID', (0, 0), (-1, -1), 1, colors.black),]))
    elements.append(sentiment_table)
    elements.append(Spacer(1, 12))

    # Latest News Section
    elements.append(Paragraph("<b>Latest News:</b>", styles['Heading2']))
    for news in latest_news:
        news_paragraph = Paragraph(f"<u>{news['title']}</u>", styles['Normal'])
        elements.append(news_paragraph)
        elements.append(Spacer(1, 6))  # Add space between news articles

    # Add Stock Price Chart
    elements.append(Spacer(1, 12))
    elements.append(Image(chart_filename, width=400, height=200))
    elements.append(Spacer(1, 12))

    # Build the PDF
    document.build(elements)
    print(f"[INFO] PDF Report generated: {pdf_filename}")

