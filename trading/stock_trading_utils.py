from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta

from config import SELL_DATES



def tradeStock(tradeType, ticker, quantity, tradingClient, hold_date):
    if tradeType == "BUY":
        tradingSide = OrderSide.BUY
    elif tradeType == "SELL":
        tradingSide = OrderSide.SELL
    else:
        print("Invalid Trade Type: Must be a buy or sell order")


    # Initialize order data
    marketOrderData = MarketOrderRequest(
        symbol=ticker,
        qty=quantity,
        side=tradingSide,
        time_in_force=TimeInForce.DAY)
    
    # Submit order to trading client
    order = tradingClient.submit_order(order_data = marketOrderData)
    logTrade(tradeType, ticker, quantity, hold_date)

def buyStock(ticker, quantity, trading_client, hold_date):
    tradeStock("BUY", ticker, quantity, trading_client, hold_date)

def sellStock(ticker, quantity, trading_client, hold_date):
    tradeStock("SELL", ticker, quantity, trading_client, hold_date)

def logTrade(tradeType, ticker, quantity, hold_date):
    today_date = datetime.now()
    logEntry = (f"[{today_date.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{tradeType} - Ticker: {ticker} - Quantity: {quantity} - "
                f"Selling On: {(today_date + timedelta(days=hold_date)).strftime('%Y-%m-%d')} \n")
    
    # Take note of what day the stock will be sold on
    SELL_DATES[ticker] = [(today_date + timedelta(days=hold_date)).date(), quantity]

    
    with open("trade_log.txt", 'a') as file:
        file.write(logEntry)

