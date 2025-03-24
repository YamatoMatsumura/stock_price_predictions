from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime

# Get API keys
with open('api_keys/alpaca/key.txt') as file:
    KEY = file.read()
with open('api_keys/alpaca/secret.txt') as file:
    SECRET = file.read()

# Initialize trading client
tradingClient = TradingClient(KEY, SECRET, paper=True)

def tradeStock(tradeType, ticker, quantity):
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
    logTrade(tradeType, ticker, quantity)

def buyStock(ticker, quantity):
    tradeStock("BUY", ticker, quantity)

def sellStock(ticker, quantity):
    tradeStock("SELL", ticker, quantity)

def logTrade(tradeType, ticker, quantity):
    logEntry = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {tradeType} - Ticker: {ticker} - Quantity: {quantity}")
    
    with open("trade_log.txt", 'a') as file:
        file.write(logEntry)

def main():
    buyStock("AAPL", 1)

if __name__ == "__main__":
    main()