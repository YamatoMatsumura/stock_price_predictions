from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


with open('api_keys/alpaca/key.txt') as file:
    KEY = file.read()
with open('api_keys/alpaca/secret.txt') as file:
    SECRET = file.read()

tradingClient = TradingClient(KEY, SECRET, paper=True)

marketOrderData = MarketOrderRequest(symbol="AAPL", qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
marketOrder = tradingClient.submit_order(order_data = marketOrderData)

# Get a list of all of our positions.
portfolio = tradingClient.get_all_positions()

# Print the quantity of shares for each position.
for position in portfolio:
    print("{} shares of {}".format(position.qty, position.symbol))