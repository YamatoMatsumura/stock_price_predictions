from alpaca.trading.client import TradingClient
import datetime

with open('api_keys/alpaca/key.txt') as file:
    KEY = file.read()
with open('api_keys/alpaca/secret.txt') as file:
    SECRET = file.read()
TRADING_CLIENT = TradingClient(KEY, SECRET, paper=True)

SIMULATED_TIME = datetime.datetime(2023,1,3) # Simulated time for testing
BUY_ORDERS = {} # Dictionary to hold buy orders
PORTFOLIO = [] # List to hold stocks in portfolio
BALANCE = 1000000 # Starting balance for simulation
