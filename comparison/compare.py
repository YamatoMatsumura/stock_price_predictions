import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters
ticker = 'QQQ'                      # S&P 500 ETF
start_date = '2023-01-01'
end_date = '2023-02-01'
initial_investment = 3000

# Download daily data
data = yf.download(ticker, start=start_date, end=end_date)

# Use adjusted close to include dividends
prices = data['Close']

# Calculate daily returns
returns = prices.pct_change().fillna(0)

# Simulate portfolio value
portfolio_values = (1 + returns).cumprod() * initial_investment

# Plot portfolio value over time
portfolio_values.plot(title=f'{ticker} Investment (${initial_investment:,}) from {start_date} to {end_date}')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{ticker}.png')
# Export to CSV if needed
portfolio_values.to_csv(f'{ticker}.csv')
