from datetime import datetime

import trading.stock_trading_utils as trade_utils


SIMULATED_TIME = datetime.datetime(2023,1,1)
TRADING_CLIENT = trade_utils.get_trading_client()
BUY_ORDERS = {}
PORTFOLIO = {}

def run_trade_script():
    # Buy and Sell stocks at 4:00 PM EST = 2:00 PM MST
    # Initiate 30 minutes before close
    if SIMULATED_TIME.hour == 13 and SIMULATED_TIME.minute == 30:
        for stock in BUY_ORDERS.keys():
            trade_utils.tradeStock(tradeType='BUY', ticker=stock, 
                                    quantity=BUY_ORDERS[stock]['quantity'], sell_date=BUY_ORDERS[stock]['sell_date'])
            
            # Add stock to portfolio once bought
            if stock in PORTFOLIO.keys():
                PORTFOLIO[stock]['quantity'] += BUY_ORDERS[stock]['quantity']
                PORTFOLIO[stock]['sell_date'] = BUY_ORDERS[stock]['sell_date']
            else:
                PORTFOLIO[stock] = {'quantity': BUY_ORDERS[stock]['quantity'], 'sell_date': BUY_ORDERS[stock]['sell_date']}
        
        for stock in PORTFOLIO.keys():
            # Check if stock is ready to be sold
            if SIMULATED_TIME.date() == PORTFOLIO[stock]['sell_date']:
                trade_utils.tradeStock(tradeType='SELL', ticker=stock, quantity=PORTFOLIO[stock]['quantity'])
                del PORTFOLIO[stock]


if __name__ == '__main__':
    run_trade_script()



