import datetime
import pandas as pd

import trading.stock_trading_utils as trade_utils



import simulation_globals as sim

def run_trade_script():

    # Start running stock pick script at 1:00 AM
    if sim.SIMULATED_TIME.hour == 1 and sim.SIMULATED_TIME.minute == 0:
        trade_utils.stock_pick_script()
    

    # Buy and Sell stocks at 4:00 PM EST = 2:00 PM MST
    # Initiate 30 minutes before close
    if sim.SIMULATED_TIME.hour == 13 and sim.SIMULATED_TIME.minute == 30:
        for stock in list(sim.BUY_ORDERS.keys()):
            historical_price = pd.read_csv(f"data/historical_prices/{stock}.csv")
            trade_utils.tradeStockSimulated(tradeType='BUY', ticker=stock, 
                                    quantity=sim.BUY_ORDERS[stock]['quantity'], sell_date=sim.BUY_ORDERS[stock]['sell_date'], 
                                    historical_prices=historical_price)

            sim.PORTFOLIO.append({
                "ticker": stock,
                "quantity": sim.BUY_ORDERS[stock]['quantity'],
                "sell_date": sim.BUY_ORDERS[stock]['sell_date'],
            })
            del sim.BUY_ORDERS[stock]

        remove_pos = []
        for position in sim.PORTFOLIO:
            # Check if stock is ready to be sold
            if sim.SIMULATED_TIME.date() == position['sell_date'].date():
                historical_price = pd.read_csv(f"data/historical_prices/{position['ticker']}.csv")
                trade_utils.tradeStockSimulated(tradeType='SELL', ticker=position['ticker'], quantity=position['quantity'], 
                                       historical_prices=historical_price)
                remove_pos.append(position)
        for pos in remove_pos:
            sim.PORTFOLIO.remove(pos)


        ''' Strategy of combining buy orders and gradually stack them up
            # Add stock to portfolio once bought
            if stock in sim.PORTFOLIO.keys():
                sim.PORTFOLIO[stock]['quantity'] += sim.BUY_ORDERS[stock]['quantity']
                sim.PORTFOLIO[stock]['sell_date'] = sim.BUY_ORDERS[stock]['sell_date']
            else:
                sim.PORTFOLIO[stock] = {'quantity': sim.BUY_ORDERS[stock]['quantity'], 'sell_date': sim.BUY_ORDERS[stock]['sell_date']}

            # Delete the stock from buy orders once bought and added to portfolio
            del sim.BUY_ORDERS[stock]
            

        print(f"Portfolio after Buy: {sim.PORTFOLIO}")
        for stock in sim.PORTFOLIO.keys():
            # Check if stock is ready to be sold
            if simulated_time.date() == sim.PORTFOLIO[stock]['sell_date']:
                print(f"Selling {stock}")
                trade_utils.tradeStockSimulated(tradeType='SELL', ticker=stock, quantity=sim.PORTFOLIO[stock]['quantity'], 
                                       historical_prices=historical_price, simulated_time=simulated_time)
                del sim.PORTFOLIO[stock]
        print(f"Portfolio after sell: {sim.PORTFOLIO}")
        '''
            


if __name__ == '__main__':
    while True:
        print(f"Simulated Time: {sim.SIMULATED_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
        run_trade_script()
        sim.SIMULATED_TIME += datetime.timedelta(minutes=30)

        if sim.SIMULATED_TIME.date().month == 5 and sim.SIMULATED_TIME.date().day == 1:
            print(f"Liquid Balance: {trade_utils.getNetWorthSimulated()}")
            break
        # input("\n")


