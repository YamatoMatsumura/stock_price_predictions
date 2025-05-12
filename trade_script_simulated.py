import datetime
import pandas as pd
import matplotlib.pyplot as plt

import trading.stock_trading_utils as trade_utils

import simulation_globals as sim
import config

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
    balance_history = {'date': [sim.SIMULATED_TIME.date()], 'balance': [sim.BALANCE]}
    balance_history = pd.DataFrame(balance_history)
    raw_balance = {'date': [sim.SIMULATED_TIME.date()], 'balance': [sim.BALANCE]}
    raw_balance = pd.DataFrame(raw_balance)
    last_update = sim.SIMULATED_TIME.date()
    while True:
        print(f"Simulated Time: {sim.SIMULATED_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
        run_trade_script()
        sim.SIMULATED_TIME += datetime.timedelta(minutes=30)

        # Update balance history every day
        if sim.SIMULATED_TIME.date() != last_update and \
            sim.SIMULATED_TIME.strftime('%Y-%m-%d') in pd.read_csv(f'data/historical_prices/{config.STOCK_NAMES[0]}.csv')['date'].values:
            last_update = sim.SIMULATED_TIME.date()
            balance = trade_utils.getNetWorthSimulated()
            new_row = pd.DataFrame({'date': [sim.SIMULATED_TIME.date()], 'balance': [balance]})
            balance_history = pd.concat([pd.DataFrame(balance_history), new_row], ignore_index=True)

            new_row = pd.DataFrame({'date': [sim.SIMULATED_TIME.date()], 'balance': [sim.BALANCE]})
            raw_balance = pd.concat([pd.DataFrame(raw_balance), new_row], ignore_index=True)


            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(balance_history['date'], balance_history['balance'], linestyle='-', color='blue', label='Liquid Balance')
 
            # Labels and title
            plt.title('Liquid Balance Over Time')
            plt.xlabel('Day')
            plt.ylabel('Balance ($)')
            plt.grid(True)
            plt.legend()
            plt.ticklabel_format(style='plain', axis='y')

            plt.savefig('data/sim_dump/liquid_balance_over_time.png')
            plt.close()
            balance_history.to_csv('data/sim_dump/liquid_balance_over_time.csv', index=False)


            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(raw_balance['date'], raw_balance['balance'], linestyle='-', color='blue', label='Raw Balance')

            # Labels and title
            plt.title('Raw Balance Over Time')
            plt.xlabel('Day')
            plt.ylabel('Balance ($)')
            plt.grid(True)
            plt.legend()
            plt.ticklabel_format(style='plain', axis='y')

            plt.savefig('data/sim_dump/raw_balance_over_time.png')
            plt.close()
            balance_history.to_csv('data/sim_dump/raw_balance_over_time.csv', index=False)

        # End sim early for testing
        if sim.SIMULATED_TIME.date().year == 2025 and sim.SIMULATED_TIME.date().month == 4:
            print(f"Liquid Balance: {trade_utils.getNetWorthSimulated()}")
            break
        # input("\n")
