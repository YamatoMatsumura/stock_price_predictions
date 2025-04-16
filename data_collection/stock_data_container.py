import requests
import pandas as pd
from datetime import datetime
import os
import numpy as np


class StockDataContainer:
    def __init__(self, ticker):
        self.ticker = ticker

        # API keys
        self.api_key_AV = None
        self.api_key_polygon = None
        self.api_key_finnhub = None
        self.AV_key_count = None
        self.tiingo_keys = []
        self.tiingo_key_index = 0  # Tracks which api key to use

        self.raw_data = pd.DataFrame()
        self.training_data = None
        self.last_updated = self._get_last_updated()
        self.script_first_time_called = True

        # Create folder to store data
        self._create_container_folder()

        # Grab all API Keys
        self._get_api_key()

        # Count number of AV keys to cycle through
        self._count_AV_keys()
    
    def _create_container_folder(self):
        if not os.path.exists('data/' + self.ticker):
            os.makedirs('data/' + self.ticker)

    def _get_api_key(self):
        with open('api_keys/Polygon.txt') as file:
            self.api_key_polygon = file.read()
        with open('api_Keys/AV/key0.txt') as file:
            self.api_key_AV = file.read()
        with open('api_keys/Finnhub.txt') as file:
            self.api_key_finnhub = file.read()
        with open('api_keys/Tiingo.txt') as file:
            keys = []
            for line in file:
                keys.append(line.strip())
            self.tiingo_keys = keys
    
    def _count_AV_keys(self):
        self.AV_key_count = 0
        contents = os.listdir('api_keys/AV')
        for item in contents:
            itemPath = os.path.join('api_keys/AV', item)
            if os.path.isfile(itemPath):
                self.AV_key_count += 1

    def _get_last_updated(self):
        try:
            df = pd.read_csv('data/' + self.ticker + '/trained_data.csv')
            return df['date'].iloc[0]
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return None
        
    def get_existing_data(self):
        self.training_data = pd.read_csv('data/' + self.ticker + '/trained_data.csv')


    def update_all_data(self, end_date=datetime.now()):   
        self.update_OHLC_data(end_date=end_date)
        self.update_sma()
        self.update_ema()
        self.update_rsi()
        # self.update_adx()
        # self.update_cci()
        # self.update_chaikin_ad()
        # self.update_obv()
        # self.update_return()
        self.update_distance_ema()
        self.update_macd()
        # self.update_bbands()
        # self.update_slowd()
        # self.update_atr()
        # self.update_lagged_features()
        self.update_volatility()
        self.update_ROC()
        self.update_date_data()

        # Drop highly correlated cols
        # self.raw_data.drop(columns=['sma_50', 'sma_200', 'ema_50', 
        #                             'ema_200', 'bb_middle', 'bb_upper', 'bb_lower'], inplace=True)

    def update_OHLC_data(self, end_date):
        while True:
            try:
                url = f"https://api.tiingo.com/tiingo/daily/{self.ticker}/prices"
                params = {
                    'startDate': '1900-01-01',
                    'endDate': end_date,
                    'format': 'json',
                    'sort': '-date'
                }
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Token {self.tiingo_keys[self.tiingo_key_index]}'
                }
                response = requests.get(url, params=params, headers=headers)
                data = response.json()

                # Check for max api calls
                if 'detail' in data and 'Error' in data['detail']:
                    self._cycle_tiingo_key()
                    continue

                break
            except requests.exceptions.RequestException as error:
                self._cycle_tiingo_key()
            
        
        # Check if no more data to fetch
        if not data:
            self.raw_data = pd.DataFrame()
            return
        df = pd.DataFrame(data)
        
        df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor'], inplace=True)
        df.rename(columns={'adjOpen': 'open', 'adjHigh': 'high', 'adjLow': 'low', 
                                'adjClose': 'close', 'adjVolume': 'volume', 'date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df.reset_index(drop=True, inplace=True)

        # Reverse to old data points are first for technical indicator calculations
        df = df.iloc[::-1].reset_index(drop=True)

        # Check if no data is currently stored
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = df
        else:
            # Fill in OHLC data
            self.raw_data = pd.concat([self.raw_data, df], axis=0, join='outer')
    
    def _cycle_tiingo_key(self):
        self.tiingo_key_index = (self.tiingo_key_index + 1) % len(self.tiingo_keys)

    def update_sma(self):
        self.raw_data['sma_50'] = self.raw_data['close'].rolling(window=50).mean()
        self.raw_data['sma_200'] = self.raw_data['close'].rolling(window=200).mean()

    def update_ema(self):
        self.raw_data['ema_50'] = self.raw_data['close'].ewm(span=50, adjust=False).mean()
        self.raw_data['ema_200'] = self.raw_data['close'].ewm(span=200, adjust=False).mean()

    def update_rsi(self):
        # Calculate daily price changes
        self.raw_data['price_change'] = self.raw_data['close'].diff()

        # Separate gains (positive changes) and losses (negative changes)
        self.raw_data['gain'] = self.raw_data['price_change'].where(self.raw_data['price_change'] > 0, 0)
        self.raw_data['loss'] = -self.raw_data['price_change'].where(self.raw_data['price_change'] < 0, 0)

        # Calculate the rolling average of gains and losses over the 14-day window
        self.raw_data['avg_gain'] = self.raw_data['gain'].rolling(window=14, min_periods=1).mean()
        self.raw_data['avg_loss'] = self.raw_data['loss'].rolling(window=14, min_periods=1).mean()

        # Calculate the relative strength (RS)
        self.raw_data['rs'] = self.raw_data['avg_gain'] / self.raw_data['avg_loss']

        # Calculate the RSI
        self.raw_data['rsi'] = 100 - (100 / (1 + self.raw_data['rs']))

        # Add binary encoding to indicate when rsi value is significant
        self.raw_data['rsi_overbought'] = (self.raw_data['rsi'] > 70).astype(int)
        self.raw_data['rsi_oversold'] = (self.raw_data['rsi'] < 30).astype(int)

        # Drop the intermediate columns
        self.raw_data.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1, inplace=True)
    
    def update_adx(self):
        # Calculate True Range
        self.raw_data['high_low'] = self.raw_data['high'] - self.raw_data['low']
        self.raw_data['high_close'] = abs(self.raw_data['high'] - self.raw_data['close'].shift())
        self.raw_data['low_close'] = abs(self.raw_data['low'] - self.raw_data['close'].shift())

        self.raw_data['tr'] = self.raw_data[['high_low', 'high_close', 'low_close']].max(axis=1)

        # Calculate +DM and -DM
        self.raw_data['plus_dm'] = self.raw_data['high'] - self.raw_data['high'].shift()
        self.raw_data['minus_dm'] = self.raw_data['low'].shift() - self.raw_data['low']

        self.raw_data['plus_dm'] = self.raw_data['plus_dm'].where(self.raw_data['plus_dm'] > 0, 0)
        self.raw_data['minus_dm'] = self.raw_data['minus_dm'].where(self.raw_data['minus_dm'] > 0, 0)

        # Calculate smoothed +DI and -DI over 14 periods
        self.raw_data['plus_di'] = (self.raw_data['plus_dm'].rolling(window=14).sum() / self.raw_data['tr'].rolling(window=14).sum()) * 100
        self.raw_data['minus_di'] = (self.raw_data['minus_dm'].rolling(window=14).sum() / self.raw_data['tr'].rolling(window=14).sum()) * 100

        # Calculate ADX (smoothed moving average of the difference between +DI and -DI)
        self.raw_data['adx'] = abs(self.raw_data['plus_di'] - self.raw_data['minus_di']).rolling(window=14).mean()

        # Drop intermediate columns
        self.raw_data.drop(['high_low', 'high_close', 'low_close', 'plus_dm', 'minus_dm'], axis=1, inplace=True)

    def update_cci(self):
        # Calculate Typical Price
        self.raw_data['typical_price'] = (self.raw_data['high'] + self.raw_data['low'] + self.raw_data['close']) / 3

        # Calculate the 20-day SMA of Typical Price
        self.raw_data['sma_tp'] = self.raw_data['typical_price'].rolling(window=20).mean()

        # Calculate Mean Deviation
        self.raw_data['mean_deviation'] = (self.raw_data['typical_price'] - self.raw_data['sma_tp']).abs().rolling(window=20).mean()

        # Calculate CCI
        self.raw_data['cci'] = (self.raw_data['typical_price'] - self.raw_data['sma_tp']) / (0.015 * self.raw_data['mean_deviation'])

        # Drop intermediate columns
        self.raw_data.drop(['typical_price', 'sma_tp', 'mean_deviation'], axis=1, inplace=True)
    
    def update_chaikin_ad(self):
        # Calculate the Money Flow Multiplier
        self.raw_data['money_flow_multiplier'] = ((self.raw_data['close'] - self.raw_data['low']) - (self.raw_data['high'] - self.raw_data['close'])) / (self.raw_data['high'] - self.raw_data['low'])

        # Calculate the Money Flow Volume
        self.raw_data['money_flow_volume'] = self.raw_data['money_flow_multiplier'] * self.raw_data['volume']

        # Calculate the Accumulation/Distribution line (running total of money flow volume)
        self.raw_data['chaikin_ad'] = self.raw_data['money_flow_volume'].cumsum()

        # Drop intermediate columns
        self.raw_data.drop(['money_flow_multiplier', 'money_flow_volume'], axis=1, inplace=True)

    def update_obv(self):
        # Calculate the OBV (On-Balance Volume)
        self.raw_data['obv'] = (self.raw_data['volume'] * (self.raw_data['close'] > self.raw_data['close'].shift())).cumsum()

        # Drop intermediate columns
        self.raw_data.drop(['volume'], axis=1, inplace=True)
    
    def update_return(self):
        # Percentage change of closing price over 10 days
        self.raw_data['return_10d'] = self.raw_data['close'].pct_change(periods=10)
    
    def update_distance_ema(self):
        self.raw_data['dist_from_ema_200'] = self.raw_data['close'] - self.raw_data['ema_200']
    
    def update_macd(self):
        # 12-day EMA
        self.raw_data['ema_12'] = self.raw_data['close'].ewm(span=12, adjust=False).mean()

        # 26-day EMA
        self.raw_data['ema_26'] = self.raw_data['close'].ewm(span=26, adjust=False).mean()

        # MACD Line
        self.raw_data['macd'] = self.raw_data['ema_12'] - self.raw_data['ema_26']

        # Signal Line (9-day EMA of the MACD line)
        self.raw_data['macd_signal'] = self.raw_data['macd'].ewm(span=9, adjust=False).mean()

        # MACD Histogram (MACD line - Signal line)
        self.raw_data['macd_histogram'] = self.raw_data['macd'] - self.raw_data['macd_signal']

        # Drop intermediate columns
        self.raw_data.drop(columns=['ema_12', 'ema_26'], inplace=True)
    
    def update_bbands(self):
        # Middle Band (20-day SMA)
        self.raw_data['bb_middle'] = self.raw_data['close'].rolling(window=20).mean()

        # Standard Deviation (20-day window)
        self.raw_data['bb_std'] = self.raw_data['close'].rolling(window=20).std()

        # Upper and Lower Bollinger Bands
        self.raw_data['bb_upper'] = self.raw_data['bb_middle'] + (2 * self.raw_data['bb_std'])
        self.raw_data['bb_lower'] = self.raw_data['bb_middle'] - (2 * self.raw_data['bb_std'])

        # Drop intermediate columns (bb_std)
        self.raw_data.drop(columns=['bb_std'], inplace=True)
    
    def update_slowd(self):
        # SlowK (14-day)
        low_14 = self.raw_data['low'].rolling(window=14).min()
        high_14 = self.raw_data['high'].rolling(window=14).max()
        self.raw_data['slowk'] = 100 * ((self.raw_data['close'] - low_14) / (high_14 - low_14))

        # SlowD (3-day SMA of SlowK)
        self.raw_data['slowd'] = self.raw_data['slowk'].rolling(window=3).mean()

        # Drop intermediate columns
        self.raw_data.drop(columns=['slowk'], inplace=True)

    def update_atr(self):
        # True Range (TR)
        self.raw_data['tr'] = self.raw_data[['high', 'low', 'close']].diff(axis=1).abs().max(axis=1)

        # ATR (14-day average of True Range)
        self.raw_data['atr'] = self.raw_data['tr'].rolling(window=14).mean()

        # Drop intermediate columns
        self.raw_data.drop(columns=['tr'], inplace=True)
    
    def update_lagged_features(self):
        self.raw_data['close_lag_1'] = self.raw_data['close'].shift(1)
        self.raw_data['close_lag_2'] = self.raw_data['close'].shift(2)
        self.raw_data['close_lag_5'] = self.raw_data['close'].shift(5)

        self.raw_data['rsi_lag_1'] = self.raw_data['rsi'].shift(1)
        self.raw_data['macd_lag_1'] = self.raw_data['macd'].shift(1)

        self.raw_data['return_lag_1'] = self.raw_data['return_10d'].shift(1)
    
    def update_volatility(self):
        self.raw_data['volatility_10d'] = self.raw_data['close'].rolling(window=10).std()
        self.raw_data['volatility_30d'] = self.raw_data['close'].rolling(window=30).std()
    
    def update_ROC(self):
        self.raw_data['roc_10'] = self.raw_data['close'].pct_change(periods=10)
        self.raw_data['roc_30'] = self.raw_data['close'].pct_change(periods=30)





    def update_holiday_proximity_data(self):
        url = f"https://finnhub.io/api/v1//stock/market-holiday?exchange=US&token={self.api_key_finnhub}"
        response = requests.get(url)
        raw_data = response.json()

        # Parse JSON
        data = raw_data['data']
        df = pd.DataFrame(columns = ['Event Name', 'Date'])
        new_row = []
        # Loop through json and grab data
        for i in range(len(data)):
            new_row.clear()
            new_row.append(data[i]['eventName'])
            new_row.append(data[i]['atDate'])

            # Append new_row to df
            df.loc[len(df)] = new_row

        # Make date col right dtype
        df['Date'] = pd.to_datetime(df['Date'])

        # Drop Christmas Eve rows
        df = df[~((df["Event Name"] == "Christmas Day") & (df['Date'].dt.day == 24))]

        # Drop Thanksgiving and keep Black Friday
        thanksgiving_df = df[(df["Event Name"] == "Thanksgiving Day")]
        df = df[(df["Event Name"] != "Thanksgiving Day")]
        # Loop through and drop Thanksgiving
        thanksgiving_df = thanksgiving_df.groupby(thanksgiving_df['Date'].dt.year)
        modified_group = []
        for name, group in thanksgiving_df:
            group = group.drop(group.index[1])
            # Rename to Black Friday
            group["Event Name"] = "Black Friday"
            modified_group.append(group)

        thanksgiving_df = pd.concat(modified_group)
        df = pd.concat([df, thanksgiving_df], ignore_index=True)
        df = df.sort_values(by='Date')

        # Drop non-official Independence Days
        df = df[~((df["Event Name"] == "Independence Day") & (df['Date'].dt.day != 4))]


        holiday_dates = df['Date'].to_numpy(dtype='datetime64[D]')
        dates = self.raw_data['Date'].to_numpy(dtype='datetime64[D]')
        dates = np.unique(dates)
        df = pd.DataFrame(columns=['Date', "Holiday Proximity"])
        # Find the nearest holiday by calcualting the interval between all holidays, and taking the min
        for date in dates:
            day_differences = []
            for holidays in holiday_dates:
                day_differences.append(np.abs(date - holidays).astype('timedelta64[D]').astype(int))
            nearest = np.min(day_differences)
            df.loc[len(df)] = [date, nearest]

        # Reformat date so it can be correctly grouped when merging
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # Check if no data is currently stored
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = df
        else:
            # Fill in holiday proximity data
            self.raw_data = pd.concat([self.raw_data, df], axis=0, join='outer')
    
    def update_date_data(self):
        date_df = pd.DataFrame({'day of year': pd.to_datetime(self.raw_data['date']).dt.dayofyear})
        self.raw_data['Sin Date'] = np.sin(2 * np.pi * date_df['day of year'] / 365)
        self.raw_data['Cos Date'] = np.cos(2 * np.pi * date_df['day of year'] / 365)
