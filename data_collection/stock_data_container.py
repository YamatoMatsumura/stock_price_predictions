import requests
import pandas as pd
import datetime as dt
import os
import random
import keyboard
import numpy as np

import data_collection.vpn_script as vpn_script


RUN_SCRIPT = True  # Controls whether to run VPN refresh script or not
ANALYTICS_WINDOW_SIZE = 10  # Window size for analytics portion (mean, median, etc.) Minimum of 10 days
ANALYTICS_RANGE = '500year'  # Cutoff range for data
TECHNICAL_WINDOW_SIZE = 5  # Window size for technical portion (SMA, EMA, Variance, etc.)
SENTIMENT_MISSING_PERCENT = 0.2  # Cutoff percent to ignore sentiment data from data set


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

        self.raw_data = None
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
            return df['Date'].iloc[0]
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return None
        
    def get_existing_data(self):
        self.training_data = pd.read_csv('data/' + self.ticker + '/trained_data.csv')

    
    def update_all_data(self, last_updated):
        print("Updating all data...")

        # self.update_sentiment_data()
        self.update_OHLC_data(last_updated)
        # self.update_mean_data()
        # self.update_return_data()
        # self.update_STDDev_data()
        # self.update_median_data()
        # self.update_SMA_data()
        # self.update_EMA_data()
        # self.update_STOCH_data()
        # self.update_RSI_data()
        # self.update_ADX_data()
        # self.update_CCI_data()
        # self.update_AROON_data()
        # self.update_BBANDS_data()
        # self.update_AD_data()
        # self.update_OBV_data()
        # self.update_holiday_proximity_data()
        # self.update_date_data()

        # Turn off vpn once done fetching data
        vpn_script.close_VPN(RUN_SCRIPT)


    def update_OHLC_data(self, start_date='1900-01-01'):
        while True:
            try:
                url = f"https://api.tiingo.com/tiingo/daily/{self.ticker}/prices"
                params = {
                    'startDate': start_date,
                    'endDate': dt.datetime.now().date(),
                    'format': 'json',
                    'sort': '-date'
                }
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Token {self.tiingo_keys[self.tiingo_key_index]}'
                }

                response = requests.get(url, params=params, headers=headers)
                data = response.json()
                break
            except requests.exceptions.RequestException as error:
                self._cycle_tiingo_key()
        
        # Check if no more data to fetch
        if not data:
            self.raw_data = pd.DataFrame()
            return

        df = pd.DataFrame(data)
        df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor'], inplace=True)
        df.rename(columns={'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low', 
                                'adjClose': 'Close', 'adjVolume': 'Volume', 'date': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df.reset_index(drop=True, inplace=True)

        # Check if no data is currently stored
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = df
        else:
            # Fill in OHLC data
            self.raw_data = pd.concat([self.raw_data, df], axis=0, join='outer')
    
    def _cycle_tiingo_key(self):
        self.tiingo_key_index = (self.tiingo_key_index + 1) % len(self.tiingo_keys)

    def update_sentiment_data(self):
        # Initialize empty dataframe
        columns = ['Date', 'Sentiment (Avg)', 'Number of Articles']
        df = pd.DataFrame(columns=columns)

        # Initialize day delta to adjust time frame to search for
        day_delta = 0

        while True:
            # Check for pause
            if keyboard.is_pressed('p'):
                print('Paused')
                vpn_script.toggle_VPN()
                while True:
                    if keyboard.is_pressed('esc'):
                        print('Resuming')
                        vpn_script.toggle_VPN()
                        break

            # Check halfway through if too many missing data points
            if day_delta == 365:
                missingDataCount = (df['Number of Articles'] == 0).sum()
                if missingDataCount > len(df) * SENTIMENT_MISSING_PERCENT:
                    print(f"Missing more than {SENTIMENT_MISSING_PERCENT * 100}% of sentiment data, not including...")
                    return
            
            # Cutoff at 2 years (plus some more just to be safe) since OHLC date cuts off about 2 years before current date
            if day_delta == 365*2 + 30:
                break

            # Initialize time frame to 24 hours
            day = dt.datetime.now() - dt.timedelta(days=day_delta)
            time_to = day.strftime('%Y%m%d') + 'T2359'
            time_from = day.strftime('%Y%m%d') + 'T0000'

            # Make API call
            url = ('https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
                    '&tickers=' + self.ticker + 
                    '&time_from=' + time_from + 
                    '&time_to=' + time_to + 
                    '&apikey=' + self.api_key_AV)
            r = requests.get(url)
            raw_data = r.json()

            # Check for max api call error
            if self._check_max_API_call(raw_data):
                raw_data = self._fix_max_API_call_fail(raw_data, url)
            
            # Check for end of sentiment data
            if self._check_sentiment_error(raw_data):
                # Break out of loop since no more sentiment data available
                break

            # Format data into pandas df
            new_row = self._parse_sentiment(raw_data)
            # Handle case when no articles found for that day
            if new_row is None:
                new_row = [(dt.datetime.now() - dt.timedelta(days=day_delta)).strftime('%Y-%m-%d'), 0, 0]
            df.loc[len(df)] = new_row

            # Adjust time frame back one day
            day_delta += 1

            # Check if only need up to a certain day
            if self.last_updated == (dt.datetime.now() - dt.timedelta(days=day_delta)).strftime('%Y-%m-%d'):
                break


        # Delete rows with missing data
        df = df[df['Number of Articles'] != 0]

        # Remove Number of Articles col since only using to filter out missing data
        df.drop(columns=['Number of Articles'], inplace=True)

        # Check if no data is currently stored
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = df
        else:
            # Fill in Sentiment data
            self.raw_data = pd.concat([self.raw_data, df], axis=0, join='outer')
    

    def _update_technical_indicator_data(self, category, custom_category=False):
        # Make API Call
        url = ('https://www.alphavantage.co/query?' +
               'function=' + category +
               '&symbol=' + self.ticker +
               '&interval=daily' + 
               '&time_period=' + str(TECHNICAL_WINDOW_SIZE) + 
               '&series_type=close' + 
               '&apikey=' + self.api_key_AV) 
        r = requests.get(url) 
        data = r.json() 

        # Check and fix max api call error
        if self._check_max_API_call(data):
            data = self._fix_max_API_call_fail(data, url)

        # Parse data
        if custom_category:
            data = data['Technical Analysis: ' + custom_category]
        else:
            data = data['Technical Analysis: ' + category]
        df = pd.DataFrame.from_dict(data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        # Check if no data is currently stored
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = df
        else:
            # Fill in sentiment data
            self.raw_data = pd.concat([self.raw_data, df], axis=0, join='outer')

    def update_SMA_data(self):
        self._update_technical_indicator_data('SMA')
    
    def update_EMA_data(self):
        self._update_technical_indicator_data('EMA')
    
    def update_STOCH_data(self):
        self._update_technical_indicator_data('STOCH')
    
    def update_RSI_data(self):
        self._update_technical_indicator_data('RSI')
    
    def update_ADX_data(self):
        self._update_technical_indicator_data('ADX')

    def update_CCI_data(self):
        self._update_technical_indicator_data('CCI')
    
    def update_AROON_data(self):
        self._update_technical_indicator_data('AROON')
    
    def update_BBANDS_data(self):
        self._update_technical_indicator_data('BBANDS')
    
    def update_AD_data(self):
        self._update_technical_indicator_data('AD', 'Chaikin A/D')
    
    def update_OBV_data(self):
        self._update_technical_indicator_data('OBV')

    
    def _update_analytics_data(self, category, category_name, annualized=False):
        # Make API Call
        url = ('https://www.alphavantage.co/query?' + 
               'function=ANALYTICS_SLIDING_WINDOW' + 
               '&SYMBOLS=' + self.ticker + 
               '&RANGE=' + ANALYTICS_RANGE +
               '&INTERVAL=DAILY' + 
               '&OHLC=close'
               '&WINDOW_SIZE=' + str(ANALYTICS_WINDOW_SIZE) + 
               '&CALCULATIONS=' + category +
               '&apikey=' + self.api_key_AV)
        r = requests.get(url)
        data = r.json()

        # Check and fix max api call error
        if self._check_max_API_call(data):
            data = self._fix_max_API_call_fail(data, url)

        # Adjusted name to take out annualized bit if needed
        if not annualized:
            adjusted_name = category
        else:
            adjusted_name = category[:-17]

        # Parse data
        data = data['payload']['RETURNS_CALCULATIONS'][category]['RUNNING_' + adjusted_name][self.ticker]
        data = data.items()


        # Convert to pandas df
        df = pd.DataFrame(data, columns=['Date', category_name])
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        
        # Check if no data is currently stored
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = df
        else:
            # Fill in sentiment data
            self.raw_data = pd.concat([self.raw_data, df], axis=0, join='outer')

    def update_mean_data(self):
        self._update_analytics_data('MEAN', 'Mean')
    
    def update_median_data(self):
        self._update_analytics_data('MEDIAN', 'Median')

    def update_return_data(self):
        self._update_analytics_data('CUMULATIVE_RETURN', 'Cumulative Return')
    
    def update_STDDev_data(self):
        self._update_analytics_data('STDDEV', 'STDDEV')


    def _parse_OHLC(self, raw_data):
        parsed_data = [] # Holds date, OHLC, and volume data in a 2d list
        for day in raw_data["Time Series (Daily)"]:
            new_row = []

            # Add formatted date data
            day = pd.to_datetime(day)
            day = day.strftime('%Y-%m-%d')
            new_row.append(day)

            # Add each OHLC and volume data
            for category in raw_data["Time Series (Daily)"][day]:
                new_row.append(raw_data["Time Series (Daily)"][day][category])
            parsed_data.append(new_row)

        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Return pandas data frame
        return pd.DataFrame(parsed_data, columns=columns)

    
    def _parse_sentiment(self, data):
        # Grab time, average sentiment, and total article data
        row = []
        total_articles = 0
        total_score = 0
    
        # Check for no articles found
        if data['items'] == "0":
            return None

        # Format date data
        day = pd.to_datetime(data['feed'][0]['time_published'])
        day = day.strftime('%Y-%m-%d')
        row.append(day)

        # Grab total score and article count
        for article in data['feed']:
            for sentiment in article['ticker_sentiment']:
                if sentiment['ticker'] == self.ticker:
                    total_score += float(sentiment['ticker_sentiment_score'])
                    total_articles += 1
        row.append(total_score / total_articles)
        row.append(int(total_articles))
        return row

    def _check_max_API_call(self, data):
        max_API_call_error = "our standard API rate limit is 25 requests per day."
        if 'Information' in data.keys() or 'Note' in data.keys() or 'error' in data.keys():
            if 'error' in data.keys():
                return True
            for key in data:
                if max_API_call_error in data[key]:
                    return True
        
        return False
    
    def _fix_max_API_call_fail(self, data, url):
        attemps = 0 # Counter to retry fixing method if multiple failed attemps

        # If need new api key
        if 'error' in data.keys():
            self._get_new_AV_API_key()
            url = self._get_new_AV_URL(url)
        # If need to refresh VPN
        else:
            vpn_script.vpn_refresh_script(self.script_first_time_called, RUN_SCRIPT)
            self.script_first_time_called = False

        # Keep trying to make request until max api call doesn't happen
        while self._check_max_API_call(data):
            r = requests.get(url)
            data = r.json()
            attemps += 1
            # If just need a new api key
            if 'error' in data.keys():
                self._get_new_AV_API_key()
                url = self._get_new_AV_URL(url)
                continue

            # Refresh vpn again if tried to reconnect multiple times and not working
            if attemps == 3:
                vpn_script.vpn_refresh_script(self.script_first_time_called, RUN_SCRIPT)
                attemps = 0
        return data

    def _check_sentiment_error(self, data):
        no_articles_error = "No articles found. Please adjust the time range or refer to the API documentation"
        if 'Information' in data.keys() or 'Note' in data.keys():
            for key in data:
                if no_articles_error in data[key]:
                    return True
        else:
            return False
    
    def _get_new_AV_API_key(self):
        nextKeyPath = 'api_keys/AV/key' + str(random.randint(0, self.AV_key_count - 1)) + '.txt'
        with open(nextKeyPath) as file:
            self.api_key_AV = file.read()
    
    def _get_new_AV_URL(self, url):
        url = url[:-16]
        url += self.api_key_AV
        return url

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
        date_df = pd.DataFrame({'day of year': pd.to_datetime(self.raw_data['Date']).dt.dayofyear})
        date_df = date_df.drop_duplicates().reset_index(drop=True)
        self.raw_data['Sin Date'] = np.sin(2 * np.pi * date_df['day of year'] / 365)
        self.raw_data['Cos Date'] = np.cos(2 * np.pi * date_df['day of year'] / 365)
