import requests
import pandas as pd
import datetime as dt
import os
import random
import keyboard

import vpn_script


RUN_SCRIPT = True  # Controls whether to run VPN refresh script or not
ANALYTICS_WINDOW_SIZE = 10  # Window size for analytics portion (mean, median, etc.) Minimum of 10 days
ANALYTICS_RANGE = '3year'  # Cutoff range for data
TECHNICAL_WINDOW_SIZE = 5  # Window size for technical portion (SMA, EMA, Variance, etc.)
SENTIMENT_MISSING_PERCENT = 0.2  # Cutoff percent to ignore sentiment data from data set


class StockDataContainer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.api_key_AV = None
        self.api_key_polygon = None
        self.AV_key_count = None
        self.data = None
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
        self.data = pd.read_csv('data/' + self.ticker + '/trained_data.csv')

    
    def update_all_data(self):
        print("Updating all data...")

        self.update_sentiment_data()
        self.update_OHLC_data()
        self.update_mean_data()
        self.update_return_data()
        self.update_STDDev_data()
        self.update_median_data()
        self.update_SMA_data()
        self.update_EMA_data()
        self.update_STOCH_data()
        self.update_RSI_data()
        self.update_ADX_data()
        self.update_CCI_data()
        self.update_AROON_data()
        self.update_BBANDS_data()
        self.update_AD_data()
        self.update_OBV_data()
        # self.updateDateData()

        # Turn off vpn once done fetching data
        vpn_script.close_VPN(RUN_SCRIPT)


    def update_OHLC_data(self):
        # Call Polygon API to get OHLC data
        end_date = dt.datetime.today().strftime('%Y-%m-%d')
        start_date = '2022-05-20' # Cutoff date is roughly 2 years before current date
        url = f'https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key_polygon}'
        response = requests.get(url)
        df = pd.DataFrame(response.json()['results'])

        # Format response
        df['Date'] = pd.to_datetime(df['t'] / 1000, unit='s')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.rename(columns= {
            'v': 'Volume',
            'vw': 'Volume Weighted Average',
            'o': 'Open',
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            'n': 'Transactions'
            })
        df.drop(columns=['t'], inplace=True)
        df = df.iloc[:, [7, 2, 4, 5, 3, 0, 1, 6]]

        # Reverse so newest date is at the top
        df = df.iloc[::-1]

        # Check if no data is currently stored
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in OHLC data
            self.data = pd.concat([self.data, df], axis=0, join='outer')

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
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in Sentiment data
            self.data = pd.concat([self.data, df], axis=0, join='outer')

    # def updateDateData(self):
    #     # Split date into individual components to feed into network
    #     self.data['Date'] = pd.to_datetime(self.data['Date'])
    

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
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in sentiment data
            self.data = pd.concat([self.data, df], axis=0, join='outer')

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
        if self.data is None or self.data.empty:
            self.data = df
        else:
            # Fill in sentiment data
            self.data = pd.concat([self.data, df], axis=0, join='outer')

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
        max_API_call_error = "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."
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
