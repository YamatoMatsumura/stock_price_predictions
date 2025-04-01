import pandas as pd


from data_collection.stock_data_container import StockDataContainer
import data_collection.data_utils as dataUtils
from config import(
    STOCK_NAMES, SEQUENCE_LENGTH
)

def run_trade_script():
    if STOCK_NAMES[0] == 'ALL':
        dataUtils.get_all_tickers()

    for stock in STOCK_NAMES:
        
        # Initialize container for stock data
        stock_data = StockDataContainer(stock)

        # dataUtils.backup_data(stock_data.ticker)
        stock_data.update_all_data((pd.to_datetime(stock_data.last_updated) + pd.Timedelta(days=1)))

        # Check for no new data
        if stock_data.raw_data.empty:
            return

        new_data = dataUtils.clean_and_process_data(stock_data.raw_data)
        num_new_data = len(new_data)

        # Merge old and new data
        stock_data.training_data = dataUtils.merge_with_old_data(stock_data)

        dataUtils.save_data(stock_data.training_data, stock_data.ticker)

        # Get rid of data that the model has already trained on
        modified_training_data = stock_data.training_data.iloc[:num_new_data + 2*SEQUENCE_LENGTH - 1 + num_new_data]
        # Reverse data so model trains from oldest to newest
        modified_training_data = modified_training_data.iloc[::-1].reset_index(drop=True)
        # Drop Date column since only used for debugging/data alignment purposes
        modified_training_data = modified_training_data.drop(columns=['Date'])


        features, labels = dataUtils.create_windowed_dataset(modified_training_data)

        training_features, training_labels, testing_features, testing_labels = dataUtils.split_dataset(features, labels)
        print(training_labels)
        print("---------------"*5)
        print(testing_labels)


        (
            scaled_training_features, 
            scaled_training_labels, 
            scaled_testing_features, 
            scaled_testing_labels, 
            feature_scaler, 
            label_scaler,
        ) = dataUtils.scale_dataset(training_features, training_labels, testing_features, testing_labels)





if __name__ == '__main__':
    run_trade_script()



