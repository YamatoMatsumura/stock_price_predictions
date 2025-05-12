import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def plot(df, columns, filename='plot.png'):
    # Create the plot
    plt.figure(figsize=(12, 6))
    df['date'] = pd.to_datetime(df['date'])

    # Plot each specified column
    for col in columns:
        if col in df.columns:
            plt.plot(df['date'], df[col], label=col, linewidth=2)

    # Adding title and labels
    plt.title('Closing Price with 50 Day SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Displaying the legend
    plt.legend()

    # Show the plot and save to file
    plt.tight_layout()
    plt.savefig(filename)

def main():
    df = pd.read_csv('../data/AAPL/trained_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[['open', 'high', 'low', 'close', 'volume']]

    summary = df.describe()
    summary.drop('count', inplace=True)
    print(summary)


    columns = ['volume', 'volatility_10d']
    filename = 'testing.png'

    # plot(df, columns, filename



if __name__ == "__main__":
    main()