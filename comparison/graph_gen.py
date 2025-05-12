import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reusable plotting function
def base_plot(fig_name, title, x, y_dict, linestyle_dict=None, color_dict=None, y_tick_interval=100):
    plt.figure(figsize=(12, 6))
    
    for label, y_values in y_dict.items():
        linestyle = linestyle_dict.get(label, '-') if linestyle_dict else '-'
        color = color_dict.get(label, None) if color_dict else None
        plt.plot(x, y_values, label=label, linestyle=linestyle, color=color)

    # Determine y-axis limits from the data
    all_values = pd.concat(y_dict.values(), axis=1)
    y_min = all_values.min().min()
    y_max = all_values.max().max()

    y_min_rounded = int(np.floor(y_min / y_tick_interval) * y_tick_interval)
    y_max_rounded = int(np.ceil(y_max / y_tick_interval) * y_tick_interval)

    plt.ylim(y_min_rounded, y_max_rounded)
    plt.yticks(np.arange(y_min_rounded, y_max_rounded + y_tick_interval, y_tick_interval))

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


# --- Load and prepare data ---

# Simulation data
results = pd.read_csv('../sim_dump_3000/liquid_balance_over_time.csv')
results['date'] = pd.to_datetime(results['date'])

# Plot 1: Simulation only
base_plot(
    fig_name='comparison_plot_isolated.png',
    title='Portfolio Value Over Time',
    x=results['date'],
    y_dict={'Simulation': results['balance']},
    linestyle_dict={'Simulation': '--'},
    color_dict={'Simulation': 'green'}
)


# Index fund data
QQQ = pd.read_csv('QQQ.csv')
SPY = pd.read_csv('SPY.csv')

QQQ['date'] = pd.to_datetime(QQQ['Date'])
SPY['date'] = pd.to_datetime(SPY['Date'])

QQQ = QQQ.rename(columns={'balance': 'QQQ'})
SPY = SPY.rename(columns={'balance': 'SPY'})
results = results.rename(columns={'balance': 'Simulation'})

# Merge all on date
merged = QQQ[['date', 'QQQ']].merge(SPY[['date', 'SPY']], on='date')
merged = merged.merge(results[['date', 'Simulation']], on='date')

# Plot 2: Comparison with SPY and QQQ
base_plot(
    fig_name='comparison_plot.png',
    title='Portfolio Value Over Time',
    x=merged['date'],
    y_dict={
        'QQQ': merged['QQQ'],
        'SPY': merged['SPY'],
        'Simulation': merged['Simulation']
    },
    linestyle_dict={
        'Simulation': '--'
    },
    color_dict={
        'Simulation': 'green'
    }
)
