import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# as convention 1 week is 5 trading days and 1 month is 21 trading days so that each year has 252 trading days

def meanrev_signal(s,
                   long_params={10:0.05, 21:0.15, 63:0.25},
                   short_params={10:0.95, 21:0.85, 63:0.75},
                   ma_fast_wdw = 20,
                   ma_slow_wdw = 60
                   ):

    """
    Calculate mean reversion signals for a given price series.
    
    This function generates trading signals based on the mean reversion strategy.
    It considers both long and short positions based on price quantiles and moving averages.
    
    Parameters:
    - s (pd.Series): The price series to analyze.
    - long_params (dict): Parameters for generating long signals, where keys are window lengths and values are quantiles.
    - short_params (dict): Parameters for generating short signals, where keys are window lengths and values are quantiles.
    - ma_fast_wdw (int): Window length for the fast moving average.
    - ma_slow_wdw (int): Window length for the slow moving average.
    
    Returns:
    - pd.Series: A series of signals where 1 represents a long signal, -1 represents a short signal, and 0 represents no signal.
    """


    # long signal 1
    signal_long1 = pd.DataFrame()

    for k,v in long_params.items():
        signal_long1[f'qtle_px{k}'] = s.rolling(k).quantile(v)

    signal_long1 = (s < signal_long1.min(axis=1)) * 1 

    # short signal 1
    signal_short1 = pd.DataFrame()

    for k,v in short_params.items():
        signal_short1[f'qtle_px{k}'] = s.rolling(k).quantile(v)

    signal_short1 = (s > signal_short1.max(axis=1)) * -1

    
    ma_fast = s.rolling(ma_fast_wdw).mean()
    ma_slow = s.rolling(ma_slow_wdw).mean()
    
    # long signal 2
    signal_long2 = ((s < ma_fast) & (s < ma_slow)) * 1
    
    # short signal 2
    signal_short2 = ((s > ma_fast) & (s > ma_slow)) * 1

    # make intersection of the two signals
    signal_long = (signal_long1 * signal_long2)
    signal_short = (signal_short1 * signal_short2)

    signal = signal_long + signal_short

    return signal



def tp_sl_rule(s,
                dt_open,
                direction,
                supportive_pctl_move = 0.3,
                counter_pctl_move = 0.2,
                   ):
    """
    Calculate the take profit (TP) and stop loss (SL) levels based on historical 
    monthly returns of a financial instrument up to a specified opening date.

    Parameters:
    - s (pandas.Series): Price data of the financial instrument.
    - dt_open (datetime-like): The opening date for the calculation. Only data up to this date will be considered.
    - direction (int): Trading direction, where 1 indicates a long position and -1 indicates a short position.
    - supportive_pctl_move (float, optional): The percentile of positive returns to determine the TP level in a supportive market move. Defaults to 0.3.
    - counter_pctl_move (float, optional): The percentile of negative returns to determine the SL level in a counter market move. Defaults to 0.2.

    Returns:
    - tuple: A tuple containing the TP and SL levels as floats. TP and SL are based on 
      the quantiles of positive and negative monthly returns, respectively, adjusted for 
      the trading direction.
    """
    monthly_rets = s.loc[:dt_open].pct_change(21).dropna()

    pos_monthly_rets = monthly_rets[monthly_rets>0]
    neg_monthly_rets = monthly_rets[monthly_rets<0]

    if direction == 1:
        tp_return = pos_monthly_rets.quantile(supportive_pctl_move)
        sl_return = neg_monthly_rets.quantile(1 - counter_pctl_move)
    elif direction == -1:
        tp_return = - neg_monthly_rets.quantile(1 - supportive_pctl_move)
        sl_return = - pos_monthly_rets.quantile(counter_pctl_move)
    
    return tp_return, sl_return



def trade_return(df):
    """
    Compute the returns of a list of trades

    Parameters:
    - df (pandas.DataFrame): a df containing three columns: price_close with the closing price of the trade, price_open with the opening price of the trade and 
    direction equal to 1 if trade is long, -1 if short

    Returns:
    - trades_returns (pandas.Series): a series with trade's return
    """
    
    trades_returns = (df['price_close'] / df['price_open'] - 1) * df['direction']
    
    return trades_returns



def hit_ratio(df):
    """
    Computes Hit ratio defined as winning trades / total number of trades
    Parameters:
    - df (pandas.DataFrame): a df with a column named return
    Returns:
    - hit_ratio (int): the Hit ratio
    """

    hit_ratio = len(df[df['return']>0]) / len(df)

    return hit_ratio



def win_loss(df):
    """
    Computes Win-Loss ratio defined as average return of winning trades / average absolute value of return of losing trades
    Parameters:
    - df (pandas.DataFrame): a df with a column named return
    Returns:
    - win_loss (int): the Win-Loss ratio
    """
        
    win_loss = df[df['return']>0]['return'].mean() / np.abs(df[df['return']<=0]['return'].mean())
    
    return win_loss



# PLOT Functions

def violin_plot_grouped(data, group_column, value_column, figsize=(20, 8)):
    """
    Creates a violin plot for a specified numerical column, grouped by a specified categorical column,
    using Matplotlib directly. This version allows specifying the figure size.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the data.
    - group_column (str): The name of the column to group the data by.
    - value_column (str): The name of the numerical column for which the violin plot will be created.
    - figsize (tuple of int, optional): The size of the figure (width, height) in inches. Defaults to (10, 6).
    """

    # Set the figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare the data for the violin plot
    unique_groups = data[group_column].unique()
    grouped_data = [data[value_column][data[group_column] == group] for group in unique_groups]

    # Create the violin plot
    ax.violinplot(grouped_data, showmeans=False, showmedians=True)
    
    # Set the x-ticks to correspond to the groups and rotate them for better readability
    ax.set_xticks(range(1, len(unique_groups) + 1))
    ax.set_xticklabels(unique_groups, rotation=45, ha="right")

    # Enhance the plot
    ax.set_title(f'Violin Plot of {value_column} Grouped by {group_column}')
    ax.set_xlabel(group_column)
    ax.set_ylabel(value_column)
    ax.grid(True)

    # Display the plot
    plt.tight_layout()  # Adjust layout to make room for the rotated labels
    plt.show()



def plot_histogram(series, title='Title', figsize=(10, 6), bins=30):
    """
    Generates a histogram for a given pandas Series, marking the 25th and 75th percentiles with vertical lines
    and displaying their values in a text box.

    Parameters:
    - series (pandas.Series): The data to plot.
    - title (str, optional): The title of the histogram. Defaults to 'Histogram'.
    - figsize (tuple of int, optional): The figure size in inches, given as (width, height). Defaults to (10, 6).
    - bins (int, optional): The number of bins for the histogram. Defaults to 30.
    """
    # Calculate percentiles
    p25 = np.percentile(series, 25)
    p50 = np.percentile(series, 50)
    p75 = np.percentile(series, 75)

    # Create the histogram
    plt.figure(figsize=figsize)
    plt.hist(series, bins=bins, color='skyblue', edgecolor='black')

    # Add vertical lines for the 25th and 75th percentiles
    plt.axvline(p25, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(p75, color='green', linestyle='dashed', linewidth=1)

    # Prepare text for the box showing percentile values
    textstr = f'25th percentile: {p25:.4f}\nmedian: {p50:.4f}\n75th percentile: {p75:.4f}'

    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

    # Set title and labels
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Show plot
    plt.show()