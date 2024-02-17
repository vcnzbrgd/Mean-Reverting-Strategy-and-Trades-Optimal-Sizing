import pandas as pd
import numpy as np


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
    signal_long2 = ((s < ma_fast) & (s < ma_slow)) * 1 # vado long se prezzo minore di entrambe medie mobili???

    # short signal 2
    signal_short2 = ((s > ma_fast) & (s > ma_slow)) * 1 # vado short se prezzo maggiore di entrambe medie mobili???

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
    
    trades_returns = (df['price_close'] / df['price_open'])**df['direction'] - 1
    
    return trades_returns