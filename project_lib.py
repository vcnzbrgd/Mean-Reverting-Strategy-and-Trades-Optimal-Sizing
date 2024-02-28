import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random

##############################################
############### MEAN REVERSION ###############
##############################################

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



def run_bt_pipeline(df,
            TRADED_SECURITIES,
            LONG_PARAMS,
            SHORT_PARAMS,
            MA_FAST_WDW,
            MA_SLOW_WDW,
            SUPPORTIVE_PCTL_MOVE,
            COUNTER_PCTL_MOVE,
            TRADES_MAX_DAYS,
            MAX_DOLLAR_LOSS,
            SIGMA_WDW,
            GAMMA,
            ):
    """
    Run BT pipeline
    """


    # list to store all trades
    trades_list = []
    trades_pnl = {}

    for security_id in tqdm.tqdm(TRADED_SECURITIES):
        # generate signal based on standard strategy
        signal_all = meanrev_signal(df[security_id],
                                    long_params = LONG_PARAMS,
                                    short_params = SHORT_PARAMS,
                                    ma_fast_wdw = MA_FAST_WDW,
                                    ma_slow_wdw = MA_SLOW_WDW
                    )
        
        # compute Historical Volatility
        hist_vol = df[security_id].pct_change().rolling(SIGMA_WDW).std() * np.sqrt(252)

        # "start your backtest at t-10". hard coding initial date for BT (and final date to avoid missing data)
        signal_all = signal_all.loc['2014-02-12':'2024-01-20']

        # extract from all signal only the actual open buy/sell triggers
        signal_do = signal_all[signal_all!= 0]

        # iterates over all buy/sell signal and execute orders
        for dt_open, direction in zip(signal_do.index, signal_do):
            
            # create a unique id for each trade
            trade_id = security_id+'#'+str(dt_open)[:10]

            # price at which the trade is open
            price_open = df.loc[dt_open, security_id]

            # compute TP/SL returns and prices
            tp_return, sl_return = tp_sl_rule(df[security_id],
                                                dt_open,
                                                direction,
                                                supportive_pctl_move = SUPPORTIVE_PCTL_MOVE,
                                                counter_pctl_move = COUNTER_PCTL_MOVE
                                            )
            price_tp = price_open * (1 + tp_return*direction)
            price_sl = price_open * (1 + sl_return*direction)

            if GAMMA == None:
                # compute the optimal sizing such that all trades loses the same amount of $ if SL is hitted
                quantity = MAX_DOLLAR_LOSS/((price_open - price_sl)*direction)
            else:
                quantity = MAX_DOLLAR_LOSS/((price_open - price_sl)*direction) * (GAMMA/hist_vol.loc[dt_open])

            # store all trade info in df
            trade = pd.DataFrame({'security_id':security_id,
                                    'dt_open':str(dt_open)[:10],
                                    'price_open':price_open,
                                    'direction':direction,
                                    'quantity':quantity,
                                    'price_tp':price_tp,
                                    'price_sl':price_sl}, index=[trade_id])

            # create a temporary df that contains prices of instrument during trade
            dt_open_idxdf = list(df.index).index(dt_open)
            temp_px = df[[security_id]].iloc[dt_open_idxdf + 1: dt_open_idxdf + TRADES_MAX_DAYS + 1]
            temp_px['price_open'] = price_open
            temp_px['direction'] = direction
            temp_px['quantity'] = quantity
            temp_px['price_tp'] = price_tp
            temp_px['price_sl'] = price_sl

            # Check if and when a TP/SL is triggered and cut the temporary df accordingly
            if direction==1:
                temp_px['tp_hitted'] = (temp_px[security_id] > temp_px['price_tp']) * 1
                temp_px['sl_hitted'] = (temp_px[security_id] < temp_px['price_sl']) * 1
            elif direction==-1:
                temp_px['tp_hitted'] = (temp_px[security_id] < temp_px['price_tp']) * 1
                temp_px['sl_hitted'] = (temp_px[security_id] > temp_px['price_sl']) * 1

            temp_px['tp_sl_hitted'] = temp_px['tp_hitted'] + temp_px['sl_hitted']

            if 1 in list(temp_px['tp_sl_hitted']):
                dt_close = str(temp_px[temp_px['tp_sl_hitted']==1].index[:1][0])[:10]
                exit_type = 'TP/SL exit'
            else:
                dt_close = str(list(temp_px.index)[-1:][0])[:10]
                exit_type = 'max duration'

            # cut temp_px at closing date
            temp_px = temp_px.loc[:dt_close]
            # compute PnL of the trade during the days it was open and append to Portfolio PnL list
            temp_px['pnl'] = temp_px[security_id] * temp_px['quantity']
            temp_px['daily_return'] = temp_px[security_id].pct_change().fillna(
                temp_px[security_id].iloc[0]/temp_px['price_open'].iloc[0]-1)

            # store closing price of the trade
            price_close = temp_px.loc[dt_close, security_id]

            # add closing trade date and price to trade df
            trade['dt_close'] = dt_close
            trade['price_close'] = price_close
            trade['duration'] = len(temp_px)
            trade['exit_condition'] = exit_type

            # compute the annualized volatility of the trade
            trade['ann_volatility'] = temp_px['daily_return'].std() * np.sqrt(252)

            # collect trade and pnl
            trades_list.append(trade)
            trades_pnl[trade_id] = temp_px

    # check if trades_list is non-empty
    if len(trades_list)==0:
        raise Exception('No trades have been executed')

    trades_list = pd.concat(trades_list)
    # compute trades' returns, sharpe 1Y and dollar value of position at each open
    trades_list['return'] = trade_return(trades_list)
    trades_list['daily_return'] = trades_list['return'] / trades_list['duration']
    trades_list['sharpe_ratio'] = trades_list['daily_return']*252 / trades_list['ann_volatility']

    # trim Sharpe ratio to reduce the impact of outliers on average measure
    trim_sharpe_up = trades_list['sharpe_ratio'].quantile(0.95)
    trim_sharpe_dwn = trades_list['sharpe_ratio'].quantile(0.05)
    trades_list['sharpe_ratio'] = trades_list['sharpe_ratio'].mask(trades_list['sharpe_ratio'] > trim_sharpe_up, trim_sharpe_up)
    trades_list['sharpe_ratio'] = trades_list['sharpe_ratio'].mask(trades_list['sharpe_ratio'] < trim_sharpe_dwn, trim_sharpe_dwn)

    # compute the dollar value postion at open and the pseudo-weight of the trade in the portfolio 
    trades_list['position_dollar_value_open'] = trades_list['quantity']*(
        trades_list['price_open']+trades_list['price_close'])/2
    trades_list['weight'] = (trades_list['position_dollar_value_open']*trades_list['duration']/
        (trades_list['position_dollar_value_open']*trades_list['duration']).sum())


    # split between TP and SL exit explicitly
    trades_list['exit_condition'] = trades_list['exit_condition'].mask(
        (trades_list['exit_condition']=='TP/SL exit')&(trades_list['return']>0), 'TP exit')
    trades_list['exit_condition'] = trades_list['exit_condition'].mask(
        (trades_list['exit_condition']=='TP/SL exit')&(trades_list['return']<0), 'SL exit')
    

    return trades_list, trades_pnl



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



##############################################
############### OPTIMAL SIZING ###############
##############################################



def optimal_number_trades(df,
                            incipit_date,
                            sigma_sl,
                            sigma_tp,
                            max_trade_duration, 
                            ptf_vol_tgt,
                            aum, 
                            aum_lost_sl):
    """
    function does....
    """

    if (incipit_date <= '2014-02-12') | (incipit_date >= '2023-02-12'):
        raise ValueError('incipit_date must be between 2014-02-12 and 2023-02-12')

    # refill Sat&Sun
    refilled = df.reindex(pd.date_range(df.index.min(), df.index.max()), method='ffill')

    # create list of dates for simulated trades
    date_index_reference = df.loc[pd.to_datetime(incipit_date):pd.to_datetime(incipit_date)+pd.DateOffset(years=1)].index
    startdates = list(date_index_reference.astype(str))

    # create df to store results of the trades
    trades_df = pd.DataFrame()
    ctrvl_aum = pd.DataFrame(index = date_index_reference)
    ctrvl_aum['aum'] = aum
    ctrvl_pos = pd.DataFrame(index = date_index_reference)

    # initialization of portfolio volatility
    portfolio_volatility = 0

    # generate random trades until portfolio volatility is below target
    while portfolio_volatility < ptf_vol_tgt:
        security_id = random.choice(list(df.columns))
        trade_entry_date = random.choice(startdates)
        direction = random.choice([-1, 1])

        # compute 5Y Monthly standard deviation starting from date of trade entry and going backward to compute trade's TP/SL
        hist_stdev = refilled.loc[
            refilled.index.isin(pd.date_range(end=trade_entry_date, periods=60+1, freq=pd.DateOffset(months=1))), security_id
                ].pct_change().dropna().std()
        
        # create trade's PnL df
        trade_pnl = df.loc[trade_entry_date:, [security_id]].head(max_trade_duration+1)
        trade_pnl['security_id'] = security_id
        trade_pnl['direction'] = direction
        trade_pnl['hist_volatility'] = hist_stdev
        trade_pnl['entry_price'] = df.loc[trade_entry_date, security_id]
        trade_pnl['tp_price'] = trade_pnl['entry_price'] * (1 + hist_stdev * sigma_tp * direction)
        trade_pnl['sl_price'] = trade_pnl['entry_price'] * (1 + hist_stdev * sigma_sl * direction)
        trade_pnl['quantity'] = (aum * aum_lost_sl) / ((trade_pnl['entry_price'] - trade_pnl['sl_price']) * direction)

        # check if TP/SL have been hitted 
        if direction == 1:
            trade_pnl['tp_hit'] = (trade_pnl[security_id] > trade_pnl['tp_price']) * 1
            trade_pnl['sl_hit'] = (trade_pnl[security_id] < trade_pnl['sl_price']) * 1
        elif direction == -1:
            trade_pnl['tp_hit'] = (trade_pnl[security_id] < trade_pnl['tp_price']) * 1
            trade_pnl['sl_hit'] = (trade_pnl[security_id] > trade_pnl['sl_price']) * 1

        trade_pnl['tp_sl_hit'] = trade_pnl['tp_hit'] + trade_pnl['sl_hit']

        # make exit condition from the trade explicit
        if trade_pnl['tp_sl_hit'].sum()==0:
            trade_pnl['exit_condition'] = 'max_duration'
            trade_pnl['exit_price'] = trade_pnl[security_id].iloc[-1]
        else:
            trade_pnl = trade_pnl.loc[:trade_pnl[trade_pnl['tp_sl_hit'] == 1].index[0]].copy()
            trade_pnl['exit_condition'] = 'stop_loss' if trade_pnl['sl_hit'].sum()==1 else 'take_profit'
            trade_pnl['exit_price'] = trade_pnl['sl_price'] if trade_pnl['sl_hit'].sum()==1 else trade_pnl['tp_price']

        # first day must be subtracted as it is the day when the trade is open
        trade_pnl['duration'] = len(trade_pnl)-1

        # store the trade
        trades_df = pd.concat([trades_df, trade_pnl[['security_id', 'direction', 'hist_volatility', 'entry_price',
            'tp_price', 'sl_price', 'quantity', 'exit_condition', 'exit_price', 'duration']].drop_duplicates()]).sort_index()
        
        # store controvalore
        ctrvl_aum = pd.concat([ctrvl_aum, 
                        (trade_pnl['entry_price'] * trade_pnl['quantity']).rename(f'{security_id}#{trade_entry_date}')], axis=1)
        
        ctrvl_pos = pd.concat([ctrvl_pos, 
                        (trade_pnl[security_id] * trade_pnl['quantity']).rename(f'{security_id}#{trade_entry_date}')], axis=1)
        
        ctrvl_aum = ctrvl_aum.ffill().fillna(0)
        aum_net = ctrvl_aum['aum'] - ctrvl_aum.iloc[:, 1:].sum(axis=1)

        portfolio_volatility = (ctrvl_pos.ffill().fillna(0).sum(axis=1) + aum_net).pct_change().std() * np.sqrt(252)

    number_of_trades = len(trades_df)
    
    return portfolio_volatility, number_of_trades



def mc_opt_n_trades(df,
                    incipit_date,
                    sigma_sl,
                    sigma_tp,
                    max_trade_duration, 
                    ptf_vol_tgt,
                    aum, 
                    aum_lost_sl,
                    iterations=100):
    """
    
    """

    prob_res = {}
    for i in tqdm.tqdm(range(iterations)):
        portfolio_volatility, number_of_trades = optimal_number_trades(df,
                                incipit_date,
                                sigma_sl,
                                sigma_tp,
                                max_trade_duration, 
                                ptf_vol_tgt,
                                aum, 
                                aum_lost_sl)
        
        prob_res[portfolio_volatility] = number_of_trades

    prob_res = pd.DataFrame(prob_res, index=['number_of_trades']).T.reset_index().rename({'index':'ptf_volat'}, axis=1)
    print(f"optimal number of trades: {round(prob_res['number_of_trades'].mean())} \u00B1 {round(prob_res['number_of_trades'].std())}")



def forecast_portfolio_volatility(df,
                                    incipit_date,
                                    sigma_sl,
                                    sigma_tp,
                                    max_trade_duration, 
                                    number_of_trades,
                                    aum, 
                                    aum_lost_sl):
    """
    function does....
    """

    if (incipit_date <= '2014-02-12') | (incipit_date >= '2023-02-12'):
        raise ValueError('incipit_date must be between 2014-02-12 and 2023-02-12')

    # refill Sat&Sun
    refilled = df.reindex(pd.date_range(df.index.min(), df.index.max()), method='ffill')

    # create list of dates for simulated trades
    date_index_reference = df.loc[pd.to_datetime(incipit_date):pd.to_datetime(incipit_date)+pd.DateOffset(years=1)].index
    startdates = list(date_index_reference.astype(str))

    # create df to store results of the trades
    trades_df = pd.DataFrame()
    ctrvl_aum = pd.DataFrame(index = date_index_reference)
    ctrvl_aum['aum'] = aum
    ctrvl_pos = pd.DataFrame(index = date_index_reference)

    # generate random trades until portfolio volatility is below target
    for _ in range(number_of_trades):
        security_id = random.choice(list(df.columns))
        trade_entry_date = random.choice(startdates)
        direction = random.choice([-1, 1])

        # compute 5Y Monthly standard deviation starting from date of trade entry and going backward to compute trade's TP/SL
        hist_stdev = refilled.loc[
            refilled.index.isin(pd.date_range(end=trade_entry_date, periods=60+1, freq=pd.DateOffset(months=1))), security_id
                ].pct_change().dropna().std()
        
        # create trade's PnL df
        trade_pnl = df.loc[trade_entry_date:, [security_id]].head(max_trade_duration+1)
        trade_pnl['security_id'] = security_id
        trade_pnl['direction'] = direction
        trade_pnl['hist_volatility'] = hist_stdev
        trade_pnl['entry_price'] = df.loc[trade_entry_date, security_id]
        trade_pnl['tp_price'] = trade_pnl['entry_price'] * (1 + hist_stdev * sigma_tp * direction)
        trade_pnl['sl_price'] = trade_pnl['entry_price'] * (1 + hist_stdev * sigma_sl * direction)
        trade_pnl['quantity'] = (aum * aum_lost_sl) / ((trade_pnl['entry_price'] - trade_pnl['sl_price']) * direction)

        # check if TP/SL have been hitted 
        if direction == 1:
            trade_pnl['tp_hit'] = (trade_pnl[security_id] > trade_pnl['tp_price']) * 1
            trade_pnl['sl_hit'] = (trade_pnl[security_id] < trade_pnl['sl_price']) * 1
        elif direction == -1:
            trade_pnl['tp_hit'] = (trade_pnl[security_id] < trade_pnl['tp_price']) * 1
            trade_pnl['sl_hit'] = (trade_pnl[security_id] > trade_pnl['sl_price']) * 1

        trade_pnl['tp_sl_hit'] = trade_pnl['tp_hit'] + trade_pnl['sl_hit']

        # make exit condition from the trade explicit
        if trade_pnl['tp_sl_hit'].sum()==0:
            trade_pnl['exit_condition'] = 'max_duration'
            trade_pnl['exit_price'] = trade_pnl[security_id].iloc[-1]
        else:
            trade_pnl = trade_pnl.loc[:trade_pnl[trade_pnl['tp_sl_hit'] == 1].index[0]].copy()
            trade_pnl['exit_condition'] = 'stop_loss' if trade_pnl['sl_hit'].sum()==1 else 'take_profit'
            trade_pnl['exit_price'] = trade_pnl['sl_price'] if trade_pnl['sl_hit'].sum()==1 else trade_pnl['tp_price']

        # first day must be subtracted as it is the day when the trade is open
        trade_pnl['duration'] = len(trade_pnl)-1

        # store the trade
        trades_df = pd.concat([trades_df, trade_pnl[['security_id', 'direction', 'hist_volatility', 'entry_price',
            'tp_price', 'sl_price', 'quantity', 'exit_condition', 'exit_price', 'duration']].drop_duplicates()]).sort_index()
        
        # store controvalore
        ctrvl_aum = pd.concat([ctrvl_aum, 
                        (trade_pnl['entry_price'] * trade_pnl['quantity']).rename(f'{security_id}#{trade_entry_date}')], axis=1)
        
        ctrvl_pos = pd.concat([ctrvl_pos, 
                        (trade_pnl[security_id] * trade_pnl['quantity']).rename(f'{security_id}#{trade_entry_date}')], axis=1)
        
        ctrvl_aum = ctrvl_aum.ffill().fillna(0)
        aum_net = ctrvl_aum['aum'] - ctrvl_aum.iloc[:, 1:].sum(axis=1)

        portfolio_volatility = (ctrvl_pos.ffill().fillna(0).sum(axis=1) + aum_net).pct_change().std() * np.sqrt(252)
    
    return portfolio_volatility



def mc_fcast_ptf_volatility(df,
                                incipit_date,
                                sigma_sl,
                                sigma_tp,
                                max_trade_duration, 
                                number_of_trades,
                                aum, 
                                aum_lost_sl,
                                iterations=100):
    """
    
    """
    

    simul_ptf_vol = []
    for _ in tqdm.tqdm(range(iterations)):
        temp_vol = forecast_portfolio_volatility(df,
                                incipit_date,
                                sigma_sl,
                                sigma_tp,
                                max_trade_duration, 
                                number_of_trades,
                                aum, 
                                aum_lost_sl)
        
        simul_ptf_vol.append(temp_vol)
    
    print(f"Estimated Portfolio Volatility: {round(np.mean(simul_ptf_vol), 3)} \u00B1 {round(np.std(simul_ptf_vol), 3)}")
