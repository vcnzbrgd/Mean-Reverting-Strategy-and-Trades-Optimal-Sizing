import pandas as pd
import numpy as np
import tqdm
import random


# as convention 1 week is 5 trading days and 1 month is 21 trading days so that each year has 252 trading days

def portfolio_nav(trades_pnl, aum = 100000):
    """
    Computes the nav of the portfolio with a certain level of AuM from a dictionary of trades

    Parameters:
    - trades_pnl (dict): dictionary with unique keys for each trade and as values a pd.Dataframe with dates as index,
    and columns: first column must be the price of the instrument, direction equal to -1/1, quantity and entry_price.

    Returns:
    - nav (pd.Series): series of the portfolio NAV
    """


    ctrvl_aum = []
    ctrvl_pos = []

    for tr in trades_pnl.keys():
        direction = trades_pnl[tr]['direction'].unique()[0]
        ctrvl_aum.append((trades_pnl[tr]['entry_price'] * trades_pnl[tr]['quantity'] * direction).rename(tr).to_frame())
        
        if direction == 1:
            trades_pnl[tr]['pnl'] = trades_pnl[tr].iloc[:,0] * trades_pnl[tr]['quantity'] * direction
            ctrvl_pos.append((trades_pnl[tr]['pnl']).rename(tr).to_frame())
        elif direction == -1:
            # adj. pnl for short, reverse pnl
            trades_pnl[tr].iloc[:,0] = (trades_pnl[tr].iloc[:,0] - trades_pnl[tr].iloc[0,0]) * (-2) + trades_pnl[tr].iloc[:,0]
            trades_pnl[tr]['pnl'] = trades_pnl[tr].iloc[:,0] * trades_pnl[tr]['quantity'] * direction

            ctrvl_pos.append((trades_pnl[tr]['pnl']).rename(tr).to_frame())

    ctrvl_aum = pd.concat(ctrvl_aum, axis=1).sort_index().ffill().fillna(0)
    ctrvl_pos = pd.concat(ctrvl_pos, axis=1).sort_index().ffill().fillna(0)

    nav = (aum - ctrvl_aum.sum(axis=1) + ctrvl_pos.sum(axis=1))
    
    return nav



def optimal_number_trades(df,
                            incipit_date,
                            sigma_sl,
                            sigma_tp,
                            max_trade_duration, 
                            ptf_vol_tgt,
                            aum_lost_sl):
    """
    Computes the optimal number of trades, given some parameters and simulating random trades based on those parameters

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - ptf_vol_tgt (float): target volatility of the portfolio
    - aum_lost_sl (float): share of AuM lost from each trade when stopped

    Returns:
    - portfolio_volatility (float): volatility of the portfolio from random trades
    - number_of_trades (float): number of trades)
    """

    # set a generic value of AUM for the portfolio
    aum = 1000000

    if (incipit_date <= '2014-02-12') | (incipit_date >= '2023-02-12'):
        raise ValueError('incipit_date must be between 2014-02-12 and 2023-02-12')
    
    if (sigma_sl >= 0) | (sigma_tp <= 0):
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    # refill Sat&Sun
    refilled = df.reindex(pd.date_range(df.index.min(), df.index.max()), method='ffill')

    # create list of dates for simulated trades
    date_index_reference = df.loc[pd.to_datetime(incipit_date):pd.to_datetime(incipit_date)+pd.DateOffset(years=1)].index
    startdates = list(date_index_reference.astype(str))

    # create df to store results of the trades
    trades_df = pd.DataFrame()

    # initialization of portfolio volatility
    portfolio_volatility = 0
    trades_pnl = {}

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
        
        # compute portfolio NAV
        trades_pnl[f'{trade_entry_date}#{security_id}'] = trade_pnl

        ptf_nav = portfolio_nav(trades_pnl, aum)

        ptf_nav = ptf_nav.reindex(pd.date_range(incipit_date, pd.to_datetime(incipit_date)+pd.DateOffset(years=1), freq='B')).ffill().fillna(aum)

        portfolio_volatility = ptf_nav.pct_change().std() * np.sqrt(252)
        
    number_of_trades = len(trades_df) 

    return portfolio_volatility, number_of_trades



def mc_opt_n_trades(df,
                    incipit_date,
                    sigma_sl,
                    sigma_tp,
                    max_trade_duration, 
                    ptf_vol_tgt,
                    aum_lost_sl,
                    iterations=100):
    """
    Applies the function optimal_number_trades a number of times (Monte Carlo) specified by parameter iterations,
    and prints the optimal number of trades to achieve the target portfolio volatility.

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - ptf_vol_tgt (float): target volatility of the portfolio
    - aum_lost_sl (float): share of AuM lost from each trade when stopped
    - iterations (int): number of iterations for the simulation
    """
    
    if (sigma_sl >= 0) | (sigma_tp <= 0):
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    prob_res = {}
    for _ in tqdm.tqdm(range(iterations)):
        portfolio_volatility, number_of_trades = optimal_number_trades(df,
                                incipit_date,
                                sigma_sl,
                                sigma_tp,
                                max_trade_duration, 
                                ptf_vol_tgt,
                                aum_lost_sl)
        
        prob_res[portfolio_volatility] = number_of_trades

    prob_res = pd.DataFrame(prob_res, index=['number_of_trades']).T.reset_index().rename({'index':'ptf_volat'}, axis=1)
    print(f"Optimal Number of Trades: {round(prob_res['number_of_trades'].mean())} \u00B1 {round(prob_res['number_of_trades'].std())} [with avg. portfolio volatility of {round(prob_res['ptf_volat'].mean(), 4)*100}%]")



def forecast_portfolio_volatility(df,
                                    incipit_date,
                                    sigma_sl,
                                    sigma_tp,
                                    max_trade_duration, 
                                    number_of_trades,
                                    aum_lost_sl):
    """
    Forecasts the portfolio volatility, given some parameters and simulating random trades based on those parameters

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - aum_lost_sl (float): share of AuM lost from each trade when stopped

    Returns:
    - portfolio_volatility (float): volatility of the portfolio from random trades
    """

    # set a generic value of AUM for the portfolio
    aum = 1000000

    if (sigma_sl >= 0) | (sigma_tp <= 0):
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    if (incipit_date <= '2014-02-12') | (incipit_date >= '2023-02-12'):
        raise ValueError('incipit_date must be between 2014-02-12 and 2023-02-12')

    # refill Sat&Sun
    refilled = df.reindex(pd.date_range(df.index.min(), df.index.max()), method='ffill')

    # create list of dates for simulated trades
    date_index_reference = df.loc[pd.to_datetime(incipit_date):pd.to_datetime(incipit_date)+pd.DateOffset(years=1)].index
    startdates = list(date_index_reference.astype(str))

    # create df to store results of the trades
    trades_df = pd.DataFrame()
    trades_pnl = {}

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
        
        # compute portfolio NAV
        trades_pnl[f'{trade_entry_date}#{security_id}'] = trade_pnl

        ptf_nav = portfolio_nav(trades_pnl, aum)

        ptf_nav = ptf_nav.reindex(pd.date_range(incipit_date, pd.to_datetime(incipit_date)+pd.DateOffset(years=1), freq='B')).ffill().fillna(aum)

        portfolio_volatility = ptf_nav.pct_change().std() * np.sqrt(252)
    
    return portfolio_volatility



def mc_fcast_ptf_volatility(df,
                                incipit_date,
                                sigma_sl,
                                sigma_tp,
                                max_trade_duration, 
                                number_of_trades,
                                aum_lost_sl,
                                iterations=100):
    """
    Applies the function forecast_portfolio_volatility a number of times (Monte Carlo) specified by parameter iterations,
    and prints the forecasted portfolio volatility resulting from the trades.

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - number_of_trades (float): number of simulated trades for the portfolio
    - aum_lost_sl (float): share of AuM lost from each trade when stopped
    - iterations (int): number of iterations for the simulation
    """
    
    if (sigma_sl >= 0) | (sigma_tp <= 0):
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    simul_ptf_vol = []
    for _ in tqdm.tqdm(range(iterations)):
        temp_vol = forecast_portfolio_volatility(df,
                                incipit_date,
                                sigma_sl,
                                sigma_tp,
                                max_trade_duration, 
                                number_of_trades,
                                aum_lost_sl)
        
        simul_ptf_vol.append(temp_vol)
    
    print(f"Estimated Portfolio Volatility: {round(np.mean(simul_ptf_vol), 4)*100}% \u00B1 {round(np.std(simul_ptf_vol), 4)*100}%")









##########################################################################










def percentage_aum_lost(df,
                            incipit_date,
                            sigma_sl,
                            sigma_tp,
                            max_trade_duration, 
                            ptf_vol_tgt,
                            number_of_trades):
    """
    Computes the % AuM lost from each trade, given some parameters and simulating random trades based on those parameters

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - ptf_vol_tgt (float): target volatility of the portfolio
    - number_of_trades (float): number of simulated trades for the portfolio

    Returns:
    - aum_lost_sl (float): share of AuM lost from each trade
    """

    # set a generic value of AUM for the portfolio
    aum = 1000000

    if (sigma_sl >= 0) | (sigma_tp <= 0):
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    if (incipit_date <= '2014-02-12') | (incipit_date >= '2023-02-12'):
        raise ValueError('incipit_date must be between 2014-02-12 and 2023-02-12')

    # refill Sat&Sun
    refilled = df.reindex(pd.date_range(df.index.min(), df.index.max()), method='ffill')

    # create list of dates for simulated trades
    date_index_reference = df.loc[pd.to_datetime(incipit_date):pd.to_datetime(incipit_date)+pd.DateOffset(years=1)].index
    startdates = list(date_index_reference.astype(str))

    # initialization of portfolio volatility and percentage aum lost
    portfolio_volatility = 0
    aum_lost_sl = 0

    # generate random trades until portfolio volatility is below target
    while portfolio_volatility < ptf_vol_tgt:

        # create df to store results of the trades
        trades_df = pd.DataFrame()
        ctrvl_aum = pd.DataFrame(index = date_index_reference)
        ctrvl_aum['aum'] = aum
        ctrvl_pos = pd.DataFrame(index = date_index_reference)

        aum_lost_sl += 0.001

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

    
    return aum_lost_sl



def mc_perc_aum_lost(df,
                        incipit_date,
                        sigma_sl,
                        sigma_tp,
                        max_trade_duration, 
                        ptf_vol_tgt,
                        number_of_trades,
                        iterations=100):
    """
    Applies the function percentage_aum_lost a number of times (Monte Carlo) specified by parameter iterations, and prints the 
    avg and stdev share of AuM lost from each trade.

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - ptf_vol_tgt (float): target volatility of the portfolio
    - number_of_trades (float): number of simulated trades for the portfolio
    - iterations (int): number of iterations for the simulation
    """

    if (sigma_sl >= 0) | (sigma_tp <= 0):
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    simul_perc_aum_lost = []
    for _ in tqdm.tqdm(range(iterations)):
        temp_percaum = percentage_aum_lost(df,
                        incipit_date,
                        sigma_sl,
                        sigma_tp,
                        max_trade_duration, 
                        ptf_vol_tgt,
                        number_of_trades)
        
        simul_perc_aum_lost.append(temp_percaum)
    
    print(f"Estimated %AUM lost per trade: {round(np.mean(simul_perc_aum_lost), 4)*100}% \u00B1 {round(np.std(simul_perc_aum_lost), 4)*100}%")



def sigma_stoploss(df,
                    incipit_date,
                    sigma_tp,
                    max_trade_duration, 
                    ptf_vol_tgt,
                    number_of_trades,
                    aum_lost_sl):
    """
    Computes the optimal sigma for stoploss from each trade, given some parameters and simulating random trades based on those parameters

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - ptf_vol_tgt (float): target volatility of the portfolio
    - number_of_trades (float): number of simulated trades for the portfolio
    - aum_lost_sl (float): share of AuM lost from each trade when stopped

    Returns:
    - sigma_sl (float): optimal sigma for stoploss
    - portfolio_volatility (float): volatility of the portfolio from random trades
    """

    if sigma_tp <= 0:
        raise ValueError('sigma_tp must be positive')
    
    # set a generic value of AUM for the portfolio
    aum = 1000000

    if (incipit_date <= '2014-02-12') | (incipit_date >= '2023-02-12'):
        raise ValueError('incipit_date must be between 2014-02-12 and 2023-02-12')

    # refill Sat&Sun
    refilled = df.reindex(pd.date_range(df.index.min(), df.index.max()), method='ffill')

    # create list of dates for simulated trades
    date_index_reference = df.loc[pd.to_datetime(incipit_date):pd.to_datetime(incipit_date)+pd.DateOffset(years=1)].index
    startdates = list(date_index_reference.astype(str))

    # initialization of portfolio volatility and percentage aum lost
    portfolio_volatility = np.inf
    sigma_sl = 0

    # generate random trades until portfolio volatility is below target
    while (np.abs(portfolio_volatility - ptf_vol_tgt) > 0.001):

        # create df to store results of the trades
        trades_df = pd.DataFrame()
        ctrvl_aum = pd.DataFrame(index = date_index_reference)
        ctrvl_aum['aum'] = aum
        ctrvl_pos = pd.DataFrame(index = date_index_reference)

        if portfolio_volatility - ptf_vol_tgt > 0:
            sigma_sl -= 0.1
        else:
            sigma_sl += 0.1

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
    
    return sigma_sl, portfolio_volatility



def mc_sigma_stoploss(df,
                    incipit_date,
                    sigma_tp,
                    max_trade_duration, 
                    ptf_vol_tgt,
                    number_of_trades,
                    aum_lost_sl,
                    iterations=100):
    """
    Applies the function sigma_stoploss a number of times (Monte Carlo) specified by parameter iterations,
    and prints the optimal sigma for stoploss to achieve the target portfolio volatility.

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - ptf_vol_tgt (float): target volatility of the portfolio
    - number_of_trades (float): number of simulated trades for the portfolio
    - aum_lost_sl (float): share of AuM lost from each trade when stopped
    - iterations (int): number of iterations for the simulation
    """
    
    if sigma_tp <= 0:
        raise ValueError('sigma_sl must be negative and sigma_tp positive')

    prob_res = {}
    for _ in tqdm.tqdm(range(iterations)):
        sigma_sl, portfolio_volatility = sigma_stoploss(df,
                                                        incipit_date,
                                                        sigma_tp,
                                                        max_trade_duration, 
                                                        ptf_vol_tgt,
                                                        number_of_trades,
                                                        aum_lost_sl)
        
        prob_res[portfolio_volatility] = sigma_sl

    prob_res = pd.DataFrame(prob_res, index=['sigma']).T.reset_index().rename({'index':'ptf_volat'}, axis=1)
    print(f"Optimal Sigma for SL: {round(prob_res['sigma'].mean())} \u00B1 {round(prob_res['sigma'].std())} [with avg. portfolio volatility of {round(prob_res['ptf_volat'].mean(), 4)*100}%]")



def optimal_sizing(df,
                    incipit_date,
                    number_of_trades = None,
                    ptf_vol_tgt = None,
                    aum_lost_sl = None,
                    sigma_sl = None,
                    sigma_tp = 1.5,
                    max_trade_duration = 63,
                    iterations = 100):
    
    """
    Calculates the optimal one among number_of_trades, ptf_vol_tgt, aum_lost_sl, sigma_sl, given the other variables.

    Parameters:
    - df (pandas.DataFrame): dataset with prices, each column is an instrument
    - incipit_date (str): string with date in format yyyy-mm-dd, it is the date at which we start the simulation. Ideally it is
    the date when the algorithm is run and random trades are generated from incipit_date to incipit_date + 1Y
    - number_of_trades (float): number of simulated trades for the portfolio
    - ptf_vol_tgt (float): target volatility of the portfolio
    - aum_lost_sl (float): share of AuM lost from each trade when stopped
    - sigma_sl (float): sigma at which SL is triggered. Must be a negative number
    - sigma_tp (float): sigma at which TP is triggered. Must be a positive number
    - max_trade_duration (float): max lenght of a trade expressed in trading days
    - iterations (int): number of iterations for the simulation
    """
    if [number_of_trades, ptf_vol_tgt, aum_lost_sl, sigma_sl].count(None) != 1:
        raise Exception('Only one among: [number_of_trades, ptf_vol_tgt, aum_lost_sl, sigma_sl] must be None')

    if number_of_trades == None:
        mc_opt_n_trades(df,
                        incipit_date = incipit_date,
                        sigma_sl = sigma_sl,
                        sigma_tp = sigma_tp,
                        max_trade_duration = max_trade_duration,
                        ptf_vol_tgt = ptf_vol_tgt,
                        aum_lost_sl = aum_lost_sl,
                        iterations = iterations)
        
    elif ptf_vol_tgt == None:
        mc_fcast_ptf_volatility(df,
                        incipit_date = incipit_date,
                        sigma_sl = sigma_sl,
                        sigma_tp = sigma_tp,
                        max_trade_duration = max_trade_duration,
                        number_of_trades = number_of_trades,
                        aum_lost_sl = aum_lost_sl,
                        iterations = iterations)
        
    elif aum_lost_sl == None:
        mc_perc_aum_lost(df,
                        incipit_date = incipit_date,
                        sigma_sl = sigma_sl,
                        sigma_tp = sigma_tp,
                        max_trade_duration = max_trade_duration,
                        ptf_vol_tgt = ptf_vol_tgt,
                        number_of_trades = number_of_trades,
                        iterations = iterations)
        
    elif sigma_sl ==None:
        mc_sigma_stoploss(df,
                    incipit_date = incipit_date,
                    sigma_tp = sigma_tp,
                    max_trade_duration = max_trade_duration,
                    ptf_vol_tgt = ptf_vol_tgt,
                    number_of_trades = number_of_trades,
                    aum_lost_sl = aum_lost_sl,
                    iterations = iterations)