import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import math
import yfinance as yf
import datetime
from pandas_datareader import data as pdr
import warnings
import statsmodels.api as sm
import plotly.graph_objects as go
yf.pdr_override()


st.set_page_config(page_title="ASAM Tracker", layout="wide")

st.title("ASAM Tracker")

def main():

    stock_symbol = "AAPL"

    #Change to January of current year
    data = yf.download(stock_symbol, period='2d')

    most_recent_trading_day = data.index[0]

    #Change dynamic values
    analysis_start_date = pd.to_datetime('2023-12-29', format='%Y-%m-%d').date() #the day first stock was bought 
    analysis_end_date = most_recent_trading_day.date() #should be a trading day
    analysis_end_date_plusone = (most_recent_trading_day + pd.Timedelta(days=1)).date() #just add one to the above mentioned date, need not be a working day
    rf = 0.01/100*252 #riskfree-- just add current daily percent before the slash
    bench_list = ['SP500','DJI','Nasdaq','Russell']
    prtfolio = bench_list[3] #switch no. to change benchmark, Ex: selecting 3 gives you the Russell 2000
    outputFolder = 'C:\\Users\\lpate\\Documents\\ASAM\\ASAM_2024_Team1\\'
    #input_file_name= pd.read_excel(outputFolder + 'ASAM_23-24_Tracker.xlsx')
    output_file_name='ASAM_Excel_Out.xlsx'

    transactions = pd.read_excel('ASAM_23-24_Tracker.xlsx', sheet_name ='Transactions',header=1) #Make sure to create a sheet with name Transactions 

    teams = transactions.Group.unique() #storing names of teams 

    transactions['Action_Postion'] = [-1 if x == 'Sell' else 1 for x in transactions['Action']] #sorting selling and buying actions

    transactions['ToalXAction_Postion'] = transactions.Total * transactions.Action_Postion #Transaction value X Action

    transactions['QuantXAction'] = transactions.Quantity * transactions.Action_Postion #Quantity X Action

    postions_calc = pd.DataFrame()
    transactionsNoDate = transactions.drop('Date', axis=1)
    for i in teams:
        a = pd.DataFrame(transactionsNoDate.groupby(['Group','Security']).sum().loc[i]['QuantXAction'])
        a['Group'] = i
        a = a.reset_index()
        a = a[['Group','Security','QuantXAction']]
        a = a[a.Security!='Cash']
        postions_calc = pd.concat([postions_calc, a], ignore_index=True)
    

    transactions_filter_buy= transactions[transactions.Action=='Buy'][['Group','Security','Price']]

    postions_calc = postions_calc.merge(transactions_filter_buy,how='right')[['Group','Security','QuantXAction','Price']]

    postions_calc= postions_calc.rename(columns= {'Security':'Tickers','QuantXAction':'Shares','Price':'Purchase'})
    postions_calc = postions_calc[postions_calc.Shares>0]

    postions_calc['Cost'] = postions_calc['Shares']* postions_calc['Purchase']
    
    total_cost = postions_calc.groupby(['Group']).sum()
    ASAM_Total_cash= total_cost.Cost.sum()
    total_cost = pd.DataFrame(total_cost['Cost'])
    total_cost.loc['ASAM'] =total_cost.sum()

    tickers_list = postions_calc.Tickers.unique().tolist()

    #Fetch the data
    daily_data = yf.download(tickers_list , start=analysis_end_date ,period= '1d' ,end= analysis_end_date_plusone )['Close'].dropna(axis=0,how='all')
    daily_data_transpose = daily_data.transpose().reset_index().rename(columns={'Ticker':'Tickers'})
    
    postions_calc = postions_calc.merge(daily_data_transpose, left_on='Tickers', right_on='Tickers', how ='left')

    postions_calc.columns = ['Group','Tickers','Shares','Purchase','Cost','Price']

    postions_calc['Value'] = postions_calc['Shares']*postions_calc['Price']
    postions_calc['Gain $'] = postions_calc['Value'] - postions_calc['Cost']
    postions_calc['Gain %'] = postions_calc['Gain $']/postions_calc['Cost']

    
    data_main = pdr.get_data_yahoo(tickers_list, start=analysis_start_date, end=analysis_end_date_plusone).dropna(axis=0,how='all') #switch this date for different cohorts
    data = data_main['Close'].T
    data_adjusted = data_main['Adj Close'].T
    ##Loading all stock data gonna take 3 mins to run 
    
    data = data.T[data.columns>=transactions[transactions.Action=='Buy'].Date.max()].T
    data_adjusted = data_adjusted.T[data_adjusted.columns>=transactions[transactions.Action=='Buy'].Date.max()].T

    history = postions_calc[['Group','Tickers']].copy()
    history = history.merge(data,how='left',left_on='Tickers',right_index=True)

    Total_Value = postions_calc[['Group','Tickers','Shares']].copy()
    Total_Value = Total_Value.merge(data,how='left',left_on='Tickers',right_index=True)
    Total_Value_sliced = Total_Value.iloc[:,3:]
    Total_Value_sliced  = Total_Value_sliced.T*Total_Value.Shares
    Total_Value_sliced  = Total_Value_sliced.T
    Total_Value.iloc[:,3:] = Total_Value_sliced

    #Graph total value
    display_Total_Value = Total_Value.groupby('Group').sum()
    display_Total_Value = display_Total_Value.drop(['Shares','Tickers'], axis=1)
    temp_value = display_Total_Value.T
    fig1 = go.Figure()
    for column in temp_value.columns:
        fig1.add_trace(go.Scatter(x=temp_value.index, y=temp_value[column], mode='lines+markers', name=column))
    
    fig1.update_layout(title='Portfolio Value', xaxis_title='Date', yaxis_title='Value ($)', width=1200, height=800)
    st.plotly_chart(fig1)

    # Apply pct_change() along the rows (now representing dates)
    daily_returns = Total_Value.drop(['Group','Shares'], axis=1)
    daily_returns = daily_returns.set_index('Tickers', inplace=False)
    daily_returns = daily_returns.T.pct_change()
    
    daily_returns_T = daily_returns.drop(daily_returns.index[0]).T
    daily_returns_T['Total_Ret'] = daily_returns_T.apply(lambda row: (1 + row).prod() - 1, axis=1)

    pos_cols = postions_calc[['Group','Tickers','Shares']].copy()
    daily_returns = pos_cols.merge(daily_returns_T,how='left',left_on=['Tickers'],right_index=True)
    daily_returns.drop_duplicates()

    tot_transactions = transactions.drop(['Date'], axis=1).groupby('Group').sum()['Total']

    merged_df = pd.merge(left=tot_transactions, right=display_Total_Value, how='left', on=['Group'])

    result_df = pd.DataFrame(columns=merged_df.columns)
    merged_df.reset_index(drop=False, inplace=True)

    for index, row in merged_df.iterrows():
        new_row = row.copy()
        new_row[1] = row[1]
        for i in range(2, len(row)):
            new_row[i] = ((row[i] / row[i - 1])-1)*100
        result_df = pd.concat([result_df, new_row.to_frame().T], ignore_index=True)

    simple_return = result_df.drop(columns=['Total'])
    simple_return.set_index('Group', inplace=True)

    #Graph simple return
    temp_return = simple_return.T
    fig2 = go.Figure()
    for column in temp_return.columns:
        fig2.add_trace(go.Scatter(x=temp_return.index, y=temp_return[column], mode='lines+markers', name=column))
    
    fig2.update_layout(title='Simple Return', xaxis_title='Date', yaxis_title='Return (%)', width=1200, height=800)
    st.plotly_chart(fig2)

    treasury_bill = yf.Ticker('^IRX')
    historical_treasury = treasury_bill.history(start=analysis_start_date, end=analysis_end_date_plusone)
    risk_free_rate = historical_treasury['Close']
    risk_free_rate.index = risk_free_rate.index.tz_localize(None)
    risk_free_rate = pd.DataFrame(risk_free_rate)
    risk_free_rate['Close'] = risk_free_rate['Close']/252

    return_rf = simple_return.T.merge(pd.DataFrame(risk_free_rate), how='left', left_index=True, right_index=True)
    for group_col in simple_return.T.columns:
        return_rf[group_col] = return_rf[group_col] - return_rf['Close']
    
    return_rf = return_rf.drop(columns=['Close'])

    avg_exc_ret = return_rf.mean()
    avg_exc_ret.name = 'Excess Average Return'
    avg_exc_ret = pd.DataFrame(avg_exc_ret)

    avg_simple_ret = simple_return.T.mean()
    avg_simple_ret.name = 'Daily Average Return'

    daily_vol = simple_return.T.std()
    daily_vol.name = 'Volatility'
    daily_vol = pd.DataFrame(daily_vol)

    port_stats = avg_exc_ret.merge(avg_simple_ret, how='left', left_index=True, right_index=True)
    final_port_stats = port_stats.merge(daily_vol, how='left', left_index=True, right_index=True)
    final_port_stats['Sharpe Ratio'] = final_port_stats['Excess Average Return']/final_port_stats['Volatility']

    st.write(final_port_stats)

    display_port_stats = final_port_stats.T

    with st.beta_container():

        for group_col in display_port_stats.columns:

            group_data = display_port_stats[group_col]

            with st.beta_container():
                st.write(f"## {group_col}")
                st.write(group_data)

    
if __name__ == "__main__":
    main()
