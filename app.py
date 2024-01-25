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
yf.pdr_override()

st.set_page_config(page_title="ASAM Tracker", layout="wide")

st.title("ASAM Tracker")

def main():

    # File Upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

    if uploaded_file is not None:
        df_upload = pd.read_excel(uploaded_file, sheet_name='Summary')

        #Chart
        df = df_upload.iloc[:11, :5]
        df.columns = df.iloc[0]
        df = df[1:]
        df['Dates'] = pd.to_datetime(df['Dates'])

        # Plot each group with a unique color
        plt.plot(df['Dates'], df['Group 1'], label='Group 1', marker='o')
        plt.plot(df['Dates'], df['Group 2'], label='Group 2', marker='o')
        plt.plot(df['Dates'], df['Group 3'], label='Group 3', marker='o')
        plt.plot(df['Dates'], df['Group 4'], label='Group 4', marker='o')

        # Customize the plot
        plt.title('Total Simple Return')
        plt.xlabel('Dates')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        # Show the plot
        st.pyplot(plt.gcf())

        #Summary Table
        tb_upload = pd.read_excel(uploaded_file, sheet_name='Summary', header=1)
        tb = tb_upload.iloc[10:19, 0:5]
        tb = tb.reset_index(drop=True)
        tb = tb.rename(columns={'Dates':'Metrics'})

        columns_to_convert = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
        tb[columns_to_convert] = tb[columns_to_convert] * 100
        tb[columns_to_convert] = tb[columns_to_convert].round(2).astype(str) + '%'

        st.dataframe(tb)

    stock_symbol = "AAPL"

    #Change to January of current year
    data = yf.download(stock_symbol, period='2d')

    most_recent_trading_day = data.index[0]

    #Change dynamic values
    analysis_start_date = pd.to_datetime('2023-12-29', format='%Y-%m-%d')#the day first stock was bought 
    analysis_end_date = most_recent_trading_day #should be a trading day
    analysis_end_date_plusone = most_recent_trading_day + pd.Timedelta(days=1) #just add one to the above mentioned date, need not be a working day
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
    transactions = transactions.drop('Date', axis=1)
    for i in teams:
        a = pd.DataFrame(transactions.groupby(['Group','Security']).sum().loc[i]['QuantXAction'])
        #a['Group'] = i
        #a = a.reset_index()
        #a = a[['Group','Security','QuantXAction']]
        #a = a[a.Security!='Cash']
        #postions_calc= postions_calc.append(a)
    
    """
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
    daily_data_transpose = daily_data.transpose().reset_index().rename(columns={'index':'Tickers'})

    postions_calc =postions_calc.merge(daily_data_transpose,left_on='Tickers',right_on='Tickers', how ='left')

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
    display_Total_Value = display_Total_Value.drop('Shares', axis=1)
    #display_Total_Value.set_index('Group', inplace=True)
    ax = display_Total_Value.T.plot(kind='line', marker='o')

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Portfolio Value')
    ax.legend(title='Group', loc='lower right', bbox_to_anchor=(1.3, 0.2))

    st.pyplot(plt.gcf())
    """

if __name__ == "__main__":
    main()
