import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl


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
        plt.show()

        #Summary Table
        tb_upload = pd.read_excel(uploaded_file, sheet_name='Summary', header=1)
        tb = tb_upload.iloc[10:19, 0:5]
        tb = tb.reset_index(drop=True)
        tb = tb.rename(columns={'Dates':'Metrics'})

        columns_to_convert = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
        tb[columns_to_convert] = tb[columns_to_convert] * 100
        tb[columns_to_convert] = tb[columns_to_convert].round(2).astype(str) + '%'

        st.dataframe(tb)


if __name__ == "__main__":
    main()
