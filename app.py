import streamlit as st
import pandas as pd
#import plotly.express as px


st.set_page_config(page_title="ASAM Tracker", layout="wide")

st.title("ASAM Tracker")

def main():

    # File Upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

    if uploaded_file is not None:

if __name__ == "__main__":
    main()
