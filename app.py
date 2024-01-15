import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="ASAM Tracker", layout="wide")

st.title("ASAM Tracker")

def main():

    # File Upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

    if uploaded_file is not None:
        # Load Excel file into DataFrame
        df = pd.read_excel(uploaded_file)

        # Display DataFrame
        st.write("Uploaded Data:")
        st.write(df)

        # Create Line Chart
        line_chart = px.line(df, x=df.columns[0], y=df.columns[1], title="Line Chart")
        
        # Display Line Chart
        st.plotly_chart(line_chart)

if __name__ == "__main__":
    main()
