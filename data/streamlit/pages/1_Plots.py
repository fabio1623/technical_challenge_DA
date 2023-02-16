import streamlit as st
import pandas as pd
import altair as alt

def display_chart(dataframe, selected_feature, selected_gas):
    dataframe = dataframe[dataframe['gas_type'].isin(selected_gas)]
    st.bar_chart(dataframe, x='gas_type', y=selected_feature)


data = pd.read_csv('../cleaned_data.csv')

selected_gas = st.multiselect("Select Gas Type", data['gas_type'].unique())

filtered_columns = [col for col in data.columns if col != 'gas_type']
selected_feature = st.selectbox("Select a column to plot (y-axis)", filtered_columns)

if selected_gas:
        display_chart(data, selected_feature, selected_gas)