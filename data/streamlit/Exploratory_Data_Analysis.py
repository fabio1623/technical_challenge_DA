import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display_data(df):
    st.write(df.shape)
    st.dataframe(df, use_container_width=True)

def display_heatmap(dataframe):
    with st.spinner('Loading...'):
            corr=dataframe.corr()

            mask=np.triu(np.ones_like(corr, dtype=bool))     # generate a mask for the upper triangle

            f, ax=plt.subplots(figsize=(11, 9))                 # set up the matplotlib figure

            cmap=sns.diverging_palette(220, 10, as_cmap=True)   # generate a custom diverging colormap

            sns.heatmap(corr, mask=mask, cmap=cmap,             # draw the heatmap with the mask and correct aspect ratio
                        vmax=.3, center=0, square=True,
                        linewidths=.5, cbar_kws={"shrink": .5})
            st.write(f)


data_visualization = pd.read_csv('../cleaned_data_visualization.csv')
data_modeling = pd.read_csv('../cleaned_data_modeling.csv')

st.title('cleaned_data.csv')

st.subheader('Dataframe - cleaned_data_visualization.csv')
display_data(data_visualization)

st.subheader('Heatmap - cleaned_data_modeling.csv')
display_heatmap(data_modeling)