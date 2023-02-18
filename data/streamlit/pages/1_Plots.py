import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


data = pd.read_csv('../cleaned_data_visualization.csv')

tab1, tab2 = st.tabs(['Fuel Consumption by Feature', 'Average Fuel Consumption by Feature'])

with tab1:
    x_features1 = ['distance', 'speed', 'temp_inside', 'temp_outside']

    selected_feature1 = st.selectbox("Select a feature for the x-axis:", x_features1)

    # create the Streamlit app
    st.title(f'Fuel Consumption by {selected_feature1.capitalize()} and Gas Type')
    st.write(f'This app displays a line chart of fuel consumption by {selected_feature1.capitalize()} and gas type.')

    with st.spinner('Loading...'):
        # plot the line chart
        fig, ax = plt.subplots()
        sns.lineplot(data=data, x=selected_feature1, y='consume', hue='gas_type', ax=ax)

        # add labels and title
        ax.set_xlabel(f'{selected_feature1.capitalize()}')
        ax.set_ylabel('Fuel Consumption (L/100km)')

        # display the chart in the Streamlit app
        st.pyplot(fig)

with tab2:
    x_features2 = ['ac', 'rain', 'sun', 'snow']

    selected_feature2 = st.selectbox("Select a feature for the x-axis:", x_features2)

    # create the Streamlit app
    st.title(f'Average Fuel Consumption by {selected_feature2.capitalize()} and Gas Type')
    st.write(f'This app displays a bar chart of fuel consumption by {selected_feature2.capitalize()} and gas type.')

    with st.spinner('Loading...'):
        # plot the bar chart
        fig, ax = plt.subplots()
        sns.barplot(data=data, x=selected_feature2, y='consume', hue='gas_type', ci=None, ax=ax)

        # add labels and title
        ax.set_xlabel(selected_feature2.capitalize())
        ax.set_ylabel('Average Fuel Consumption (L/100km)')

        # display the chart in the Streamlit app
        st.pyplot(fig)