import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


def display_data_and_descriptions(df, title):
    st.subheader(title)

    col1, col2 = st.columns(2)
    with col1:
        description = {
            'gas_type': {0: 'E10', 1: 'SP98'},
            'snow': {0: 'False', 1: 'True', -1: 'Unknown'}
        }
        st.write('Column definitions:')
        st.write(description)
    with col2:
        st.write('Shape:')
        st.write(df.shape)
        
    st.dataframe(df, use_container_width=True)


def compare_and_fit_models(models_with_name_list, X, y):
    fitted_models_with_scaler_list = []

    r2_list = []
    mse_list = []
    rmse_list = []
    mae_list = []

    latest_iteration = st.empty()
    bar = st.progress(0)

    for idx, model_with_name in zip(range(1, len(models_with_name_list)+1), models_with_name_list):
        latest_iteration.text(f'{idx} / {len(models_with_name_list)} models tested')
        bar.progress(round(100 / len(models_with_name_list)) * idx)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Scaling data = X_train
        min_max_scaler = MinMaxScaler().fit(X_train)
        X_train_normalized = min_max_scaler.transform(X_train)
        X_train_normalized = pd.DataFrame(X_train_normalized)

        # Scaling data = X_test
        X_test_normalized = min_max_scaler.transform(X_test)
        X_test_normalized = pd.DataFrame(X_test_normalized)

        model_with_name['model'].fit(X_train_normalized, y_train)

        # Make predictions on the test data
        y_pred = model_with_name['model'].predict(X_test_normalized)

        # R2 validation
        r2 = r2_score(y_test, y_pred)

        # MSE validation
        mse=mean_squared_error(y_test, y_pred)

        # RMSE validation
        rmse = np.sqrt(mse)

        # MAE validation
        mae=mean_absolute_error(y_test, y_pred)

        fitted_models_with_scaler_list.append({
            'model': model_with_name['model'],
            'min_max_scaler' : min_max_scaler
        })

        r2_list.append(r2)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)


    summary = {
        'Algorithm': [model_with_name['name'] for model_with_name in models_with_name_list],
        'R2': r2_list,
        'MSE': mse_list,
        'RMSE': rmse_list,
        'MAE': mae_list
    }
    summary = pd.DataFrame(summary)

    return {
        'summary' : summary,
        'models_with_scalers' : fitted_models_with_scaler_list
    }

def save_models_with_scaler(selected_algorithm_names, models_with_scaler_list):
    for algorithm_name, model_with_scaler in zip(selected_algorithm_names, models_with_scaler_list):
        algorithm_name = algorithm_name.lower().replace(' ', '_')
        with open(f'../models/regression/{algorithm_name}.pkl', 'wb') as file:
            pickle.dump(model_with_scaler, file)


def display_algorithms_comparison(df):
    df = df.sample(frac=1)

    X = df.drop(['consume'], axis=1)
    y = df['consume']

    algorithms = [
        {
            "name": "Linear Regression",
            "model": LinearRegression()
        },
        {
            "name": "Ridge Regression",
            "model": Ridge()
        },
        {
            "name": "Lasso Regression",
            "model": Lasso()
        },
        {
            "name": "Decision Tree Regressor",
            "model": DecisionTreeRegressor()
        },
        {
            "name": "Elastic Net",
            "model": ElasticNet()
        },
        {
            "name": "XGB Regressor",
            "model": XGBRegressor()
        },
        {
            "name": "LGBM Regressor",
            "model": LGBMRegressor()
        },
        {
            "name": "KNeighbors",
            "model": KNeighborsRegressor()
        },
        {
            "name": "MLP Regressor",
            "model": MLPRegressor()
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor()
        }
    ]

    with st.form("model_generator_form"):
        selected_algorithm_names = st.multiselect("Algorithms To Compare", [obj['name'] for obj in algorithms])

        submitted = st.form_submit_button("Compare")
        if submitted:
            models_with_name_list = [obj for obj in algorithms if obj['name'] in selected_algorithm_names]
            results = compare_and_fit_models(models_with_name_list, X, y)
            st.dataframe(results['summary'], use_container_width=True)
            save_models_with_scaler(selected_algorithm_names, results['models_with_scalers'])


st.title('Regression Models Generator [consume]')

data = pd.read_csv('../cleaned_data_modeling.csv')

tab1, tab2 = st.tabs(['Dataframe', 'Models Generator'])

with tab1:
    display_data_and_descriptions(data, 'Dataframe - cleaned_data_modeling.csv')
with tab2:
    display_algorithms_comparison(data)