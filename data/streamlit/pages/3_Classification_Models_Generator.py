import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
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

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    confusion_matrices = []

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

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        fitted_models_with_scaler_list.append({
            'model': model_with_name['model'],
            'min_max_scaler' : min_max_scaler
        })

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        confusion_matrices.append(matrix)


    summary_df = {
        'Algorithm': [model_with_name['name'] for model_with_name in models_with_name_list],
        'Accuracy': accuracy_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1': f1_list
    }
    summary_df = pd.DataFrame(summary_df)

    return {
        'summary' : {
            'dataframe' : summary_df,
            'confusion_matrices' : confusion_matrices
        },
        'models_with_scalers' : fitted_models_with_scaler_list
    }

def display_confusion_matrix(selected_algorithm_names, confusion_matrices):
    st.subheader('Confusion Matrices')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Add some extra space between subplots
    fig.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axes.flatten()):
        if i < len(confusion_matrices):
            matrix = confusion_matrices[i]
            ax.matshow(matrix, cmap='Blues')
            ax.set_title(f"{selected_algorithm_names[i]}")
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_xticklabels([''] + ['E10', 'SP98'], fontsize=8)
            ax.set_yticklabels([''] + ['E10', 'SP98'], fontsize=8)
            for (j, k), label in np.ndenumerate(matrix):
                ax.text(k, j, label, ha='center', va='center')
        else:
            ax.axis('off')

    st.pyplot(fig)


def save_models_with_scaler(selected_algorithm_names, models_with_scaler_list):
    for algorithm_name, model_with_scaler in zip(selected_algorithm_names, models_with_scaler_list):
        algorithm_name = algorithm_name.lower().replace(' ', '_')
        with open(f'../models/classification/{algorithm_name}.pkl', 'wb') as file:
            pickle.dump(model_with_scaler, file)


def display_algorithms_comparison(df):
    df = df.sample(frac=1)

    X = df.drop(['gas_type'], axis=1)
    y = df['gas_type']

    algorithms = [
        {
            'name': 'Logistic Regression', 
            'model': LogisticRegression()
        },
        {
            'name': 'k-Nearest Neighbors', 
            'model': KNeighborsClassifier()
        },
        {
            'name': 'Decision Trees', 
            'model': DecisionTreeClassifier()
        },
        {
            'name': 'Random Forests', 
            'model': RandomForestClassifier()
        },
        {
            'name': 'Naive Bayes', 
            'model': GaussianNB()
        },
        {
            'name': 'Support Vector Machines', 
            'model': SVC()
        },
        {
            'name': 'Neural Networks', 
            'model': MLPClassifier()
        }
    ]

    with st.form("model_generator_form"):
        selected_algorithm_names = st.multiselect("Algorithms To Compare", [obj['name'] for obj in algorithms])

        submitted = st.form_submit_button("Compare")
        if submitted:
            models_with_name_list = [obj for obj in algorithms if obj['name'] in selected_algorithm_names]
            results = compare_and_fit_models(models_with_name_list, X, y)
            st.dataframe(results['summary']['dataframe'], use_container_width=True)
            display_confusion_matrix(selected_algorithm_names, results['summary']['confusion_matrices'])
            save_models_with_scaler(selected_algorithm_names, results['models_with_scalers'])


st.title('Classification Models Generator [gas_type]')

data = pd.read_csv('../cleaned_data_modeling.csv')

tab1, tab2 = st.tabs(['Dataframe', 'Models Generator'])

with tab1:
    display_data_and_descriptions(data, 'Dataframe - cleaned_data_modeling.csv')
with tab2:
    display_algorithms_comparison(data)