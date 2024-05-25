import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(model, X_train, y_train, X_test, y_test, loss_fn):
    if model is None:
        raise ValueError('The model variable cannot be None.')

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate the loss for the training set using the provided loss function
    y_pred_train_loss = loss_fn(y_train, y_pred_train)
    loss_train = np.mean(y_pred_train_loss)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)

    # Ensure loss_test is a single value
    loss_test = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(loss_test, list):
        loss_test = loss_test[0]

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    # Format the values to 2 decimal places
    return {
        'loss_train': float(f"{loss_train:.2f}"),
        'mse_train': float(f"{mse_train:.2f}"),
        'mae_train': float(f"{mae_train:.2f}"),
        'r2_train': float(f"{r2_train:.2f}"),
        'loss_test': float(f"{loss_test:.2f}"),
        'mse_test': float(f"{mse_test:.2f}"),
        'mae_test': float(f"{mae_test:.2f}"),
        'r2_test': float(f"{r2_test:.2f}"),
        'rmse_train': float(f"{rmse_train:.2f}"),
        'rmse_test': float(f"{rmse_test:.2f}"),
    }

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from django.shortcuts import render
from .metrics import calculate_metrics
import matplotlib.pyplot as plt

def model_performance(request):
    file_path = 'dashboard/StudentsPerformance.csv'
    data = pd.read_csv(file_path)

    # Preprocess the data
    data['gender'] = data['gender'].map({'male': 1, 'female': 0})
    data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
    data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})
    data = pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])

    # Prepare the target and feature columns
    feature_columns = data.columns.drop(['math score', 'reading score', 'writing score'])
    X_target1 = data[feature_columns].astype('int32')
    y_target1 = data['math score'].astype('int32')
    X_target2 = data[feature_columns].astype('int32')
    y_target2 = data['reading score'].astype('int32')
    X_target3 = data[feature_columns].astype('int32')
    y_target3 = data['writing score'].astype('int32')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_target1, y_target1, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_target2, y_target2, test_size=0.2, random_state=43)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_target3, y_target3, test_size=0.2, random_state=44)

    # Load models
    math_model = keras.models.load_model('main/model_math.keras')
    reading_model = keras.models.load_model('main/model_reading.keras')
    writing_model = keras.models.load_model('main/model_writing.keras')

    # Calculate metrics
    math_metrics = calculate_metrics(math_model, X_train, y_train, X_test, y_test, mean_squared_error)
    reading_metrics = calculate_metrics(reading_model, X_train2, y_train2, X_test2, y_test2, mean_squared_error)
    writing_metrics = calculate_metrics(writing_model, X_train3, y_train3, X_test3, y_test3, mean_squared_error)

    # Generate and save graphs


    return render(request, 'dashboard/dashboard.html', {
        'math_metrics': math_metrics,
        'reading_metrics': reading_metrics,
        'writing_metrics': writing_metrics,

    })
