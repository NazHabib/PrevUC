# dashboard/models.py
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .metrics import calculate_metrics


def load_model(model_name):
    return tf.keras.models.load_model(f'models/{model_name}.keras')

def calculate_all_metrics():
    file_path = 'StudentsPerformance.csv'
    data = pd.read_csv(file_path)

    data['gender'] = data['gender'].map({'male': 1, 'female': 0})
    data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
    data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})
    data = pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])

    feature_columns = data.columns.drop(['math score', 'reading score', 'writing score'])
    X_target1 = data[feature_columns].astype('int32')
    y_target1 = data['math score'].astype('int32')
    X_target2 = data[feature_columns].astype('int32')
    y_target2 = data['reading score'].astype('int32')
    X_target3 = data[feature_columns].astype('int32')
    y_target3 = data['writing score'].astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(X_target1, y_target1, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_target2, y_target2, test_size=0.2, random_state=43)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_target3, y_target3, test_size=0.2, random_state=44)
    math_model = load_model('model_math')
    math_metrics = calculate_metrics(math_model, X_train, y_train, X_test, y_test)

    reading_model = load_model('model_reading')
    reading_metrics = calculate_metrics(reading_model, X_train, y_train2, X_test, y_test2)

    writing_model = load_model('model_writing')
    writing_metrics = calculate_metrics(writing_model, X_train, y_train3, X_test, y_test3)

    return {
        'math': math_metrics,
        'reading': reading_metrics,
        'writing': writing_metrics,
    }
