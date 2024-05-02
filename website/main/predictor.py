import os
from tensorflow.keras.models import load_model

predictor_dir = os.path.dirname(os.path.abspath(__file__))

model_paths = {
    'math': os.path.join(predictor_dir, 'model_math.keras'),
    'reading': os.path.join(predictor_dir, 'model_reading.keras'),
    'writing': os.path.join(predictor_dir, 'model_writing.keras'),
}

models = {key: load_model(path) for key, path in model_paths.items()}

def predict_scores(input_data):
    predictions = {
        'math_score': models['math'].predict(input_data).flatten()[0],
        'reading_score': models['reading'].predict(input_data).flatten()[0],
        'writing_score': models['writing'].predict(input_data).flatten()[0],
    }
    return predictions
