import numpy as np
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
