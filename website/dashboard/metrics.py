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

    loss_test = model.evaluate(X_test, y_test, verbose=0)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    return {
        'loss_train': loss_train,
        'mse_train': mse_train,
        'mae_train': mae_train,
        'r2_train': r2_train,
        'loss_test': loss_test,
        'mse_test': mse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
    }