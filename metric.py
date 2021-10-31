from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math

def metric_mae(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    perc_y_true = y_true.cpu().detach().numpy()
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    return mae


def metric_rmse(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    perc_y_true = y_true.cpu().detach().numpy()
    mse = mean_squared_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    rmse = math.sqrt(mse)
    return rmse

def metric_mape(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    perc_y_true = y_true.cpu().detach().numpy()
    mape = mean_absolute_percentage_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    return mape