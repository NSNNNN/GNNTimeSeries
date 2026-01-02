import numpy as np

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))

def mape(y, y_hat, eps=1e-5):
    return np.mean(np.abs((y - y_hat) / (np.abs(y) + eps))) * 100
