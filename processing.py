import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from pandas import DataFrame, concat

from RobustSTL import RobustSTL
from PyEMD import EMD, EEMD, Visualisation

def sinewave(N, period, amplitude):
    x1 = np.arange(0, N, 1)
    frequency = 1/period
    theta = 0
    y = amplitude * np.sin(2 * np.pi * frequency * x1 + theta)
    return y

def get_time_series(time_series_name):
    if 'three sine waves' in time_series_name:
        N = 1500
        y1 = sinewave(N, 24, 1) # plt.plot(range(len(y1)), y1)
        y2 = sinewave(N, 168, 1.5) # plt.plot(range(len(y2)), y2)
        y3 = sinewave(N, 672, 2) # plt.plot(range(len(y3)), y3)
        y = y1+y2+y3+np.random.normal(0, 0.2, N)
        y[672:] += 10  # 模拟从样本中间开始的突然变化
    elif 'Sunspots' in time_series_name:
        y = pd.read_csv("datasets/sunspots.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'S&P500' in time_series_name:
        y = pd.read_csv("datasets/S&P500.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'temperatures' in time_series_name:
        y = pd.read_csv("datasets/temperatures.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'Beijing_PM25' in time_series_name:
        y = pd.read_csv("datasets/Beijing_PM25.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'Bike_sharing' in time_series_name:
        y = pd.read_csv("datasets/Bike_sharing.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'ETTh1' in time_series_name:
        y = pd.read_csv("datasets/ETTh1.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'ETTh2' in time_series_name:
        y = pd.read_csv("datasets/ETTh2.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'UCR_insectEPG5_177' in time_series_name:
        y = pd.read_csv("datasets/UCR_insectEPG5_177.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    elif 'UCR_TkeepFirstMARS_157' in time_series_name:
        y = pd.read_csv("datasets/UCR_TkeepFirstMARS_157.csv")
        y = y.values
        y = list(map(list, zip(*y)))
        y = y[0]
        y = np.array(y)
    else:
        print('This series is None.')
        assert False
    return y

def scaling(y, time_series_name):
    if 'Sunspots' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'three sine waves' in time_series_name:
        scaled_y = y
        scaler = None
    elif 'S&P500' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'temperatures' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'Beijing_PM25' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'Bike_sharing' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'ETTh1' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'ETTh2' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'UCR_insectEPG5_177' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    elif 'UCR_TkeepFirstMARS_157' in time_series_name:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        scaled_y = list(map(list, zip(*scaled_y)))
        scaled_y = np.array(scaled_y[0])
    else:
        print('This series is None.')
        assert False
    return scaled_y, scaler

def decompose(scaled_y, decompose_way):
    if decompose_way == 'EMD':
        emd = EMD()
        decomposited_y = emd(scaled_y)
    elif decompose_way == 'EEMD':
        eemd = EEMD()
        decomposited_y = eemd(scaled_y)
    elif decompose_way == 'RobustSTL':
        decomposited_y = RobustSTL(scaled_y, 50, reg1=10.0, reg2=0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)
    return decomposited_y

def to_supervised(data, n_in, n_out, dropnan=True):
    data = data.reshape(len(data), 1)
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def series_to_supervised(decomposited_y, n_in, n_out):
    supervised_y = []
    for i in range(len(decomposited_y)):
        supervised_i = to_supervised(decomposited_y[i], n_in, n_out)
        supervised_y.append(supervised_i.values)
    return supervised_y

def split_test(y, test_rate):
    set1_len = int(test_rate*len(y))
    set2_len = len(y) - set1_len
    set1 = y[:set1_len]
    set2 = y[set1_len:]
    return set1, set2

def split_valid(decomposited_train_valid_y, train_rate, valid_rate):
    train_len = int(len(decomposited_train_valid_y[0])*train_rate/(train_rate+valid_rate))
    valid_len = len(decomposited_train_valid_y[0]) - train_len
    decomposited_train_y = decomposited_train_valid_y[:,:train_len]
    decomposited_valid_y = decomposited_train_valid_y[:, train_len:]
    return decomposited_train_y, decomposited_valid_y

def split_x_y(data, n_in, n_out):
    data_x, data_y = [], []
    for dec_component in data:
        x, y = [], []
        for sample in dec_component:
            x.append(np.array(sample[:n_in]))
            y.append(np.array(sample[-n_out:]))
        data_x.append(np.array(x))
        data_y.append(np.array(y))
    return data_x, data_y