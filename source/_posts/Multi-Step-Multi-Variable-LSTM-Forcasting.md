---
title: Multi Step & Multi Variable LSTM Forcasting
date: 2019-12-17 22:23:34
tags: ['keras','model']
categories: ['Models']
---
## Import Library

{% code lang:python %}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline

import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

{% endcode %}

## Inputs
Time series type: day, hour, month

{% code lang:python %}

#hour example

file_name = "ivr进线all.csv"


y = 'ivr_cnt'

timeType = 'hour'

split_perc = 0.7

#choose a number of time steps of input and output
n_steps_in, n_steps_out = 10,20


{% endcode %}


## Model

{% code lang:python %}
def readData(file_name, y, timeType):
    
    filepath = file_name
    #read data
    data = pd.read_csv(filepath)
    
    #define output value
    data.rename(columns = {y: 'y'}, inplace = True)
    df = data
    df['dt'] = pd.to_datetime(df['dt'],format='%Y-%m-%d')
    df = df[~df['y'].isnull()]

    
    #reframe to the timeseries type we want: month or day or hour
    
    if timeType == 'hour':
        df = df.sort_values(['dt','h'], ascending=True)
        df['dt_h'] = df['dt'].astype(str) + '-'+ df['h'].map(str)
        df.set_index('dt_h',inplace=True)

        
    elif timeType == 'month':
            df = df.sort_values(['dt','m'], ascending=True)
            df['dt_m'] = df['dt'].map(str) + '-'+ df['m'].map(str)
            df.set_index('dt_m',inplace=True)
        
    elif timeType == 'day':
            df = df.sort_values(['dt'], ascending=True)
            df.set_index('dt',inplace=True)
    
    return df 

df = readData(file_name, y, timeType)
df = df['y']

#scale data and get values
def scaleData(df):
    m = df.mean()
    s = df.std()
    df_scaled = (df - m) / s
    
    values = df_scaled.values
    
    return values, df_scaled 

values, df_scaled = scaleData(df)



def splitTrainTest(df_scaled, values, split_perc):

    #split into train and test sets

    n_train = round(split_perc*(len(df_scaled)))
    train = values[:n_train]
    test = values[n_train:]
    
    return train, test

train, test = splitTrainTest(df_scaled, values, split_perc)

#split a multivariate sequence into samples

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    
    for i in range(len(sequence)):
        
        #find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        #checkif we are beyond the dataset
        if out_end_ix > len(sequence):
            break
            
        #gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

    #convert into input/output
model_train_X, model_train_y = split_sequence(train, n_steps_in, n_steps_out)
model_test_X, model_test_y = split_sequence(test, n_steps_in, n_steps_out)
   
    
#convert into input/output
#reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

model_train_X = model_train_X.reshape((model_train_X.shape[0], model_train_X.shape[1], n_features))
model_test_X = model_test_X.reshape((model_test_X.shape[0], model_train_X.shape[1], n_features))


def trainModel(n_steps_in, n_steps_out,model_train_X, model_train_y,model_test_X, model_test_y):
    
    #define model
    n_features = 1

    model = Sequential()
    model.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape = (n_steps_in, n_features)))
    model.add(LSTM(100, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out))
    model.compile(optimizer = 'adam', loss = 'mse')

      # fit model
    history = model.fit(model_train_X, model_train_y, epochs = 50, verbose = 1, validation_data = (model_test_X, model_test_y))
    
    plt.plot(history.history['loss'], label = 'train loss')
    plt.plot(history.history['val_loss'], label = 'test loss')
    plt.legend()
    plt.show()
    
    
    return  model


model = trainModel(n_steps_in, n_steps_out,model_train_X, model_train_y,model_test_X, model_test_y)

def train_mse(model, model_train_X,model_train_y,n_steps_out):
    
    yhat_t = model.predict(model_train_X)
    train_y_out = [[] for i in range(n_steps_out)]
    yhat_t_out = [[] for i in range(n_steps_out)]
    mse = [[] for i in range(n_steps_out)]

    #get the result seperatepy from each step out

    for i in range(len(yhat_t)):
        for k in range(n_steps_out):
            yhat_t_out[k].append(yhat_t[i][k])  #get each yhat result from each step out
            train_y_out[k].append(model_train_y[i][k])  #get each model result from each step out


    #mse from each time step
    for i in range(n_steps_out):
        mse[i] = ((np.array(yhat_t_out[i]) - np.array(train_y_out[i]))**2).mean()
        
        
    #plot yhat vs true value on every predicted day

    plt.figure(figsize=(8,4))
    for i in range(n_steps_out):
        plt.subplot(n_steps_out, 1, i +1)

        pd.Series(train_y_out[i]).plot()
        pd.Series(yhat_t_out[i]).plot()

    print('train_mse: ', mse)
    plt.show()

train_mse(model, model_train_X,model_train_y,n_steps_out)


def test_mse(model,model_test_X,model_test_y,n_steps_out):
    
    # test the model on test data
    yhat_ts = model.predict(model_test_X)

    test_y_out = [[] for i in range(n_steps_out)]
    yhat_ts_out = [[] for i in range(n_steps_out)]
    mse_ts = [[] for i in range(n_steps_out)]

    #get the result seperatepy from each step out

    for i in range(len(yhat_ts)):
        for k in range(n_steps_out):
            yhat_ts_out[k].append(yhat_ts[i][k])  #get each yhat result from each step out
            test_y_out[k].append(model_test_y[i][k])  #get each model result from each step out


    #mse from each time step
    for i in range(n_steps_out):
        mse_ts[i] = ((np.array(yhat_ts_out[i]) - np.array(test_y_out[i]))**2).mean()
        
        
    #plot yhat vs true value on every predicted day

    plt.figure(figsize=(8,4))
    for i in range(n_steps_out):
        plt.subplot(n_steps_out, 1, i +1)
        #plt.plot(test_y_out[i], color = 'blue')
        #plt.plot(yhat_ts_out[i], color = 'r')

        pd.Series(test_y_out[i]).plot()
        pd.Series(yhat_ts_out[i]).plot()


    print('test_mse: ', mse_ts)
    plt.show()

    test_mse(model,model_test_X,model_test_y,n_steps_out)



def out_timeValue(df,timeType):
    
    df_time = df
    df_time.index = df_time.index.map(str)
    
    forecast_time = []
    a = df_time.tail(1).index
    last_date = a[0]
    
    if timeType == 'hour':
        last_hour = datetime.strptime(last_date, "%Y-%m-%d-%H")
        for i in range(n_steps_out):
            t = last_hour + timedelta(hours=i+1)
            forecast_time.append(t)
        
    elif timeType == 'day':
        last_date = last_date.split(" ", 1)[0]
        last_day = datetime.strptime(last_date, "%Y-%m-%d")
        for i in range(n_steps_out):
            t = last_day + timedelta(days=i+1)
            forecast_time.append(t)
        
    elif timeType == 'month':
        last_month = datetime.strptime(last_date, "%Y-%m")
        for i in range(n_steps_out):
            t = last_hour + timedelta(month=i+1)
            forecast_time.append(t)
    
    return forecast_time
    
    
forecast_time = out_timeValue(df,timeType)


def timeToStr(forecast_time,timeType):
    str_time = []
    if timeType == 'hour':
        for i in range(len(forecast_time)):
            t = forecast_time[i].strftime("%Y-%m-%d-%H")
            str_time.append(t)
            
    elif timeType == 'day':
        for i in range(len(forecast_time)):
            t = forecast_time[i].strftime("%Y-%m-%d")
            str_time.append(t)
        
    elif timeType == 'month':
        for i in range(len(forecast_time)):
            t = forecast_time[i].strftime("%Y-%m")
            str_time.append(t)


    return str_time


str_time = timeToStr(forecast_time,timeType)

#forecasting n_timesteps
def forecast(df, n_steps_in, n_features):
    
    forcast_data = df.values

    x_input =values[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose = 0)
    
    
    #scale back
    true_forecast = []
    for i in range(len(yhat[0])):
        true_forecast.append(yhat[0][i]*df.std() + df.mean())
    

    return true_forecast


true_forecast = forecast(df, n_steps_in, n_features)

def plotFinalResult(str_time, true_forecast,df):

    df_fc = pd.DataFrame({'time': str_time,
                          'y':true_forecast})
    df_fc.set_index('time',inplace=True)
    df_fc['label'] = 'predict'
    df_fc = df_fc.reset_index()

    y_real = df.values.tolist()
    real_time = df.index.tolist()
    df_rl = pd.DataFrame({'time': real_time,
                          'y':y_real})
    df_rl.set_index('time',inplace=True)
    df_rl['label'] = 'real'
    df_rl = df_rl.reset_index()


    #plot

    plt.figure(figsize=(15,4))
    plt.plot(df_rl['time'],df_rl['y'], color = 'b')
    plt.plot(df_fc['time'],df_fc['y'], color = 'r')
    plt.show()
    
    
plotFinalResult(str_time, true_forecast,df)

{% endcode %}