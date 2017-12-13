# ML_Reseach

```{.python}
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
import math

%matplotlib inline
 
# csv 파일로부터 데이타를 읽어옵니다.
dataset =pd.read_csv('ADA.csv',names=['date','price'], usecols=[1], engine='python', skipfooter=3)
 
# 데이터의 분포가 큰 경우 일정 범위 내로 데이터의 모양을 재 조정하게 되는데 
# sklearn의 Scaler함수를 이용하여 재 조정 할 수 있습니다.
# 밑 MinMaxScaler는 데이터 분포를 0~1 사이로 재 조정하는 함수입니다.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
 
# 학습 데이터 사이즈를 전체 데이터의 0.67, 나머지를 테스트 데이터로 사용합니다.
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
 
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
 
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
 
# 모델에서 읽을 수 있는 형대로 다시 모양을 만들어 줍니다.
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
 
# 연속된 데티터를 처리하기 위한 구조
model = Sequential()
# LSTM모델을 추가
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
 
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# RMSE(Root Mean Seuared Error)
# 오차를 제곱해서 평균을 한 값의 제곱근을 뜻합니다. 통계학의 표준편차와 유사합니다.
# 실 도입전 모델을 검증하기위한 용도로 쓰입니다.
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
 
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
 
raw = []
num = 0
# 데이터의 수가 너무 많아 그래프가 잘 보이지 않기 때문에
# 500개 단위로 끊어서 데이터를 표시
for a in dataset:
    num+=1
    if num%500 == 0:
        raw.append(a)
        
train = []
num = 0
for a in trainPredictPlot:
    num+=1
    if num%500 == 0:
        train.append(a)
        
test = []
num = 0
for a in testPredictPlot:
    num+=1
    if num%500 == 0:
        test.append(a)
 
 
plt.figure(figsize=(15,20))
plt.subplot(211)
plt.plot(scaler.inverse_transform(raw),'r')
 
 
plt.grid()
plt.plot(train)
plt.plot(test,'g')
plt.grid()
```
