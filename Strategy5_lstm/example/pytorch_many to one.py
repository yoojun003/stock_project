# -\*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:02:58 2019

@author: STU16
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import torch
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import pandas as pd

torch.manual_seed(123)

seq_length = 7 #many
data_dim = 5 #입력차원
hidden_dim = 10
output_dim = 1 #종가의 dim, 출력 차원
learning_rate = 0.01
iterations = 500 #epoch

###load dataset
name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:,['Symbol','Name']]
code = '051900' #종목코드 입력
start_date = '2015-01-01' #시작일
end_date = '2019-12-01' #종료일
sample = pd.DataFrame(columns = ['Symbol','Name'])

k = 0
for i in range(len(name_list)):
    if name_list.iloc[i,0] in code:
        sample.at[k,'Symbol'] = name_list.iloc[i,0]
        sample.at[k,'Name'] = name_list.iloc[i,1]
        k += 1

stock_name = [i for i in sample['Name']]
data = fdr.DataReader(code,start_date,end_date)
#many(7일)-to-one(1)
data = data[['Open','High','Low','Volume','Close']]
stock_info = data.values.astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다

train_size = int(len(stock_info) * 0.7)
train_set = stock_info[0:train_size]
test_set = stock_info[0:train_size-seq_length:]
val_set = stock_info[0:train_size-seq_length:]
len(val_set)
len(net(testX_tensor).data.numpy())

len(stock_info[0:train_size-seq_length:])
val_set
def minmax_scaler(data):
    num = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return num/(denominator + 1e-7)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0,len(time_series)-seq_length):
        _x = time_series[i:i+seq_length,:]
        _y = time_series[i+seq_length,[-1]]
        #print(_x,'->',_y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)
trainX, trainY = build_dataset(train_set,seq_length)        
testX, testY = build_dataset(test_set,seq_length)        

trainX_tensor = torch.FloatTensor(trainX)        
trainY_tensor = torch.FloatTensor(trainY)        
testX_tensor = torch.FloatTensor(testX)        
testY_tensor = torch.FloatTensor(testY)        

#Apply Rnn:lstm
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net,self).__init__()
        self.rnn = torch.nn.LSTM(input_dim,hidden_dim, num_layers = layers,batch_first = True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim, bias = True)
        
    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:,-1])
        return x
    
net = Net(data_dim,hidden_dim,output_dim,1) 
           
#loss & ooptimizer setting
criterion = torch.nn.MSELoss()
optimizer  = optim.Adam(net.parameters(),lr = learning_rate)

#Training & Evaluaction
for i in range(iterations):
    optimizer.zero_grad()
    outputs = net(trainX_tensor)
    loss = criterion(outputs, trainY_tensor)
    loss.backward()
    optimizer.step()
    #print(i,loss.item())

plt.plot(testY)
plt.plot(net(testX_tensor).data.numpy())
plt.legend(['real','prediction'])
plt.show()    


