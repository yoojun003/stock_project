import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)


def minmax_scaler(data):
    num = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return num/(denominator + 1e-7)

# In[10]:
# # Load Dataset
#########변수###########
code = '051900' #종목코드 입력
start_date = '2015-01-01' #시작일
end_date = '2019-11-22' #종료일
validation_date = '2019-11-29'#19/11/25~11/29일 검증 기간

#종목 이름 가져오기
name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:,['Symbol','Name']]
sample = pd.DataFrame(columns = ['Symbol','Name'])
k = 0
for i in range(len(name_list)):
    if name_list.iloc[i,0] in code:
        sample.at[k,'Symbol'] = name_list.iloc[i,0]
        sample.at[k,'Name'] = name_list.iloc[i,1]
        k += 1
stock_name = [i for i in sample['Name']] #종목 이름

data = fdr.DataReader(code,start_date,end_date) 
val_data = fdr.DataReader(code,end_date,validation_date) 

data.tail()
val_data.tail()
#Compute Mid Price

# In[11]:
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

# # Create Windows

# In[12]:


seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])


# # Normalize Data

# In[13]:


normalized_data = []
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]

#np.random.shuffle(train) #

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape


# # Build a Model

# In[14]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

# In[15]:
# # Training
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=100)



# In[21]:
# # Prediction
pred = model.predict(x_test)
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()





