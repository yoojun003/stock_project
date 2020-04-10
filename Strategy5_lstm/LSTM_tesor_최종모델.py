# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 01:17:38 2019

@author: park
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 00:09:11 2019

@author: park
"""

import tensorflow as tf
pip install --upgrade tensorflow==1.13.1
#tf.__version__
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
#pip install -U finance-datareader
import FinanceDataReader as fdr
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)
 
# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
tf.set_random_seed(777)
  
# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()
 
# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원
 
# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()
 
# 모델(LSTM 네트워크) 생성
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim, 
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell
 
# 하이퍼파라미터
input_data_column_cnt = 5  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1 # 결과데이터의 컬럼 개수
seq_length = 28            # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20   # 각 셀의 (hidden)출력 크기
forget_bias = 1.0          # 망각편향(기본값 1.0)
num_stacked_layers = 1     # stacked LSTM layers 개수
keep_prob = 1.0            # dropout할 때 keep할 비율
epoch_num = 1000           # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01       # 학습률

# # Load Dataset
#########변수###########
start_date = '2015-01-01' #시작일
end_date = '2019-11-25' #종료일, 검증일-1, ex)26일 검증을 위한다면 25입력
validation_date = '2019-11-29'#19/11/25~11/29일 검증 기간

#종목 이름 가져오기
name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:,['Symbol','Name']]
lstm_df = pd.DataFrame()

#################모델 적용#################
#한종목씩 코드를 가져와 lstm 모델 적용
#모델 돌릴 시 rmse와 test결과를 파일로 저장, 저장위치를 지정해줘야함
#검증을 위해 결과를 dataframe으로 구현하여 csv파일로 저장, 저장 위치를 지정해줘야함

for i,n in enumerate(name_list['Symbol']):
    tf.set_random_seed(777)
    tf.reset_default_graph()
    code = n
    stock_name = name_list['Name'][i]
    raw_df = fdr.DataReader(code,start_date,end_date)
    if len(raw_df) < 80: 
        continue
    #상장 후 80일 미만인 종목은 PASS, 다음 종목 모델 적용
    
    raw_df  = raw_df[['Open','High','Low','Close','Volume']]
    stock_info = raw_df.values.astype(np.float)
    price = stock_info[:,:-1] #['Open','High','Low','Close'], close 는 수정종가
    norm_price = min_max_scaling(price) # 가격형태 데이터 정규화 처리
    volume = stock_info[:,-1:]
    norm_volume = min_max_scaling(volume) # 거래량형태 데이터 정규화 처리
    x = np.concatenate((norm_price, norm_volume), axis=1) # axis=1, 세로로 합친다
    y = x[:,[-2]] # 타켓은 주식 'Close'열 
    dataX = [] 
    dataY = [] 
    
    for i in range(0, len(y) - seq_length):
        _x = x[i : i+seq_length]
        _y = y[i + seq_length] 
        dataX.append(_x) 
        dataY.append(_y) 
    
    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])
    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])
    X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
    Y = tf.placeholder(tf.float32, [None, 1])   
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])  
    stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
    multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()
    hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
    hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)
    loss = tf.reduce_sum(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))
    
    train_error_summary = [] # 학습용 데이터의 오류를 중간 중간 기록한다
    test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
    test_predict = ''        # 테스트용데이터로 예측한 결과
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epoch_num):
        _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        if ((epoch+1) % 100 == 0) or (epoch == epoch_num-1): 
            train_predict = sess.run(hypothesis, feed_dict={X: trainX})
            train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
            train_error_summary.append(train_error)
            test_predict = sess.run(hypothesis, feed_dict={X: testX})
            test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
            test_error_summary.append(test_error)
      
#    plt.figure(1)
#    plt.plot(train_error_summary, 'r',label = 'train error')
#    plt.plot(test_error_summary, 'b',label = 'test error')
#    plt.xlabel('Epoch(x100)')
#    plt.ylabel('Root Mean Square Error')
#    plt.legend(fontsize='x-large')
#    plt.title('RMSE Error',fontsize = 20)
#    plt.grid()
#    plt.savefig(r'C:\Users\STU16\Desktop/교육/200_주식/MSE_IMG/'+stock_name+'.png', format='png')
#    plt.show()
    
#    plt.figure(2)
#    plt.plot(testY, 'r', label = 'real')
#    plt.plot(test_predict, 'b',label = 'predict')
#    plt.xlabel('Time Period')
#    plt.ylabel('Stock Price')
#    plt.title(stock_name)
#    plt.grid()
#    plt.legend(fontsize='x-large')
#    plt.savefig(r'C:\Users\STU16\Desktop/교육/200_주식/PREDICT_IMG/'+stock_name+'.png', format='png')
#    plt.show()
    
    recent_data = np.array([x[len(x)-seq_length : ]])
    val_predict = sess.run(hypothesis, feed_dict={X: recent_data})
    val_predict = reverse_min_max_scaling(price,val_predict) # 금액데이터 역정규화한다
    val_df = fdr.DataReader(code,end_date,validation_date)
    val_df  = val_df[['Open','High','Low','Close']]
    val_df = val_df.reset_index()
    temp = pd.DataFrame({'date': val_df['Date'][1],
              'name':stock_name,
              'code':code,
              'yesterday':val_df['Close'][0],
              'today':val_df['Close'][1],
              'predict':np.round(val_predict[0])})
    lstm_df = pd.concat([lstm_df,temp],axis=0)   
    lstm_result = lstm_df
    lstm_result.to_csv('C:/result/lstm_result_1126.csv',
                       header = True,
                       index = False)
    

