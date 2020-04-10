#tf:ANN 종가만 예측

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pandas_datareader.data as web
tf.reset_default_graph()
tf.set_random_seed(101)

def evaluate_ts(features,y_true,y_pred): #평가 함수 
    print('Evaluation of the predictions : ')
    print('MSE : ', np.mean(np.square(y_true - y_pred))) #실제 차이
    print('MAE : ', np.mean(np.abs(y_true - y_pred))) #실제 차이
    
    print('Benchmark : if prediction == last feature') #학습 전 차이 
    print('MSE : ', np.mean(np.square(features[:,-1] - y_true)))
    print('MAE : ', np.mean(np.abs(features[:,-1] - y_true)))
    
    plt.plot(matrix_to_array(y_true),'b')
    plt.plot(matrix_to_array(y_pred),'r')
    plt.xlabel('Days')
    plt.xlabel('Prediction and true values')
    plt.title('Predicted(Red) vs Real(Blue)')
    plt.show()
    
    error = np.abs(matrix_to_array(y_pred)-matrix_to_array(y_true))
    plt.plot(error, 'r')
    fit = np.polyfit(range(len(error)),error, deg = 1) #error값을 사용해 1차 함수 생성
    print('fit:',fit) #결과는 (계수,상수)
    print('degree : ',fit[0],'graph range : ',range(len(error)),'constant : ',fit[1])
    plt.plot(fit[0] * range(len(error))+fit[1],'--') #np.polyfit으로 생성된 1차 함수 그래프 
    plt.xlabel('Days')
    plt.ylabel('Prediction error L1 norm')
    plt.title('Prediction error (absolute) and trendline')
    plt.show()


# Settings for the dataset creation
feat_dimension = 20
train_size = 252
test_size = 252 - feat_dimension

# Settings for tensorflow
learning_rate = 0.5
optimizer = tf.train.AdamOptimizer
n_epochs = 20000

# Fetch the values, and prepare the train/test split
start = datetime.datetime(2015,1,1)
end = datetime.datetime(2016,12,31)
raw_dataframe = web.DataReader("051900.KS", "yahoo", start, end) 
raw_dataframe = raw_dataframe.loc[:,['Open','High','Low','Close','Adj Close','Volume']]
stock_values = [i for i in raw_dataframe['Adj Close']]
minibatch_cos_X, minibatch_cos_y = format_dataset(stock_values, feat_dimension)

train_X = minibatch_cos_X[:train_size, :].astype(np.float32)
train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
test_X = minibatch_cos_X[train_size:, :].astype(np.float32)
test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)


# Here, the tensorflow code
X_tf = tf.placeholder("float", shape=(None, feat_dimension), name="X")
y_tf = tf.placeholder("float", shape=(None, 1), name="y")


# Here the model: a simple linear regressor
def regression_ANN(x, weights, biases):
    return tf.add(biases, tf.matmul(x, weights))


# Store layers weight & bias
weights = tf.Variable(tf.truncated_normal([feat_dimension, 1], mean=0.0, stddev=1.0), name="weights")
biases = tf.Variable(tf.zeros([1, 1]), name="bias")


# Model, cost and optimizer
y_pred = regression_ANN(X_tf, weights, biases)
cost = tf.reduce_mean(tf.square(y_tf - y_pred))
train_op = optimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # For each epoch, the whole training set is feeded into the tensorflow graph
    for i in range(n_epochs):
        train_cost, _ = sess.run([cost, train_op], feed_dict={X_tf: train_X, y_tf: train_y})
        if i % 1000 == 0:
            print("Training iteration", i, "MSE", train_cost)

    # After the training, let's check the performance on the test set
    test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf: test_X, y_tf: test_y})
    print("Test dataset:", test_cost)

    # Evaluate the results
    evaluate_ts(test_X, test_y, y_pr)

    # How does the predicted look like?
    plt.plot(range(len(stock_values)), stock_values, 'b')
    plt.plot(range(len(stock_values)-test_size, len(stock_values)), y_pr, 'r--')
    plt.xlabel("Days")
    plt.ylabel("Predicted and true values")
    plt.title("Predicted (Red) VS Real (Blue)")
    plt.show()
