
import pandas as pd
import pandas_datareader.data as web
import datetime
import torch #없으면 prompt창에서 conda install torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import argparse
from copy import deepcopy # Add Deepcopy for args
from sklearn.metrics import mean_absolute_error

import seaborn as sns 
import matplotlib.pyplot as plt

print(torch.__version__)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 9)


# # Pandas Datareader Test

# In[2]:



 
# We will look at stock prices over the past year, starting at January 1, 2016
start = (2000, 12, 1)
start = datetime.datetime(*start)
end = datetime.date.today()

google = web.DataReader('028050.KS', 'yahoo', start, end)


# In[64]:


google.Low.plot(grid=True)
#low컬럼을 기준으로 plot 그림

# In[6]:


google.tail()
google.head()
#6개 feature(주식정보), index는 날짜, 타입은 데이터프레임
#6차원 벡터, 10개 데이터 입력 시 다음날 가격 예측
 
print(google.isna().sum())


# # Data Preparation
#t+1~t+5까지 예측하는 프로그램
#many to one(t+1만 예측)에 모델을 생성할 경우 전날 가격이랑 똑같은 가격을 예측
#왜냐면 loss function으로 mse가 사용되는데 마지막 값을 예측하면서 mse값을 줄어들게끔 학습함

# In[115]:
#dataset은 torch 유틸리티 사용

class StockDataset(Dataset): 
    
    def __init__(self, symbol, x_frames, y_frames, start, end): 
        
        self.symbol = symbol 
        self.x_frames = x_frames #최근 n일에 데이터
        self.y_frames = y_frames #x_frames이후 n일에 데이터
        
        self.start = datetime.datetime(*start) #시작일
        self.end = datetime.datetime(*end) #종료일

        self.data = web.DataReader(self.symbol, 'yahoo', self.start, self.end)
        print(self.data.isna().sum()) #dataframe에 na값 체크
        
    def __len__(self):
        return len(self.data) - (self.x_frames + self.y_frames) + 1
    #len : dataset의 길이를 리턴 
    #이유 dataset이 batch를 만들 때 dataset에 크기를 알아야하기 때문    
    #1~5일치 데이터가 있다면 x는 2일치 y는 이후 2일치라고 가정한다면
    #1,2,3,4,5 => [1,2-3,4],[2,3-4,5] 2개의 레코드 생성
    
    def __getitem__(self, idx):
        idx += self.x_frames
        data = self.data.iloc[idx-self.x_frames:idx+self.y_frames]
        data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] #순서 변경
        data = data.apply(lambda x: np.log(x+1) - np.log(x[self.x_frames-1]+1))
        #추세를 보기위해 nomarlazation lambda사용
        #ex)1 1 2 2    -> 0.5 0.5 1 1 비율로 변경
        #   10 10 20 20-> 0.5 0.5 1 1
        #0인 주가가 있을경우 log함수에선 발산, 1씩 더함, 1을 극소값을 사용해도 괜찮을 듯
        data = data.values #np.array로 convert
        X = data[:self.x_frames] 
        y = data[self.x_frames:]
        
        return X, y
    #getitem : 인덱싱하는 문법
    #i번째 요청이 올 경우 dataset에서 x,y를 반환

#test
dataset = StockDataset('028050.KS',10,5,(2001,1,1),(2005,1,1))
type(dataset.data.iloc[0:2].values[:2])

dataloder = DataLoader(dataset, 2)
for X, y in dataloder:
    print(X.shape,y.shape)
    break
#torch.Size([2, 10, 6]) torch.Size([2, 5, 6])
#x = [batchsize,최근10일,feature], y = [batchsize,이후 5일,feature]
##########위 예제####
class dummyset():
    def __init__(self, num_data):
        self.x = np.array(list(range(num_data*2))).reshape(-1,2)
        self.y = np.array(list(range(num_data)))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
        
dataset = dummyset(100)
print(dataset.x)
print(len(dataset))
print(dataset[0])
###############################
dataloder = DataLoader(dataset, 3,shuffle = True, drop_last = True)
#suffle = True : 중복없이 suffle 수행
#drop_last = True : 마지막 결과는 삭제

for batch in dataloder:
    x = batch[0]
    y = batch[1]
    print(batch)
    break

for x,y in dataloder:
    print(x.shape,y.shape)
     #위와 동일, 마지막은 100//3 = 1에 결과가 나옴
###############################################    
#결과값
#x = batchsize * dimension (3,2)
#y = batchsize에 dimension (3)
    

# # Model Define
#모델 아키텍쳐
# In[165]:


class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(LSTM, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn 
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers) # pyroch:nn.lstm
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor()
        
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    
    def make_regressor(self):
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor
    
    #nlp만드는 함수
    
    def forward(self, x):
        #print('x shape',x.shape) #구조확인용
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print('output shape : ',lstm_out.shape, self.hidden[0].shape,self.hidden[1].shape) #구조확인용
        #결과 lstm_out.shape : [10,128,50] 50은 hidden dim, 각 10step에서 50개의 hid_dim 생성
        # self.hidden[0].shape : [1,128,50] 1은 맨 마지막 t에서 나온 cell state의 dim, 각 10step에서 50개의 hid_dim 생성
        # self.hidden[1].shap : [1,128,50] 1은 맨 마지막 t에서 나온 cell state의 dim, 각 10step에서 50개의 hid_dim 생성
        
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1))     
        #마지막 가격까지 regressor에 넣어서 ht유추
        #view는 reshape
        
        #print('pred shape', y_pred.shape) #결과 :[128,5] 5일치를 예측함(t+5) #구조확인용
        return y_pred
        #lstm_out은 ht, 각 타임 step에서의 output값,
    

# In[153]:

#성능지표로 보여주기 위해 metric 구현
#ex) 11만원 12만원 ->(12-11)/11 모델 성능 평가
def metric(y_pred, y_true):
    perc_y_pred = np.exp(y_pred.cpu().detach().numpy()) 
    perc_y_true = np.exp(y_true.cpu().detach().numpy())
    #y_pred는 추세에 관한 비율 값,log(xt)-log(x_last+1) = exp(log(xt/x_last)) = xt/x_last
    #+1은 추가 term은 0으로 근사하기 떄문에 바로 exp를 사용함, y_pred값 역 정규화
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values') #차이의 절대값에 합 예측값과 실제 주가가격의 차이로 평가
    return mae*100

#가격차이를 matric 
#실제 10만원 , 예측 13만원, 12만원...  
#-1만원/10만원 = 10%가 mae값이 됨, 
#현재 예시 결과는 2~3프로 나옴 하루 주가 변동폭보다 큼 0.1%를 목표로 조정해보자
    
# # Train, Validate, Test 

# In[148]:

#dataset의 역할 : batch사이즈 크기의 데이터를 i번째에서 반환하는 역할
#datalodder : batch size * dimension 크기 곱해서 넘겨줌
def train(model, partition, optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'], 
                             batch_size=args.batch_size, 
                             shuffle=True, drop_last=True)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    train_acc = 0.0 #초기화
    train_loss = 0.0 #초기화
    for i, (X, y) in enumerate(trainloader):
        #X의 구조 = [10, n, 6]형태를 [n,10,6]으로 변환해야함, (배치사이즈,sequence length,독립변수 갯수)
        #Y(종가)의 구조 : [10, m, 1] or [10,m]
        #print('raw',X.shape,y.shape) #구조확인용
        X = X.transpose(0, 1).float().to(args.device) #X의 구조 = [10, n, 6]형태를 [n,10,6]으로 변환해야함
        y_true = y[:, :, 3].float().to(args.device) 
        #print(torch.max(X[:, :, 3]), torch.max(y_true))
        #input_feature는 6차원 예측은 종가만 하기 위해서 3을 선정

        model.zero_grad()
        optimizer.zero_grad()
        model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

        #print('processed',X.shape,y_true.shape) #구조확인용
        #[128,5] close값(종속변수)을 제외하고 sequence로 만들었기 때문에 5개 길이에 sequence가 128개 생성
        y_pred = model(X)
        loss = loss_fn(y_pred.view(-1), y_true.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += metric(y_pred, y_true)[0]

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / len(trainloader)
    return model, train_loss, train_acc


# In[150]:


def validate(model, partition, loss_fn, args):
    valloader = DataLoader(partition['val'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    model.eval()

    val_acc = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:, :, 3].float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            loss = loss_fn(y_pred.view(-1), y_true.view(-1))

            val_loss += loss.item()
            val_acc += metric(y_pred, y_true)[0]

    val_loss = val_loss / len(valloader)
    val_acc = val_acc / len(valloader)
    return val_loss, val_acc


# In[156]:


def test(model, partition, args):
    testloader = DataLoader(partition['test'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    model.eval()

    test_acc = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:, :, 3].float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            test_acc += metric(y_pred, y_true)[0]

    test_acc = test_acc / len(testloader)
    return test_acc


# In[160]:

def experiment(partition, args):

    model = LSTM(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size, args.dropout, args.use_bn)
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()

    loss_fn = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')
    
    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #
        
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        model, train_loss, train_acc = train(model, partition, optimizer, loss_fn, args)
        val_loss, val_acc = validate(model, partition, loss_fn, args)
        te = time.time()
        
        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # ============================ #
        
        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(epoch, train_acc, val_acc, train_loss, val_loss, te-ts))
        
    test_acc = test(model, partition, args)    
    
    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc
    return vars(args), result


# # Manage Experiment

# In[169]:


import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd

def save_exp_result(setting, result):
    exp_name = setting['exp_name']
    del setting['epoch']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = './results/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    with open(filename, 'w') as f:
        json.dump(result, f)

    
def load_exp_result(exp_name):
    dir_path = './results'
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        if exp_name in filename:
            with open(join(dir_path, filename), 'r') as infile:
                results = json.load(infile)
                list_result.append(results)
    df = pd.DataFrame(list_result) # .drop(columns=[])
    return df


# In[178]:



def plot_acc(var1, var2, df):

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 6)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    sns.barplot(x=var1, y='train_acc', hue=var2, data=df, ax=ax[0])
    sns.barplot(x=var1, y='val_acc', hue=var2, data=df, ax=ax[1])
    sns.barplot(x=var1, y='test_acc', hue=var2, data=df, ax=ax[2])
    
    ax[0].set_title('Train Accuracy')
    ax[1].set_title('Validation Accuracy')
    ax[2].set_title('Test Accuracy')

    
def plot_loss_variation(var1, var2, df, **kwargs):

    list_v1 = df[var1].unique()
    list_v2 = df[var2].unique()
    list_data = []

    for value1 in list_v1:
        for value2 in list_v2:
            row = df.loc[df[var1]==value1]
            row = row.loc[df[var2]==value2]

            train_losses = list(row.train_losses)[0]
            val_losses = list(row.val_losses)[0]

            for epoch, train_loss in enumerate(train_losses):
                list_data.append({'type':'train', 'loss':train_loss, 'epoch':epoch, var1:value1, var2:value2})
            for epoch, val_loss in enumerate(val_losses):
                list_data.append({'type':'val', 'loss':val_loss, 'epoch':epoch, var1:value1, var2:value2})

    df = pd.DataFrame(list_data)
    g = sns.FacetGrid(df, row=var2, col=var1, hue='type', **kwargs)
    g = g.map(plt.plot, 'epoch', 'loss', marker='.')
    g.add_legend()
    g.fig.suptitle('Train loss vs Val loss')
    plt.subplots_adjust(top=0.89) # 만약 Title이 그래프랑 겹친다면 top 값을 조정해주면 됩니다! 함수 인자로 받으면 그래프마다 조절할 수 있겠죠?


def plot_acc_variation(var1, var2, df, **kwargs):
    list_v1 = df[var1].unique()
    list_v2 = df[var2].unique()
    list_data = []

    for value1 in list_v1:
        for value2 in list_v2:
            row = df.loc[df[var1]==value1]
            row = row.loc[df[var2]==value2]

            train_accs = list(row.train_accs)[0]
            val_accs = list(row.val_accs)[0]
            test_acc = list(row.test_acc)[0]

            for epoch, train_acc in enumerate(train_accs):
                list_data.append({'type':'train', 'Acc':train_acc, 'test_acc':test_acc, 'epoch':epoch, var1:value1, var2:value2})
            for epoch, val_acc in enumerate(val_accs):
                list_data.append({'type':'val', 'Acc':val_acc, 'test_acc':test_acc, 'epoch':epoch, var1:value1, var2:value2})

    df = pd.DataFrame(list_data)
    g = sns.FacetGrid(df, row=var2, col=var1, hue='type', **kwargs)
    g = g.map(plt.plot, 'epoch', 'Acc', marker='.')

    def show_acc(x, y, metric, **kwargs):
        plt.scatter(x, y, alpha=0.3, s=1)
        metric = "Test Acc: {:1.3f}".format(list(metric.values)[0])
        plt.text(0.05, 0.95, metric,  horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(facecolor='yellow', alpha=0.5, boxstyle="round,pad=0.1"))
    g = g.map(show_acc, 'epoch', 'Acc', 'test_acc')

    g.add_legend()
    g.fig.suptitle('Train Accuracy vs Val Accuracy')
    plt.subplots_adjust(top=0.89)


# In[170]:

!mkdir results #경로 지정
# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.symbol = '028050.KS'
args.batch_size = 258
args.x_frames = 7
args.y_frames = 3
 
# ====== Model Capacity ===== #
args.input_dim = 6
args.hid_dim = 20
args.n_layers = 2

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.7 #드랍아웃 쓸지 말지 
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam' #'RMSprop' #SGD, RMSprop, ADAM... #optimizer 선정
args.lr = 0.01 #learning rate조정 
args.epoch = 2 #epoch 갯수

# ====== Experiment Variable ====== #
name_var1 = 'lr'
name_var2 = 'n_layers' 
list_var1 = [0.1,0.01,0.001] #lr 3개 비교
list_var2 = [1,2,3] #n_layer는 1~3개 비교

# ========== 실제 데이터 넣기 ============ #
trainset = StockDataset(args.symbol, args.x_frames, args.y_frames, (2000,1,1), (2012,1,1)) #학습기간
valset = StockDataset(args.symbol, args.x_frames, args.y_frames, (2012,1,1), (2016,1,1)) #validation기간
testset = StockDataset(args.symbol, args.x_frames, args.y_frames, (2016,1,1), (2019,2,1)) #test기간
partition = {'train': trainset, 'val':valset, 'test':testset}
#args.x_frames와 args.y_frames를 그떄 그때 바꿔줄수 있음

#결과문
for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)
                
        setting, result = experiment(partition, deepcopy(args))
        save_exp_result(setting, result)


# In[179]:

var1 = 'lr'
var2 = 'n_layers'
df = load_exp_result('exp1')
df.head()

plot_acc(var1, var2, df)
plot_loss_variation(var1, var2, df, sharey=False) #sharey를 True로 하면 모둔 subplot의 y축의 스케일이 같아집니다.
plot_acc_variation(var1, var2, df, margin_titles=True, sharey=True)

#과제는 튜닝해보기
#hyperparameter 조정해서 matric값 줄여보기

# In[ ]:

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_data = np.array([x[len(x)-seq_length : ]])
print("recent_data.shape:", recent_data.shape)
print("recent_data:", recent_data)

# 내일 종가를 예측해본다
test_predict = sess.run(hypothesis, feed_dict={X: recent_data})
print("test_predict", test_predict[0])
test_predict = reverse_min_max_scaling(price,test_predict) # 금액데이터 역정규화한다
print("Tomorrow's stock price", test_predict[0]) # 예측한 주가를 출력한다


