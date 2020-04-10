# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:07:22 2019

@author: STU16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import datetime
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)

name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:,['Symbol','Name']]

#########변수###########
code = ['004490','039010','192440','015230','010240','282880','013120','045100','094820','014530','236200','052330','017890','036670','102710']
#종목코드 입력
start_date = '2018-12-01' #시작일
end_date = '2019-12-01' #종료일
########################


sample = pd.DataFrame(columns = ['Symbol','Name'])
k = 0
for i in range(len(name_list)):
    if name_list.iloc[i,0] in code:
        sample.at[k,'Symbol'] = name_list.iloc[i,0]
        sample.at[k,'Name'] = name_list.iloc[i,1]
        k += 1

selected = [i for i in sample['Name']]
 
table = pd.DataFrame()
for i in sample.values:
    temp = fdr.DataReader(i[0],start_date,end_date)
    temp_2 = pd.DataFrame({i[1]:temp['Close']})
    table = pd.concat([table,temp_2],axis = 1)


# reorganise data pulled by setting date as index with
# columns of tickers and their corresponding adjusted prices

# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * len(table)

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * len(table)

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
num_portfolios = 50000

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights) 
    #weigh_value = np.array(1/len(selected))
    #weigts_values = weigh_value.repeat(len(selected)).tolist()
    #비중을 동일하게 분배하고 싶은 경우 위 weights코드 삭제 후 적용
    #변동성과 수익률을 고려한 포트폴리오의 비중을 계산하기 때문에 불필요 한것으로 생각됨
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]

# reorder dataframe columns
df = df[column_order]

#find min Volatility & max sharpe values in the dataframe
min_volatility = df['Volatility'].min()
#변동성이 가장 낮은 포트폴리오

max_sharpe = df['Sharpe Ratio'].max()
#샤프지수가 가장 높은 포트폴리오

#use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

#plot frontier, max sharpe & min volatility values with a scatterplot
plt.figure()
plt.scatter(x=df['Volatility'], y=df['Returns'], c=df['Sharpe Ratio'],cmap=plt.cm.jet,edgecolors='black')
plt.xlabel('Volatility(%)',fontsize = 10)
plt.ylabel('Expected Returns(%)',fontsize = 10)
plt.title('Efficient Frontier',fontsize = 15)
cbar= plt.colorbar(label = 'Sharpe Ratio(%)')
#plt.savefig('../../assets/images/markdown_img/180601_colorbar_numeric_data.svg')
#저장하려면 주석 해제 후 사용

plt.scatter(sharpe_portfolio['Volatility'], sharpe_portfolio['Returns'], marker='*', s=150, c='red')
#최대 샤프지수

plt.scatter(min_variance_port['Volatility'], min_variance_port['Returns'], marker='*', s=150, c= 'blue' )
#최소 변동성
plt.show()


# print the details of the 2 special portfolios
print(min_variance_port.T)
#리스크를 싫어하는 투자자에 경우 최소변동성을 가진 포트폴리오 구성
#기대 수익률 3.40%, 그 떄의 변동성은 15.4%

print(sharpe_portfolio.T)
#리스크가 보정된 최대 수익률을 원하는 투자자
#샤프 비율이 최대인 포트폴리오 구성 
#기대수익률은 20.3%, 변동성은 19.1%

