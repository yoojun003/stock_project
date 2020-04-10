##################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:31:42 2019

@author: STU16
"""
import datetime
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import mpl_finance
#pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
#promt에서 설치필요

#########변수###########
code = '104480'
#['GS글로벌']
#종목코드 입력
start_date = '2019-6-18' #시작일
end_date = '2019-12-18' #종료일

name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:,['Symbol','Name']]
sample = pd.DataFrame(columns = ['Symbol','Name'])
k = 0
for i in range(len(name_list)):
    if name_list.iloc[i,0] in code:
        sample.at[k,'Symbol'] = name_list.iloc[i,0]
        sample.at[k,'Name'] = name_list.iloc[i,1]
        k += 1

stock_name = [i for i in sample['Name']] #종목이름
data = fdr.DataReader(code,start_date,end_date) #단일 종목
data = data.reset_index()

def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (peak_upper, peak_lower, mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return peak_upper, peak_lower, (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]

mdd = get_mdd(data['Close'])
mdd #결과 1~8일까지 최대낙폭은 -3%

################# plt.show()까지 한꺼번에 실행해야함####
fig = plt.figure(figsize=(8, 5))
fig.set_facecolor('w')
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
axes = []
axes.append(plt.subplot(gs[0]))
axes.append(plt.subplot(gs[1], sharex=axes[0]))
axes[0].get_xaxis().set_visible(False)

x = np.arange(len(data.index))
ohlc = data[['Open', 'High', 'Low', 'Close']].astype(int).values
dohlc = np.hstack((np.reshape(x, (-1, 1)), ohlc))

mpl_finance.candlestick_ohlc(axes[0], dohlc, width=0.5, colorup='r', colordown='b') # 봉차트
axes[1].bar(x, data['Volume'], color='k', width=0.6, align='center') # 거래량 차트
plt.tight_layout()

axes[0].plot(mdd[:2], data.loc[mdd[:2], 'Close'], 'k') #mdd선 표현
plt.legend(stock_name)
plt.show()
#################plot#######

'019570','038530','060260','065620','085810'
'리더스 기술투자', '골드퍼시픽', '뉴보텍', '제낙스', '알티개스트'
