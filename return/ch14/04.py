import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
from zipline.api import order_target, record, symbol
from zipline.algorithm import TradingAlgorithm

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2016, 3, 29)
data = web.DataReader("AAPL", "yahoo", start, end)

data = data[['Adj Close']]
data.columns = ['AAPL']
data = data.tz_localize('UTC')

def initialize(context):
    context.i = 0
    context.sym = symbol('AAPL')
    context.hold = False

def handle_data(context, data):
    context.i += 1
    if context.i < 20:
        return

    buy = False
    sell = False

    ma5 = data.history(context.sym, 'price', 5, '1d').mean()
    ma20 = data.history(context.sym, 'price', 20, '1d').mean()

    if ma5 > ma20 and context.hold == False:
        order_target(context.sym, 100)
        context.hold = True
        buy = True
    elif ma5 < ma20 and context.hold == True:
        order_target(context.sym, -100)
        context.hold = False
        sell = True

    record(AAPL=data.current(context.sym, "price"), ma5=ma5, ma20=ma20, buy=buy, sell=sell)

algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data)
result = algo.run(data)

plt.plot(result.index, result.ma5)
plt.plot(result.index, result.ma20)
plt.legend(loc='best')

plt.plot(result.ix[result.buy == True].index, result.ma5[result.buy == True], '^')
plt.plot(result.ix[result.sell == True].index, result.ma5[result.sell == True], 'v')

plt.show()
###########################################
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:59:17 2019

@author: LG
"""

import FinanceDataReader as fdr
import datetime
from zipline.api import order, record, symbol
from zipline.algorithm import TradingAlgorithm
from zipline.finance import commission
from zipline.utils.factory import create_simulation_parameters
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.float_format', '{:3f}'.format)
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)

#########################날짜 인덱스 가져오기

# APPL 날짜 인덱스 가져오기
start = datetime.datetime(2018, 11, 20)
end = datetime.datetime(2019, 12, 5)
d_target = "AAPL"
d_data = fdr.DataReader(d_target, start, end)
d_data = d_data[['Close']]
d_data.columns = ['AAPL']
d_data = d_data.tz_localize('UTC')


#########################zipline 함수

# 초기화 함수 : 날짜, 종목symbol, 현재/남은 주식개수, 수수료, 주식보유여부
def initialize(context):
    context.i = 0
    context.sym = symbol('AAPL')
    context.set_commission(commission.PerDollar(cost=0.00165))


# 거래함수
def handle_data(context, data):
    # 초기에 100주 매수
    if context.i == 0:
        order(context.sym, 100)

    context.i += 1

    # 현재 주가
    now = data.current(context.sym, 'price')

    record(AAPL=now)


#########################타겟 종목

# 은진동환
codes = ['012340']
'012340', '028080', '033310','042110','052460','071850','083470','089150','089970','108230','114450','115160'

#########################zipline 알고리즘

table = DataFrame()
for i in range(len(codes)):
    code = codes[i]
    data = fdr.DataReader(code, start, end)
    data = data[['Close']]
    data.columns = ['target']
    data = data.tz_localize('UTC')
    fst = data.iloc[1][0]
    df = DataFrame(data)

    # APPL 날짜 인덱스 + 타겟 데이터
    data3 = d_data[len(d_data) - len(df):]
    data3['AAPL'] = np.where(1, df['target'], df['target'])

    # 알고리즘 수행
    algo = TradingAlgorithm(sim_params=create_simulation_parameters(capital_base=1000000),
                            initialize=initialize, handle_data=handle_data)
    result = algo.run(data3)

    # 결과 테이블
    table = pd.concat([table, DataFrame(result['portfolio_value'])], axis=1)

# 마지막 컬럼에 평균(최종) 수익률 추가
table = pd.concat([table, table.mean(axis=1)], axis=1)

#########################종목 코드의 이름 뽑아내기
name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:, ['Symbol', 'Name']]

# 입력한 종목의 이름
sample = pd.DataFrame(columns=['Name'])
k = 0
for i in range(len(name_list)):
    if name_list.iloc[i, 0] in codes:
        sample.at[k, 'Name'] = name_list.iloc[i, 1]
        k += 1

# 종목이름
selected = [i for i in sample['Name']]
selected += ['최종']
table.columns = selected

#########################결과

# 종료금액
ends = int(round(table.iloc[-1, -1]))
print('초기금액 : 1000000원')
print('종료금액 : {}원'.format(ends))
print('수익 : {}원'.format(ends - 1000000))
print('수익률 : {}%'.format(round((ends - 1000000) / 1000000 * 100, 2)))

# 모두 합친 그래프
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-dark')
plt.plot(table.index, table)
plt.xlabel('Date')
plt.ylabel('Cash(won)')
plt.legend(table.columns)
plt.title('Final Portfolio Value')
plt.show()

# 최종 수익률 그래프
plt.style.use('seaborn-dark')
plt.plot(table.index, table.iloc[:,-1])
plt.xlabel('Date')
plt.ylabel('Cash(won)')
plt.title('Final Portfolio Value')
plt.show()


print ('버전: ', plt.__version__)
print ('설치 위치: ', plt.__file__)
print ('설정 위치: ', plt.get_configdir())
print ('캐시 위치: ', plt.get_cachedir())
import matplotlib.font_manager as fm

font_path = 'C:/Windows/Fonts/NanumSquareEB.ttf'
fontprop = fm.FontProperties(fname=font_path, size=18)

plt.ylabel('가격', fontproperties=fontprop)
plt.title('가격변동 추이', fontproperties=fontprop)
plt.plot(range(50), data, 'r')
plt.show()


####################################################################
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

err_wtable = []
err_etable = []
blue_1 = []
chun1=['019570', '036200', '036710','038530','048870','060260','065620','085810','094480']
for code in chun1:
    url = "https://finance.naver.com/item/main.nhn?code=" + code
    html = req.urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    if len(soup.select('div.section.cop_analysis div.sub_section')) == 0:  # 재무제표가 없을 때
        err_wtable.append(code)
        continue
    fs_html = soup.select('div.section.cop_analysis div.sub_section')[0]
    fs_date = [item.get_text().strip() for item in fs_html.select('thead th')]  # 날짜 뽑기(열)
    if len(fs_date) != 23:  # 재무제표가 비어있을 때
        err_etable.append(code)
        continue
    fs_data = [item.get_text().strip().replace(',', '') for item in
               fs_html.select('td')]  # 수치 추출(내용물) # 쉼표 때문에 나중에 float으로 안변하니까 쉼표 제거
    for i in range(len(fs_data)):
        if fs_data[i] == '-' or fs_data[i] == '':
            fs_data[i] = np.nan
    for i in range(len(fs_data)):
        fs_data[i] = float(fs_data[i])

    s = pd.Series(fs_data[:10])  # 매출액
    s32 = 100 * (s[1] - s[0]) / s[0]
    s21 = 100 * (s[2] - s[1]) / s[1]
    s10 = 100 * (s[3] - s[2]) / s[2] if pd.isnull(s[3]) == False else s21

    print(code + ' ' + str(float(s32) + float(s21) + float(s10)))
"""048870, 036200, 094480, 036710"""
