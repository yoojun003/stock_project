#from zipline.api import order_target, record, symbol
import pandas_datareader.data as web
import datetime
import matplotlib
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, Series
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#import mpl_finance
#import matplotlib.finance as matfin
dic1=[{"永鼎股份(5G)":"600105.SS"},
      {"大唐电信(5G)":"600198.SS"},
      {"*ST信威(5G)":"600485.SS"},
      {"宜安科技(5G)":"300328.SZ"},
      {"永鼎股份(smart car)":"600105.SS"},
      {"威帝股份(smart car)":"603023.SS"},
      {"奋达科技(smart car)":"002681.SZ"},
      {"日上集团(smart car)":"002593.SZ"},
      {"海尔智家(smart home)":"600690.SS"},
      {"巨星科技(smart home)":"002444.SZ"},
      {"全志科技(smart home)":"300458.SZ"},
      {"梦百合(smart home)":"603313.SS"},
      {"德尔未来(graphene)":"002631.SZ"},
      {"中科电气(graphene)":"300035.SZ"},
      {"长信科技(graphene)":"300088.SZ"},
      {"华丽家族(graphene)":"600503.SS"}]

dic2=[{'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)}]

dic3=['5G','5G','5G','5G',
      'smart car','smart car','smart car','smart car',
      'Smart home','Smart home','Smart home','Smart home',
      'graphene','graphene','graphene','graphene']

dic4=['글로벌 5G 회의', '글로벌 5G 회의', '글로벌 5G 회의', '글로벌 5G 회의',
      '스마트카 응용대회', '스마트카 응용대회', '스마트카 응용대회', '스마트카 응용대회',
      '내년 새로운 전략 계획', '내년 새로운 전략 계획', '내년 새로운 전략 계획', '내년 새로운 전략 계획',
      '그래핀 촉매제 개발', '그래핀 촉매제 개발', '그래핀 촉매제 개발', '그래핀 촉매제 개발']

dic4=['世界5G大会下周将开', '世界5G大会下周将开', '世界5G大会下周将开', '世界5G大会下周将开',
      '车联网技术与应用大会', '车联网技术与应用大会', '车联网技术与应用大会', '车联网技术与应用大会',
      '明年新战略纷纷出炉', '明年新战略纷纷出炉', '明年新战略纷纷出炉', '明年新战略纷纷出炉',
      '科学家研发石墨烯钛催', '科学家研发石墨烯钛催', '科学家研发石墨烯钛催', '科学家研发石墨烯钛催']

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
#zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simfang.ttf')
for i in range(8,12):
    a=web.DataReader(dic1[i][list(dic1[i].keys())[0]],'yahoo',datetime.datetime(2019, 6, 12), datetime.datetime.today())
    b=np.array(a['Adj Close']).reshape(-1,1)
    min_max_scaler = MinMaxScaler()
    l = min_max_scaler.fit_transform(b)
    plt.plot(list(a.index), l, label=list(dic1[i].keys())[0])
    plt.title(dic3[i],size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.text(x=dic2[i]['date']-datetime.timedelta(days=16),y=1.068,s=dic4[i],size=20,color='red')
    plt.legend(prop={'size': 15})
    plt.axvline(x = dic2[i]['date'], color = 'm')
    plt.grid(color='#BDBDBD', linestyle='--', linewidth=0.5)
plt.show()

#################################################################################
dic5=[{"KMW(5G)":"032500"},
      {"삼성전자(5G)":"005930"},
      {"코위버(5G)":"056360"},
      {"대한광통신(5G)":"010170"},
      {"DB 히텍(smart car)":"000990"},
      {"LG 이노텍(smart car)":"011070"},
      {"만도(smart car)":"204320"},
      {"모바일어플라이언스(smart car)":"087260"},
      {"상지카일룸(smart home)":"042940"},
      {"옴니시스템(smart home)":"057540"},
      {"코맥스(smart home)":"036690"},
      {"코콤(smart home)":"015710"},
      {"국일제지(graphene)":"078130"},
      {"아이컴포넌트(graphene)":"059100"},
      {"상보(graphene)":"027580"},
      {"크리스탈신소재(graphene)":"900250"}]

dic6=[{'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 11, 18)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 9)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)},
      {'date':datetime.datetime(2019, 12, 5)}]

font_name=font_manager.FontProperties(fname='c:/windows/fonts/malgun.ttf').get_name()
rc('font',family=font_name)
for i in range(12,16):
    a=fdr.DataReader(dic5[i][list(dic5[i].keys())[0]], datetime.datetime(2019, 6, 12), datetime.datetime.today())["Close"]
    b = np.array(a).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    l = min_max_scaler.fit_transform(b)
    plt.plot(a.index, l, label=list(dic5[i].keys())[0])
    plt.title(dic3[i],size=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.text(x=dic6[i]['date']-datetime.timedelta(days=16),y=1.068,s=dic4[i],size=20,color='red')
    plt.legend(prop={'size': 15})
    plt.axvline(x = dic6[i]['date'], color = 'm')
    plt.grid(color='#BDBDBD', linestyle='--', linewidth=0.5)
plt.show()

k=fdr.DataReader(dic5[0][list(dic5[0].keys())[0]], datetime.datetime(2019, 6, 12), datetime.datetime.today())["Close"]
d=DataFrame(np.ones((128,1)),index=list(k.index))
col=[]
for i in range(12,16):
    a=fdr.DataReader(dic5[i][list(dic5[i].keys())[0]], datetime.datetime(2019, 6, 12), datetime.datetime.today())["Close"]
    col.append(list(dic5[i].keys())[0])
    b=web.DataReader(dic1[i][list(dic1[i].keys())[0]],'yahoo',datetime.datetime(2019, 6, 12), datetime.datetime.today())['Adj Close']
    col.append(list(dic1[i].keys())[0])
    c=pd.merge(DataFrame(a),DataFrame(b),left_index=True, right_index=True)
    d=pd.merge(DataFrame(c),DataFrame(d),left_index=True, right_index=True)
d=d.iloc[:,:-1]
d.columns=col
ll=d.corr()
sns.heatmap(data=ll[ll>=0.5],annot=True,fmt = '.2f', cmap='Blues')

#########################중국과 한국
a = web.DataReader("603023.SS", 'yahoo', datetime.datetime(2019, 6, 12), datetime.datetime.today())['Adj Close']
b = np.array(a).reshape(-1, 1)
min_max_scaler = MinMaxScaler()
l = min_max_scaler.fit_transform(b)
plt.plot(a.index, l)
plt.title("威帝股份(smart car)", size=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.axvline(x=datetime.datetime(2019, 12, 5), color='m')
plt.text(x=datetime.datetime(2019, 12, 5) - datetime.timedelta(days=16), y= 1.068,s='车联网技术与应用大会', size=20, color='red')
plt.grid(color='#BDBDBD', linestyle='--', linewidth=0.5)

dic4=['世界5G大会下周将开', '世界5G大会下周将开', '世界5G大会下周将开', '世界5G大会下周将开',
      '车联网技术与应用大会', '车联网技术与应用大会', '车联网技术与应用大会', '车联网技术与应用大会',
      '明年新战略纷纷出炉', '明年新战略纷纷出炉', '明年新战略纷纷出炉', '明年新战略纷纷出炉',
      '科学家研发石墨烯钛催', '科学家研发石墨烯钛催', '科学家研发石墨烯钛催', '科学家研发石墨烯钛催']

dic4=['글로벌 5G 회의', '글로벌 5G 회의', '글로벌 5G 회의', '글로벌 5G 회의',
      '스마트카 응용대회', '스마트카 응용대회', '스마트카 응용대회', '스마트카 응용대회',
      '내년 새로운 전략 계획', '내년 새로운 전략 계획', '내년 새로운 전략 계획', '내년 새로운 전략 계획',
      '그래핀 촉매제 개발', '그래핀 촉매제 개발', '그래핀 촉매제 개발', '그래핀 촉매제 개발']

a = fdr.DataReader('057540', datetime.datetime(2019, 6, 12), datetime.datetime.today())["Close"]
b = np.array(a).reshape(-1, 1)
min_max_scaler = MinMaxScaler()
l = min_max_scaler.fit_transform(b)
plt.plot(a.index, l)
plt.title("옴니시스템(smart home)", size=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.axvline(x=datetime.datetime(2019, 12, 9), color='m')
plt.text(x=datetime.datetime(2019, 12, 9) - datetime.timedelta(days=16), y= 1.068,s='내년 새로운 전략 계획', size=20, color='red')
plt.grid(color='#BDBDBD', linestyle='--', linewidth=0.5)

dic1=[{"永鼎股份(5G)":"600105.SS"},
      {"大唐电信(5G)":"600198.SS"},
      {"*ST信威(5G)":"600485.SS"},
      {"宜安科技(5G)":"300328.SZ"},
      {"永鼎股份(smart car)":"600105.SS"},
      {"威帝股份(smart car)":"603023.SS"},
      {"奋达科技(smart car)":"002681.SZ"},
      {"日上集团(smart car)":"002593.SZ"},
      {"海尔智家(smart home)":"600690.SS"},
      {"巨星科技(smart home)":"002444.SZ"},
      {"全志科技(smart home)":"300458.SZ"},
      {"梦百合(smart home)":"603313.SS"},
      {"德尔未来(graphene)":"002631.SZ"},
      {"中科电气(graphene)":"300035.SZ"},
      {"长信科技(graphene)":"300088.SZ"},
      {"华丽家族(graphene)":"600503.SS"}]

dic5=[{"KMW(5G)":"032500"},
      {"삼성전자(5G)":"005930"},
      {"코위버(5G)":"056360"},
      {"대한광통신(5G)":"010170"},
      {"DB 히텍(smart car)":"000990"},
      {"LG 이노텍(smart car)":"011070"},
      {"만도(smart car)":"204320"},
      {"모바일어플라이언스(smart car)":"087260"},
      {"상지카일룸(smart home)":"042940"},
      {"옴니시스템(smart home)":"057540"},
      {"코맥스(smart home)":"036690"},
      {"코콤(smart home)":"015710"},
      {"국일제지(graphene)":"078130"},
      {"아이컴포넌트(graphene)":"059100"},
      {"상보(graphene)":"027580"},
      {"크리스탈신소재(graphene)":"900250"}]

a = fdr.DataReader('127710', datetime.datetime(2019, 6, 12), datetime.datetime.today())["Close"]
plt.plot(a.index, a)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.grid(color='#BDBDBD', linestyle='--', linewidth=0.5)


#################################################################
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:01:59 2019

@author: LG
"""

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

font_path = "c:/windows/fonts/malgun.ttf"
font_prop1 = font_manager.FontProperties(fname=font_path, size=12)
font_prop2 = font_manager.FontProperties(fname=font_path, size=14)

#########################종목 코드의 이름 뽑아내기
name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:, ['Symbol', 'Name']]


#########################zipline 함수

# 초기화 함수 : 날짜, 종목symbol, 수수료
def initialize(context):
    context.i = 0
    context.sym = symbol('AAPL')
    context.set_commission(commission.PerDollar(cost=0.00165))


# 거래함수
def handle_data(context, data):
    # 초기에 100주 매수
    if context.i == 0:
        order(context.sym, round(1000 / len(codes)))

    context.i += 1

    # 현재 주가
    now = data.current(context.sym, 'price')

    record(AAPL=now)


#########################타겟 종목
# 춘일 테마주
codes = ['010170']
codes = ['059100', '027580']
codes = ['087260']
codes = ['057540']

#########################날짜 인덱스 가져오기

# APPL 날짜 인덱스 가져오기
start_a = datetime.datetime(2018, 12, 1)
end_a = datetime.datetime(2019, 12, 1)
d_target = "AAPL"
d_data = fdr.DataReader(d_target, start_a, end_a)
d_data = d_data[['Close']]
d_data.columns = ['AAPL']
d_data = d_data.tz_localize('UTC')

#########################zipline 알고리즘

table = DataFrame()
for i in range(len(codes)):
    code = codes[i]
    start = datetime.datetime(2018, 12, 1)
    end = datetime.datetime(2019, 12, 1)
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
    algo = TradingAlgorithm(sim_params=create_simulation_parameters(capital_base=round(10000000 / len(codes))),
                            initialize=initialize, handle_data=handle_data)
    result = algo.run(data3)

    # 결과 테이블
    table = pd.concat([table, DataFrame(result['portfolio_value'])], axis=1)

# 마지막 컬럼에 평균(최종) 수익률 추가
table = pd.concat([table, table.sum(axis=1)], axis=1)

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
print('초기금액 : 10000000원')
print('종료금액 : {}원'.format(ends))
print('수익 : {}원'.format(ends - 10000000))
print('수익률 : {}%'.format(round((ends - 10000000) / 10000000 * 100, 2)))

5g
일주일:-3.01%
1달:-0.31%
3달:-0.71%
6달:-7.51%
1년:-18.61%

graphene
일주일:0.23%
1달:-2.52%
3달:-3.44%
6달:0.95%
1년:-10.36%

smart car
일주일:-3.71%
1달:-0.41
3달:3.69%
6달:9.99%
1년:-6.01%

smart home
일주일:-1.04%
1달:-2.99%
3달:0.31%
6달:1.56%
12달:-0.24%

###########################################
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:11:20 2019

@author: LG
"""

import FinanceDataReader as fdr
import datetime
from zipline.api import record, symbol, order
from zipline.algorithm import TradingAlgorithm
from zipline.finance import commission
from zipline.utils.factory import create_simulation_parameters
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.float_format', '{:3f}'.format)
from matplotlib import font_manager, rc

font_path = "c:/windows/fonts/malgun.ttf"
font_prop1 = font_manager.FontProperties(fname=font_path, size=12)
font_prop2 = font_manager.FontProperties(fname=font_path, size=14)

#########################날짜 인덱스 가져오기

# APPL 날짜 인덱스 가져오기
start_a = datetime.datetime(2018, 12, 1)
end_a = datetime.datetime(2019, 12, 31)
d_target = "AAPL"
d_data = fdr.DataReader(d_target, start_a, end_a)
d_data = d_data[['Close']]
d_data.columns = ['AAPL']
d_data = d_data.tz_localize('UTC')


#########################zipline 함수

# 초기화 함수 : 날짜, 종목symbol, 현재/남은 주식개수, 수수료, 주식보유여부
def initialize(context):
    context.i = 0
    context.sym = symbol('AAPL')
    context.stock = 0
    context.rs = round(1000 / len(codes))
    context.set_commission(commission.PerDollar(cost=0.00165))
    context.hold = False


# 거래함수
def handle_data(context, data):
    # 날짜
    context.i += 1

    # 초기에 50주 매수
    if context.i == 1:
        order(context.sym, round(500 / len(codes)))
        context.stock += round(500 / len(codes))
        context.rs -= round(500 / len(codes))
        context.hold = True

    # 해당 날짜 주가
    now = data.current(context.sym, 'price')

    # 수익률>=15%이면 매도
    if now >= fst * 1.15 and context.hold:
        order(context.sym, -context.stock)
        context.stock -= context.stock
        context.rs = 0
        context.hold = False

    # 매수
    if now <= fst * 0.95 and 0 < context.stock < round(1000 / len(codes)):
        if now <= fst * 0.7:
            orders = round(300 / len(codes)) if context.rs >= round(300 / len(codes)) else context.rs
        elif now <= fst * 0.75:
            orders = round(250 / len(codes)) if context.rs >= round(250 / len(codes)) else context.rs
        elif now <= fst * 0.8:
            orders = round(200 / len(codes)) if context.rs >= round(200 / len(codes)) else context.rs
        elif now <= fst * 0.85:
            orders = round(150 / len(codes)) if context.rs >= round(150 / len(codes)) else context.rs
        elif now <= fst * 0.9:
            orders = round(100 / len(codes)) if context.rs >= round(100 / len(codes)) else context.rs
        else:
            orders = round(50 / len(codes)) if context.rs >= round(50 / len(codes)) else context.rs
        order(context.sym, orders)
        context.stock += orders
        context.rs -= orders

    record(AAPL=now, stock=context.stock, order=context.rs)


#########################타겟 종목

# codes = ['039010','007700','192440','094820','120240','102710']
codes = ['004490', '039010', '192440', '015230', '010240', '282880', '013120', '045100', '094820', '014530', '236200',
         '052330', '017890', '036670', '102710']

#########################zipline 알고리즘

table = DataFrame()
for i in range(len(codes)):
    code = codes[i]
    target = "target"
    start = datetime.datetime(2018, 12, 1)
    end = datetime.datetime(2019, 12, 31)
    data = fdr.DataReader(code, start, end)
    data = data[['Close']]
    data.columns = [target]
    data = data.tz_localize('UTC')
    fst = data.iloc[0][0]
    df = DataFrame(data)

    # APPL 날짜 인덱스 + 타겟 데이터
    data3 = d_data[len(d_data) - len(df):]
    data3['AAPL'] = np.where(1, df['target'], df['target'])

    # 알고리즘 수행
    algo = TradingAlgorithm(sim_params=create_simulation_parameters(capital_base=round(10000000 / len(codes))),
                            initialize=initialize, handle_data=handle_data)
    result = algo.run(data3)

    # 결과 테이블
    table = pd.concat([table, DataFrame(result['portfolio_value'])], axis=1)

# 마지막에 컬럼에 평균(최종) 수익률 추가
table = pd.concat([table, table.sum(axis=1)], axis=1)

#########################종목 코드의 이름 뽑아내기

name_list = fdr.StockListing('KRX')
name_list = name_list.loc[:, ['Symbol', 'Name']]

# 입력한 종목의 코드, 이름
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
print('초기금액 : 10000000원')
print('종료금액 : {}원'.format(ends))
print('수익금액 : {}원'.format(ends - 10000000))
print('수익률 : {}%'.format(round((ends - 10000000) / 10000000 * 100, 2)))

# 모두 합친 그래프
ax = plt.subplot()
plt.style.use('seaborn-dark')
plt.plot(table.index, table.iloc[:, :-1] / 10000)
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.xlabel('날짜', fontproperties=font_prop1)
plt.ylabel('금액(만원)', fontproperties=font_prop1)
plt.ticklabel_format(style='plain', axis='y')
plt.legend(table.columns[:-1], loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.title('전략1 종목별 수익', fontproperties=font_prop2)
plt.show()

# 최종 수익률 그래프
plt.style.use('seaborn-dark')
plt.plot(table.index, table.iloc[:, -1] / 10000)
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.xlabel('날짜', fontproperties=font_prop1)
plt.ylabel('금액(만원)', fontproperties=font_prop1)
plt.ticklabel_format(style='plain', axis='y')
plt.title('전략1 총수익', fontproperties=font_prop2)
plt.show()
