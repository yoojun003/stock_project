import urllib.parse
from selenium import webdriver
import time
import pandas as pd
import copy
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from matplotlib import pyplot as plt
import numpy as np

MARKET_CODE_DICT = {
    'kospi': 'stockMkt',
    'kosdaq': 'kosdaqMkt',
    'konex': 'konexMkt'
}

DOWNLOAD_URL = 'kind.krx.co.kr/corpgeneral/corpList.do'

def download_stock_codes(market=None, delisted=False):
    params = {'method': 'download'}

    if market.lower() in MARKET_CODE_DICT:
        params['marketType'] = MARKET_CODE_DICT[market]

    if not delisted:
        params['searchType'] = 13

    params_string = urllib.parse.urlencode(params)
    request_url = urllib.parse.urlunsplit(['http', DOWNLOAD_URL, '', params_string, ''])

    df = pd.read_html(request_url, header=0)[0]
    df.종목코드 = df.종목코드.map('{:06d}'.format)
    return df


def stock_li_crawling(stock):
    all_stock_li = []
    cnt = 0
    browser = webdriver.Chrome('C:/Users/KWON/Desktop/TeamProject/data/chromedriver.exe')
    for code in stock['종목코드']:
        url = 'https://navercomp.wisereport.co.kr/v2/company/c1030001.aspx?cmp_cd=' + code + '&cn='
        browser.get(url)
        browser.implicitly_wait(5)

        a = browser.find_elements_by_class_name('lvl1')
        b = [i.find_elements_by_tag_name('td') for i in a]
        stock_li = [x.text for i in range(len(b)) for x in b[i] if b[i][0].text == '펼치기 총포괄이익' if x.text != '펼치기 총포괄이익']

        all_stock_li.append([code, stock_li])

        cnt += 1
        if cnt % 50 == 0:
            time.sleep(1)
    browser.close()
    return all_stock_li


def re_list(stock_data):
    a = copy.deepcopy(stock_data)
    for i in range(len(a)):
        del a[i][1][-3::2]
    a = pd.DataFrame([(i[1]) for i in a], index=[i[0] for i in a])
    a.columns=[['2014','2015','2016','2017','2018','전년대비']]
    return a


def bluechip(data):
    cnt = 0
    dellist = []
    for x in range(len(data)):
        for y in data.iloc[x,:-1]:
            if y == None:
                dellist.append(data.iloc[x].name)
                break
            elif (y == ' ') | ('-' in y):
                dellist.append(data.iloc[x].name)
                cnt += 1
                break
    result_data = data.drop(dellist)
    print(cnt,' 개 종목 삭제됨')
    return result_data


def spell_check(data):
    a = copy.deepcopy(data)
    for x in range(len(a)):
        for i,y in enumerate(a.iloc[x]):
            if ',' in y:
                a.iloc[x,i] = y.replace(',','')
    return a


def surplus(data, y_cnt, r_rate):
    dellist = []
    for i in range(len(data)):
        cnt = 3
        for a in data.iloc[i,5:5-y_cnt:-1]:
            if a / data.iloc[i,6-cnt] * 100 - 100 <= r_rate:
                dellist.append(data.iloc[i].name)
                break
            else:
                cnt += 1
                continue
    print(len(dellist),' 개 종목이 삭제됩니다.')
    result_data = data.drop(dellist)
    return result_data



# 실행 영역
## 종목코드 추출
kospi_code = download_stock_codes('kospi')
kospi_code.head()
kospi_code_list = kospi_code[:][['회사명','종목코드']]

kosdaq_code = download_stock_codes('kosdaq')
kosdaq_code.head()
kosdaq_code_list = kosdaq_code[:][['회사명','종목코드']]

## 년도별 총 포괄이익 스크래핑
kospi_stock_list = stock_li_crawling(kospi_code_list) # kospi 종목에 따른 총포괄이익추출
kosdaq_stock_list = stock_li_crawling(kosdaq_code_list) # kosdac 종목에 따른 총포괄이익추출

len(kospi_stock_list)
len(kosdaq_stock_list)

## 데이터 정제
kospi_stock_data = re_list(kospi_stock_list)
kosdaq_stock_data = re_list(kosdaq_stock_list)

## 우량주 목록 선별, 적자기업 빼기
kospi_blue = bluechip(kospi_stock_data)
kosdaq_blue = bluechip(kosdaq_stock_data)

## 데이터들을 숫자형으로 정제 후, 성장형 우량주 선별
s_kospi = spell_check(kospi_blue)
s_kosdaq = spell_check(kosdaq_blue)

s_kospi2 = s_kospi.iloc[:,:5].astype('float')
s_kosdaq2 = s_kosdaq.iloc[:,:5].astype('float')
s_kospi2.iloc[0][4] - s_kospi2.iloc[0][3] <= 0
s_kosdaq2

### 3년간 흑자와 성장, 흑자폭 성장률 50% 이상
kospi_3_sp = surplus(s_kospi2,3,50)
kosdaq_3_sp = surplus(s_kosdaq2,3,50)

len(kospi_3_sp)
len(kosdaq_3_sp)

kospi_3_sp.index
kosdaq_3_sp.index

### 4년간 흑자와 성장, 흑자폭 성장률 50% 이상
kospi_4_sp = surplus(s_kospi2,4,50)
kosdaq_4_sp = surplus(s_kosdaq2,4,50)

len(kospi_4_sp)
len(kosdaq_4_sp)

## 심리적 저점구간 그래프 확인

kospi_3_sp.index
kosdaq_3_sp.index

kospi_4_sp.index
kosdaq_4_sp.index

code_name = '000060'
test = fdr.DataReader(code_name,'2016')
test_close = test['Close'][:-1]

max_point = test_close[test_close == max(test_close)] # 최대점, index는 날짜
l_point = test_close[test_close == min(test_close)] # 최저점

for i in range(len(test_close)):
    if test_close.index[i] < l_point.index[0]:
        continue
    elif test_close.index[i] == l_point.index[0]:
        v_gradient = -1; sw = 0; c_point = l_point[0]
        continue

    if (sw == 0) & (c_point < test_close[i]):
        sw = 1
    elif (sw == 1) & (c_point > test_close[i]):
        sw = 0
        tmp = (c_point-l_point[0]) / ((test_close.index[i-1]-l_point.index[0]).days)
        if v_gradient == -1:
            v_gradient = tmp
            min_point = test_close[test_close==c_point]
        elif (v_gradient > 0) & (v_gradient >= tmp):
            v_gradient = tmp
            min_point = test_close[test_close==c_point]
        else:
            pass
    c_point = test_close[i]

v_gradient # 지지선 기울기
min_point # 지지선의 한 점 index = 날짜, 값 = 주가
l_point # 지지선의 최 저점 index = 날짜, 값 = 주가
sw # 스위치 0 일경우 주가 상승, 1 일 경우 주가 하락

# 그래프 그리기
start_p = l_point[0] - ((l_point.index[0]-test_close[:1].index[0]).days) * v_gradient # 지지선 시작점 설정
end_p  = ((test_close[-1:].index[0]-l_point.index[0]).days) * v_gradient + l_point[0] # 지지선 끝점 설정

x_sup = [test_close[:1].index[0], test_close[-1:].index[0]] # 지지선 x축
y_sup = [start_p, end_p] # 지지선 y축

max_plt = [max_point[0] for i in range(len(test_close.index))] # 최고점 직선 설정

start_30_sup = round(np.percentile(np.arange(start_p ,max_point[0]),30))
end_30_sup = round(np.percentile(np.arange(end_p ,max_point[0]),30))
y_30_sup = [start_30_sup, end_30_sup]

test_close.plot() # 주가 그래프
plt.plot(x_sup,y_sup) # 지지선 그래프
plt.plot(x_sup,y_30_sup) # 지지선과 최고점 직선 사이의 백분율 30% 직선, 이 작선과 지지선 사이가 심리적 저점 구간
plt.plot(test_close.index,max_plt) # 최고점 직선 그래프









# 내일

## 그래서 매수 타이밍인가? (back_test를 위한 기간 설정)
test_date = datetime(2018,12,3)
test_data = test_close[test_close.index == test_date]
com_low = ((test_data.index[0]-l_point.index[0]).days) * v_gradient + l_point[0]  # 해당 날짜의 최저가 지지선 구간
com_30 = ((test_data.index[0]-l_point.index[0]).days) * gradient_30_sup + l_30_sup # 해당 날짜의 백분율 30% 구간
### 심리적 저점구간에 있을 경우 매수한다. 실전에서는 데이터를 모아 주가 그래프를 보며 today로 실행해볼것.
if (test_data[0] >= com_low) & (test_data[0] <= com_30):
    print('해당 종목 코드 : ',code_name,' 종목 매수 추천')
else:
    print('해당 종목 코드 : ',code_name,' 종목 매수 비추천')