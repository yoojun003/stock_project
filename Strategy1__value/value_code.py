# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""

#필요 라이브러리
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request as req
import numpy as np 

#저평가주 종목코드 반환함수(네이버크롤링)
#함수명 : Bluechip_Classifier
#매개변수명
#CODE : 종목코드
def Bluechip_Classifier(CODE):
    err_wtable = []; err_etable = []; blue_1 = []
    for code in CODE:
        url = "https://finance.naver.com/item/main.nhn?code=" + code
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        if len(soup.select('div.section.cop_analysis div.sub_section'))==0: # 재무제표가 없을 때
            err_wtable.append(code)
            continue
        fs_html = soup.select('div.section.cop_analysis div.sub_section')[0]
        fs_date = [item.get_text().strip() for item in fs_html.select('thead th')] # 날짜 뽑기(열)
        if len(fs_date) != 23: # 재무제표가 비어있을 때
            err_etable.append(code)
            continue
        fs_data = [item.get_text().strip().replace(',','') for item in fs_html.select('td')]  # 수치 추출(내용물) # 쉼표 때문에 나중에 float으로 안변하니까 쉼표 제거
        for i in range(len(fs_data)):
            if fs_data[i] == '-' or fs_data[i] == '':
                fs_data[i] = np.nan
        for i in range(len(fs_data)):
            fs_data[i] = float(fs_data[i])            
        
        s = pd.Series(fs_data[:10]) # 매출액
        s32 = 100 * (s[1] - s[0])/s[0]
        s21 = 100 * (s[2] - s[1])/s[1]
        s10 = 100 * (s[3] - s[2])/s[2] if pd.isnull(s[3]) == False else s21
        op = pd.Series(fs_data[10:20]) # 영업이익
        op = op.loc[op.notnull() ,]
        roe = pd.Series(fs_data[50:60]) # ROE
        roe = roe.loc[roe.notnull() ,]
        dr = pd.Series(fs_data[60:70]) # 부채비율
        dr = dr.loc[dr.notnull() ,]
        qr = pd.Series(fs_data[70:80]) # 당좌비율
        qr = qr.loc[qr.notnull() ,]
        rr = pd.Series(fs_data[80:90]) # 유보율
        rr = rr.loc[rr.notnull() ,] 
        eps = pd.Series(fs_data[90:100]) # EPS
        exp_eps = eps[6:9].mean()*4 if pd.isnull(eps[3]) == True else eps[3]  
        exp1 = exp_eps * 10 # 공식 1
        exp3 = s10 * exp_eps # 공식 3 14.19 * 3655
        pbr = pd.Series(fs_data[120:130]) # EPS
        pbr = pbr.loc[pbr.notnull() ,] 
        stock = soup.select('div.rate_info > div.today > p.no_today span.blind') # 현재주가
        stock = float(stock[0].text.replace(',',''))
        min_exp = min(exp1, exp3) # 공식 중 최저
        if (s32>=5) & (s21>=5) & (s10>=5) & (op>0).all() & (roe>=5).all() & (dr<=200).all() & (qr>=100).all() & (rr>=500).all() & (pbr<6).all() & (min_exp > stock):
            blue_1.append(code)
    return blue_1

#유하게 뽑기

    err_wtable = []; err_etable = []; blue_1 = []
    for F,code in enumerate(b):
        url = "https://finance.naver.com/item/main.nhn?code=" + code
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        fs_html = soup.select('div.section.cop_analysis div.sub_section')[0]
        fs_date = [item.get_text().strip() for item in fs_html.select('thead th')] # 날짜 뽑기(열)
        fs_data = [item.get_text().strip().replace(',','') for item in fs_html.select('td')]  # 수치 추출(내용물) # 쉼표 때문에 나중에 float으로 안변하니까 쉼표 제거
        for i in range(len(fs_data)):
            if fs_data[i] == '-' or fs_data[i] == '':
                fs_data[i] = np.nan
        for i in range(len(fs_data)):
            fs_data[i] = float(fs_data[i])            
        op
        s = pd.Series(fs_data[:4]) # 매출액
        s32 = 100 * (s[1] - s[0])/s[0]
        s21 = 100 * (s[2] - s[1])/s[1]
        s10 = 100 * (s[3] - s[2])/s[2] if pd.isnull(s[3]) == False else s21
        op = pd.Series(fs_data[10:14]) # 영업이익
        op = op.loc[op.notnull() ,]
        roe = pd.Series(fs_data[50:54]) # ROE
        roe = roe.loc[roe.notnull() ,]
        dr = pd.Series(fs_data[60:64]) # 부채비율
        dr = dr.loc[dr.notnull() ,]
        qr = pd.Series(fs_data[70:74]) # 당좌비율
        qr = qr.loc[qr.notnull() ,]
        rr = pd.Series(fs_data[80:84]) # 유보율
        rr = rr.loc[rr.notnull() ,] 
        eps = pd.Series(fs_data[90:94]) # EPS
        exp_eps = eps[6:9].mean()*4 if pd.isnull(eps[3]) == True else eps[3]  
        exp1 = exp_eps * 10 # 공식 1
        exp3 = s10 * exp_eps # 공식 3 14.19 * 3655
        pbr = pd.Series(fs_data[120:124]) # EPS
        pbr = pbr.loc[pbr.notnull() ,] 
        stock = soup.select('div.rate_info > div.today > p.no_today span.blind') # 현재주가
        stock = float(stock[0].text.replace(',',''))
        min_exp = min(exp1, exp3) # 공식 중 최저
        if (s32>=5) & (s21>=5) & (s10>=5) & (op>0).all() & (roe>=5).all() & (dr<=200).all() & (qr>=100).all() & (rr>=500).all() & (pbr<6).all(): #& (min_exp > stock):
            blue_1.append(code)
        print(F)


#종목코드만 넣으면 테이블을 만들어 주는 함수
#함수명 :make_table
#매개변수 
#CODE : 종목코드
#필요 라이브러리
from fs4 import BeautifulSoup
import urllib.request as req
import pandas as pd

def make_table(CODE):
    s_name = []; s_code = [];s_wics = [];s_date = [];s_sort = []
    for code in CODE: 
        url = "https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd=" + code + "&cn="
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        s_name.append(soup.select("span.name")[0].text) # 회사명 넣기
        s_code.append(code) # 코드 넣기
        s_wics.append(soup.select("dt.line-left")[9].text[7:]) # 업종 넣기
        s_date.append(soup.select("dt.right")[1].text[:2]) # 결산월 넣기
        s_sort.append(soup.select("dt.line-left")[8].text[0:6]) # 시장 넣기
    stock = pd.DataFrame({'회사명':s_name, '종목코드':s_code, '업종':s_wics, '결산월':s_date, '시장':s_sort})
    stock['시장'] = ['KOSPI' if stock.시장[i][0:5] == 'KOSPI' else 'KOSDAQ' for i in range(len(stock))]
    return stock


#여러 지표로 저평가 기업 종목코드 반환하는 함수
#함수명 
#Bluechip_Classifier_adj

#매개변수명
#CODE : 종목코드 
# S32 : 3년 전과 2년 전 사이 매출액상승률.입력예시)  해당 기간 매출성장률 5% 이상 s32 >= 5
# S21 : 2년 전과 1년 전 사이 매출액상승률. 입력예시) s21 >= 5
# S10 : 1년 전과 현재 사이 매출액상승률.  입력예시)  s10 >= 5
# OP : 영업이익. 입력예시) 영업이익 5억 이상 op >= 5
# ROE : 자기자본이익률. 입력예시) 자기자본이익률 5% 이상 roe >= 5
# DR : 부채비율. 입력예시) 부채비율 200% 이하 dr <= 200
# QR : 당좌비율. 입력예시) 당좌비율 100% 이상 qr >= 100
# RR : 유보율. 입력예시) 유보율 500% 이상 rr >= 500
# PBR : 입력예시) PBR 6배 미만 pbr < 6

# 필요 라이브러리
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request as req
import numpy as np
 
# 입력 예시 : Bluechip_Classifier_adj(lst,s32>=5,s21>=5,s10>=5,op>0,roe>=5,dr<=200,qr>=100,rr>=500,pbr<6)

def Bluechip_Classifier_adj(CODE,S32,S21,S10,OP,ROE,DR,QR,RR,PBR):
    err_wtable = []; err_etable = []; blue_1 = []
    for code in CODE:
        url = "https://finance.naver.com/item/main.nhn?code=" + code
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        if len(soup.select('div.section.cop_analysis div.sub_section'))==0: # 재무제표가 없을 때
            err_wtable.append(code)
            continue
        fs_html = soup.select('div.section.cop_analysis div.sub_section')[0]
        fs_date = [item.get_text().strip() for item in fs_html.select('thead th')] # 날짜 뽑기(열)
        if len(fs_date) != 23: # 재무제표가 비어있을 때
            err_etable.append(code)
            continue
        fs_data = [item.get_text().strip().replace(',','') for item in fs_html.select('td')]  # 수치 추출(내용물) # 쉼표 때문에 나중에 float으로 안변하니까 쉼표 제거
        for i in range(len(fs_data)):
            if fs_data[i] == '-' or fs_data[i] == '':
                fs_data[i] = np.nan
        for i in range(len(fs_data)):
            fs_data[i] = float(fs_data[i])            
        
        s = pd.Series(fs_data[:10]) # 매출액
        s32 = 100 * (s[1] - s[0])/s[0]
        s21 = 100 * (s[2] - s[1])/s[1]
        s10 = 100 * (s[3] - s[2])/s[2] if pd.isnull(s[3]) == False else s21
        op = pd.Series(fs_data[10:20]) # 영업이익
        op = op.loc[op.notnull() ,]
        roe = pd.Series(fs_data[50:60]) # ROE
        roe = roe.loc[roe.notnull() ,]
        dr = pd.Series(fs_data[60:70]) # 부채비율
        dr = dr.loc[dr.notnull() ,]
        qr = pd.Series(fs_data[70:80]) # 당좌비율
        qr = qr.loc[qr.notnull() ,]
        rr = pd.Series(fs_data[80:90]) # 유보율
        rr = rr.loc[rr.notnull() ,] 
        eps = pd.Series(fs_data[90:100]) # EPS
        exp_eps = eps[6:9].mean()*4 if pd.isnull(eps[3]) == True else eps[3]  
        exp1 = exp_eps * 10 # 공식 1
        exp3 = s10 * exp_eps # 공식 3 14.19 * 3655
        pbr = pd.Series(fs_data[120:130]) # EPS
        pbr = pbr.loc[pbr.notnull() ,] 
        stock = soup.select('div.rate_info > div.today > p.no_today span.blind') # 현재주가
        stock = float(stock[0].text.replace(',',''))
        min_exp = min(exp1, exp3) # 공식 중 최저
        if (S32) & (S21) & (S10) & (OP).all() & (ROE).all() & (DR).all() & (QR).all() & (RR).all() & (PBR).all() & (min_exp > stock):
            blue_1.append(code)
    return blue_1















#악재 검색코드
#종목코드를 투입하면 금융감독원 전자공시시스템에 접속하여 10가지 
#악재가 단 하나도 포함되지 않는 종목코드를 리턴하는 함수

# 함수명
# BadNews_Eraser
# 매개변수
# CODE : 종목코드

# 필요 라이브러리
pip install selenium
from selenium import webdriver
import time

def BadNews_Finder(CODE):
    badnews = ['유상증자','감자','신주인수권부사채','전환사채','교환사채','불성실', '관리종목', '비적정', '횡령', '배임']
    dart = "http://dart.fss.or.kr/dsab002/main.do"
    driver = webdriver.Chrome("c:/data/chromedriver.exe")
    driver.get(dart)
    time.sleep(2)
    fail = []
    pass_ = []
    for code in CODE:
        driver.find_element_by_id('textCrpNm').clear() # 검색창 빈칸 만들기
        driver.find_element_by_id('textCrpNm').send_keys(code) # 검색창에 종목코드 입력
        driver.find_element_by_xpath('//*[@id="date6"]').click()
        for bn in badnews:
            driver.find_element_by_id('reportName').send_keys(bn) # 상세검색창에 악재 입력
            driver.find_element_by_xpath('//*[@id="searchpng"]').click()
            time.sleep(3)
            # 스크롤링 시작
            req = driver.page_source # 현재 페이지 소스 저장
            soup = BeautifulSoup(req, 'html.parser')
            if (len(soup.select("div .page_list")[0].text.strip()) > 0): #or (len(soup.select("div .page_list")) != 0): # 악재가 검색되면?
                fail.append(code)
            driver.find_element_by_id('reportName').clear()
 return fail

bl = BadNews_Finder(blue_1)
blue = []
for i in blue_1:
    if i not in bl:
        blue.append(i)


# 호재 검색 코드
# 모든 종목코드에서 어떤 악재도 발견되지 않았을 때에는
# TypeError: argument of type 'numpy.bool_' is not iterable 라는 오류 뜬다.
# 멀쩡한데 index 에러가 뜰 때는 초(sec)를 더 늦춰보자

def GoodNews_Searcher(CODE):
    Goodnews = ['무상증자','액면','유상감자','상속','증여','자사주','자산재평가']
    dart = "http://dart.fss.or.kr/dsab002/main.do"
    driver = webdriver.Chrome("c:/data/chromedriver.exe")
    driver.get(dart)
    time.sleep(2)
    fail = []
    for code in CODE:
        driver.find_element_by_id('textCrpNm').clear() # 검색창 빈칸 만들기
        driver.find_element_by_id('textCrpNm').send_keys(code) # 검색창에 종목코드 입력
        driver.find_element_by_xpath('//*[@id="date6"]').click()
        for gn in Goodnews:
            driver.find_element_by_id('reportName').clear()
            driver.find_element_by_id('reportName').send_keys(gn) # 상세검색창에 악재 입력
            driver.find_element_by_xpath('//*[@id="searchpng"]').click()
            time.sleep(3)
            # 스크롤링 시작
            req = driver.page_source # 현재 페이지 소스 저장
            soup = BeautifulSoup(req, 'html.parser')
            if (len(soup.select("div .page_list")[0].text.strip()) > 0): #or (len(soup.select("div .page_list")) != 0): # 악재가 검색되면?
                fail.append(code + ' ' + gn)
    return fail

# ['003000 자산재평가',
#  '101330 무상증자',
#  '015230 무상증자',
#  '010240 무상증자',
#  '155650 무상증자',
#  '264660 무상증자',
#  '104830 무상증자']



# 과거기준으로 뽑기 위한 과정

# 당좌 100 이상 부채 150 이하

pip install -U finance-datareader
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request as req
import numpy as np 
from datetime import datetime as dt
import FinanceDataReader as fdr


def Bluechip_Classifier_p(CODE):
    err_wtable = []; err_etable = []; blue_1 = []; key = []
    for F,code in enumerate(item.종목코드):
        try :
            stock = fdr.DataReader(code, dt(2018,12,1)).ix[0,-3]
        except KeyError:
            continue
        except ValueError:
            continue
        url = "https://finance.naver.com/item/main.nhn?code=" + code
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        if len(soup.select('div.section.cop_analysis div.sub_section'))==0: # 재무제표가 없을 때
            err_wtable.append(code)
            continue
        fs_html = soup.select('div.section.cop_analysis div.sub_section')[0]
        fs_date = [item.get_text().strip() for item in fs_html.select('thead th')] # 날짜 뽑기(열)
        if len(fs_date) != 23: # 재무제표가 비어있을 때
            err_etable.append(code)
            continue
        fs_data = [item.get_text().strip().replace(',','') for item in fs_html.select('td')]  # 수치 추출(내용물) # 쉼표 때문에 나중에 float으로 안변하니까 쉼표 제거
        for i in range(len(fs_data)):
            if fs_data[i] == '-' or fs_data[i] == '':
                fs_data[i] = np.nan
        for i in range(len(fs_data)):
            fs_data[i] = float(fs_data[i])            
        
        s = pd.Series(fs_data[:3]+fs_data[4:6]) # 매출액
        s32 = 100 * (s[1] - s[0])/s[0]
        s21 = 100 * (s[2] - s[1])/s[1]
        op = pd.Series(fs_data[10:15]) # 영업이익
        op = op.loc[op.notnull() ,]
        roe = pd.Series(fs_data[50:53]+fs_data[54:56]) # ROE
        roe = roe.loc[roe.notnull() ,]
        dr = pd.Series(fs_data[60:63]+fs_data[64:66]) # 부채비율
        dr = dr.loc[dr.notnull() ,]
        qr = pd.Series(fs_data[70:73]+fs_data[74:76]) # 당좌비율
        qr = qr.loc[qr.notnull() ,]
        rr = pd.Series(fs_data[80:83]+fs_data[84:86]) # 유보율
        rr = rr.loc[rr.notnull() ,] 
        eps = pd.Series(fs_data[90:93]+fs_data[94:96]) # EPS
        exp_eps = eps[2]  
        exp1 = exp_eps * 10 # 공식 1
        exp3 = s21 * exp_eps # 공식 3 14.19 * 3655
        pbr = pd.Series(fs_data[120:123]+fs_data[124:126]) # EPS
        pbr = pbr.loc[pbr.notnull() ,] 
        min_exp = min(exp1, exp3) # 공식 중 최저
        if (s32>=5) & (s21>=5) & (op>0).all() & (roe>=5).all() & (dr<=150).all() & (qr>=100).all() & (rr>=500).all() & (pbr<6).all() & (min_exp > stock):
            blue_1.append(code)
        print(F,blue_1)
    return blue_1



# 매출액 15-16만 fnguide에서 가져오기

def Bluechip_Classifier_p2(CODE):
    err_wtable = []; err_etable = []; blue_1 = []; key = []; value = []
    for F,code in enumerate(sb):
        url = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=3&gicode=A" + code + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701"
        html = req.urlopen(url)
        soup = BeautifulSoup(html,'html.parser')
        try :
            fs_b = soup.find('body').select('.ul_wrap #highlight_D_Y td')
        except AttributeError:
            continue
        if len(fs_b) == 0:
            continue
        fs_a = []            
        for i in fs_b:
                 fs_a.append(i.text.replace(',','').replace('\xa0','').replace('N/A','').replace('IFRS','').replace('()','').replace('완전잠식',''))
        
        for i in range(len(fs_a)):
            if fs_a[i] == '':
                fs_a[i] = np.nan
        s = pd.Series(fs_a[1:3]).astype(float)
        s10 = 100 * (s[1] - s[0])/s[0]
        if (s10>=5):
            blue_1.append(code)
        print(F,blue_1)


# 백테스팅 표 만들기 (시점)


def BTS(CODE):
    s_name = []; s_code = []; price = []; w1 = []; m1 = []; m3 = []; m6 = []; y1 = []; m = []
    for F,code in enumerate(CODE):
        g = fdr.DataReader(code, dt(2018,12,1),dt(2019,12,9))
        gf = g.iloc[0,-3]
        url = "https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd=" + code + "&cn="
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        s_name.append(soup.select("span.name")[0].text) # 회사명 넣기
        s_code.append(code) # 코드 넣기
        price.append(gf)
        w1.append(str(round(100* (g.ix[4,-3] - gf)/gf,2)))
        m1.append(str(round(100* (g.ix[19,-3] - gf)/gf,2)))
        m3.append(str(round(100* (g.ix[59,-3] - gf)/gf,2)))
        m6.append(str(round(100* (g.ix[119,-3] - gf)/gf,2)))
        y1.append(str(round(100* (g.ix[-1,-3] - gf)/gf,2)))
        m.append(max(float(w1[F]),float(m1[F]),float(m3[F]),float(m6[F]),float(y1[F])))
    maxmin = pd.Series(m).sort_values()[2]
    table = pd.DataFrame({'회사명': s_name , '종목코드': s_code, '1주': w1, '1개월': m1, '3개월': m3, '6개월':m6, '1년':y1, 'MAX':m})
    return table


# 누적?     
       
def BTS_b(CODE):
    s_name = []; s_code = []; price = []; w1 = []; m1 = []; m3 = []; m6 = []; y1 = []; m = []
    for F,code in enumerate(CODE):
        g = fdr.DataReader(code, dt(2018,12,1),dt(2019,12,1))
        gf = g.iloc[0,-3]
        url = "https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd=" + code + "&cn="
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        s_name.append(soup.select("span.name")[0].text) # 회사명 넣기
        s_code.append(code) # 코드 넣기
        price.append(gf)
        w1.append(str(round(100* (max(g.ix[:4,-3]) - gf)/gf,2)))
        m1.append(str(round(100* (max(g.ix[:19,-3]) - gf)/gf,2)))
        m3.append(str(round(100* (max(g.ix[:59,-3]) - gf)/gf,2)))
        m6.append(str(round(100* (max(g.ix[:119,-3]) - gf)/gf,2)))
        y1.append(str(round(100* (max(g.ix[:-1,-3]) - gf)/gf,2)))
        m.append(max(float(w1[F]),float(m1[F]),float(m3[F]),float(m6[F]),float(y1[F])))
    maxmin = pd.Series(m).sort_values()[2]
    table = pd.DataFrame({'회사명': s_name , '종목코드': s_code, '1주': w1, '1개월': m1, '3개월': m3, '6개월':m6, '1년':y1, 'MAX':m})
    return table


# 부채 150 이하 당좌 100 이상
# 과거

# , 회사명,종목코드,1주,1개월,3개월,6개월,1년,MAX
# 0,현대통신,039010,2.29,2.29,5.05,5.05,5.05,5.05
# 1,F&F,007700,3.29,3.29,41.77,94.86,148.97,148.97
# 2,슈피겐코리아,192440,0.0,14.1,43.59,78.42,78.42,78.42
# 3,일진파워,094820,20.99,20.99,28.65,28.65,28.65,28.65
# 4,대정화금,120240,0.75,0.75,2.64,7.17,45.28,45.28
# 5,이엔에프테크놀로지,102710,0.0,0.0,21.17,59.12,92.7,92.70

# past = ['039010', '007700', '192440', '094820', '120240', '102710']

# 현재

# now_1 = ['004490', '039010', '192440', '015230', '010240', '282880', '013120', 
#  '045100', '094820', '014530', '236200', '052330', '017890', '036670', '102710']        

# blacklist = ['900280', '036800', '115310', '208710', '101330', '063760', '088130', '031980',
# '036830', '183300', '269620', '241790', '039440', '011170']






# 종목코드와 저장경로를 넣으면 해당 종목코드의 그래프를 저장하는 함수


# 함수명
# make_gs

# 매개변수명
# CODE : 종목코드
# where : 경로. 입력예시 : c:/data/img/

# 필요 라이브러리

from bs4 import BeautifulSoup
import urllib.request as reqN
import numpy as np
from datetime import datetime as dt
import FinanceDataReader as fdr
from mpl_finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\windows/fonts/malgun.ttf").get_name()
rc('font', family=font_name)

past = ['039010', '007700', '192440', '094820', '120240', '102710']
def make_gs(CODE, where):
    for code in CODE:
        url = "https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd=" + code + "&cn="
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        name = soup.select("span.name")[0].text # 회사명 넣기
        
        data = fdr.DataReader(code, dt(2018,10,1), dt(2019,12,1)).iloc[:,:5]
        idx = data.index.astype(str)

        ma = fdr.DataReader(code, dt(2018,10,1),dt(2019,12,1)).ix[:,-3]
        ma5 = fdr.DataReader(code, dt(2018,9,20),dt(2019,12,1)).ix[:,-3].rolling(window=5).mean()[4:]
        ma20 = fdr.DataReader(code, dt(2018,8,30),dt(2019,12,1)).ix[:,-3].rolling(window=20).mean()[19:]
        ma60 = fdr.DataReader(code, dt(2018,7,4),dt(2019,12,1)).ix[:,-3].rolling(window=60).mean()[59:]
        ma120 = fdr.DataReader(code, dt(2018,4,4),dt(2019,12,1)).ix[:,-3].rolling(window=120).mean()[119:]

        fig, ax = plt.subplots(figsize=(12,7)) # 여기서 차트 크기를 조정 할 수 있습니다.

        # 이동평균선을 차트에 추가 합니다.
        ax.plot(idx, ma5, label='5일', color='green', linewidth=2.5)
        ax.plot(idx, ma20, label='20일', color='red', linewidth=2.5)
        ax.plot(idx, ma60, label='60일', color='orange', linewidth=2.5)
        ax.plot(idx, ma120, label='120일', color='purple', linewidth=2.5)
        
        # 아래 명령어를 통해 시고저종 데이터를 사용하여 캔들 차트를 그립니다.
        candlestick2_ohlc(ax,data['Open'],data['High'], data['Low'],data['Close'],width=0.6, colorup = 'red', colordown ='blue')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        
        # 아래는 날짜 인덱싱을 위한 함수 입니다.
        def mydate(x,pos):
            try:
                return idx[int(x-0.5)]
            except IndexError:
                return ''
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

     
        plt.title(name + ' , ' + code, size = 20)
        plt.xlabel('년/월',fontsize=15)
        plt.ylabel('가격(원)',fontsize=15)
        plt.grid(color='#BDBDBD', linestyle='--', linewidth=0.5 )
        plt.legend(loc='best')
        plt.axvline(x='2018-12-03', color='black')
        fig.autofmt_xdate()
        plt.savefig(where + name + '.png')



# blacklist = ['900280', '036800', '115310', '208710', '101330', '063760', '088130', '031980',
# '036830', '183300', '269620', '241790', '039440', '011170']


# 과거기준으로 하려면?

        ma = fdr.DataReader(code, dt(2017,12,1),dt(2018,12,1)).ix[:,-3]
        ma5 = fdr.DataReader(code, dt(2017,11,27),dt(2018,12,1)).ix[:,-3].rolling(window=5).mean()[4:]
        ma20 = fdr.DataReader(code, dt(2017,11,6),dt(2018,12,1)).ix[:,-3].rolling(window=20).mean()[19:]
        ma60 = fdr.DataReader(code, dt(2017,9,1),dt(2018,12,1)).ix[:,-3].rolling(window=60).mean()[59:]
        ma120 = fdr.DataReader(code, dt(2017,6,8),dt(2018,12,1)).ix[:,-3].rolling(window=120).mean()[119:]








































# 현재주가와 최소적정주가의 차이(%)를 나타내주는 함수

# 함수명 
# nameyet

# 매개변수명
# blue : 종목코드

# 필요 라이브러리
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request as req
import numpy as np 


def nameyet(blue):
    cod = []; nam = []; mi = []; no = []; cha = [];
    for code in blue:
        url = "https://finance.naver.com/item/main.nhn?code=" + code
        html = req.urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        if len(soup.select('div.section.cop_analysis div.sub_section'))==0: # 재무제표가 없을 때
            err_wtable.append(code)
            continue
        fs_html = soup.select('div.section.cop_analysis div.sub_section')[0]
        fs_date = [item.get_text().strip() for item in fs_html.select('thead th')] # 날짜 뽑기(열)
        if len(fs_date) != 23: # 재무제표가 비어있을 때
            err_etable.append(code)
            continue
        fs_data = [item.get_text().strip().replace(',','') for item in fs_html.select('td')]  # 수치 추출(내용물) # 쉼표 때문에 나중에 float으로 안변하니까 쉼표 제거
        for i in range(len(fs_data)):
            if fs_data[i] == '-' or fs_data[i] == '':
                fs_data[i] = np.nan
        for i in range(len(fs_data)):
            fs_data[i] = float(fs_data[i])            
        
        s = pd.Series(fs_data[:10]) # 매출액
        s32 = 100 * (s[1] - s[0])/s[0]
        s21 = 100 * (s[2] - s[1])/s[1]
        s10 = 100 * (s[3] - s[2])/s[2] if pd.isnull(s[3]) == False else s21
        eps = pd.Series(fs_data[90:100]) # EPS
        exp_eps = eps[6:9].mean()*4 if pd.isnull(eps[3]) == True else eps[3]  
        exp1 = exp_eps * 10 # 공식 1
        exp3 = s10 * exp_eps # 공식 3 14.19 * 3655
        stock = soup.select('div.rate_info > div.today > p.no_today span.blind') # 현재주가
        stock = float(stock[0].text.replace(',',''))
        cod.append(code)
        nam.append(soup.select('div.wrap_company h2')[0].text)
        mi.append(round(min(exp1, exp3)))
        no.append(stock)
        cha.append(round(100*(min(exp1,exp3)-stock)/stock))
    return pd.DataFrame({'종목코드':cod, '회사명':nam, '최소적정주가':mi, '현재주가':no, '차이(%)':cha})

#  e.g)

#   종목코드        회사명    최소적정주가      현재주가  차이(%)
# 0   014530       극동유화    4467.0    3475.0   29.0
# 1   236200       슈프리마   37920.0   33600.0   13.0
# 2   017890       한국알콜   16093.0    7550.0  113.0
# 3   025770     한국정보통신    7587.0    6750.0   12.0
# 4   036670        KCI   10640.0   10250.0    4.0
# 5   004490       세방전지   41813.0   36950.0   13.0
# 6   002030        아세아  222680.0  103500.0  115.0
# 7   052330         코텍   14837.0   12750.0   16.0
# 8   102710  이엔에프테크놀로지   36550.0   23594.0   54.0
# 9   045100      한양이엔지   24613.0   12200.0  102.0
# 10  010240         흥국    6053.0    4415.0   37.0
# 11  192440     슈피겐코리아   69880.0   52000.0   34.0
# 12  094820       일진파워    6800.0    5500.0   24.0
# 13  039010       현대통신   10884.0    8050.0   35.0
# 14  015230       대창단조   60907.0   28750.0  112.0
# 15  013120       동원개발    7571.0    4070.0   86.0
# 16  282880       코윈테크   27770.0   23850.0   16.0



