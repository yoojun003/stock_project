import sys
from PyQt5.QtWidgets import *
import Kiwoom
import time
from pandas import DataFrame
import datetime

MARKET_KOSPI   = 0
MARKET_KOSDAQ  = 10

class PyMon:
    def __init__(self):
        self.kiwoom = Kiwoom.Kiwoom()
        self.kiwoom.comm_connect()
        self.get_code_list()

    def get_code_list(self):
        self.kospi_codes = self.kiwoom.get_code_list_by_market(MARKET_KOSPI)
        self.kosdaq_codes = self.kiwoom.get_code_list_by_market(MARKET_KOSDAQ)

    def get_ohlcv(self, code, start):
        self.kiwoom.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("기준일자", start)
        self.kiwoom.set_input_value("수정주가구분", 1)
        self.kiwoom.comm_rq_data("opt10081_req", "opt10081", 0, "0101")
        time.sleep(0.2)

        df = DataFrame(self.kiwoom.ohlcv, columns=['open', 'high', 'low', 'close', 'volume'],
                       index=self.kiwoom.ohlcv['date'])
        return df

    def check_speedy_rising_volume(self, code):
        #today = datetime.datetime.today().strftime("%Y%m%d")
        today="20191121"
        df = self.get_ohlcv(code, today)
        #print(code)
        volumes = df['volume']

        if len(volumes) < 21:
            return False

        sum_vol20 = 0
        today_vol = 0

        for i, vol in enumerate(volumes):
            if i == 0:
                today_vol = vol
            elif 1 <= i <= 20:
                sum_vol20 += vol
            else:
                break

        avg_vol20 = sum_vol20 / 20
        print(avg_vol20)
        print(today_vol)

        if today_vol > avg_vol20 * 10:
            print("True")
            return True

    def update_buy_list(self, buy_list):
        """f = open("D:/Pythondata/book-master/ch19/day05/buy_list.txt", "wt")
        for code in buy_list:
            print(code)
            f.writelines("매수;", code, ";시장가;10;0;매수전")
        f.close()"""

        with open("D:/Pythondata/book-master/ch19/day05/buy_list.txt", "w") as file:  # 오버라이트됨
            for code in buy_list:
                #file.write("매수;", code, ";시장가;10;0;매수전") #에러
                file.write("매수;"+code+";시장가;10;0;매수전\n")

    def run(self):
        buy_list = []
        num = len(self.kosdaq_codes[0:100])

        for i, code in enumerate(self.kosdaq_codes[0:100]):
            print(i, '/', num)
            if self.check_speedy_rising_volume(code):
                print("급등주: "+code)
                buy_list.append(code)
                #print(buy_list)
        self.update_buy_list(buy_list)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pymon = PyMon()
    pymon.run()

