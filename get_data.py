import pandas as pd
from pandas_datareader import data
import os, shutil

def getStock(symbol, start, end, name):

	print "started.................. ", name
	df =  data.get_data_yahoo(symbol, start, end)
	name_ = symbol.replace("^","")
	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + name_
	df['Return_%s' % name_] = df['AdjClose_%s' % name_].pct_change()

	name = name + ".csv"
	df.to_csv(name, sep='\t', encoding='utf-8')
	dir_ = './data/'
	if not os.path.exists(dir_):
	    os.makedirs(dir_)
	shutil.move(name, dir_)
	print "DONE...............", name
	print ''


def getStockDataFromWeb(start, end):
	stocks = {}
	stocks['sp500'] = '^GSPC'
	stocks['nasdaq'] = '^IXIC'
	stocks['djia'] = '^DJI'
	stocks['frankfurt'] = '^GDAXI'
	stocks['london'] = '^FTSE'
	stocks['paris'] = '^FCHI'
	stocks['hkong'] = '^HSI'
	stocks['nikkei'] = '^N225'
	stocks['australia'] = '^AXJO'
	stocks['amazon'] = 'AMZN'
	stocks['apple'] = 'AAPL'
	stocks['microsoft'] = 'MSFT'

	for stock in stocks:
		getStock(stocks[stock], start, end, stock)

if __name__ == "__main__":
	getStockDataFromWeb('01-01-2010', '31-12-2016')
