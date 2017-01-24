import pandas as pd
import pandas.io.data


def getStock(symbol, start, end):

	print "started ", symbol
	df =  pandas.io.data.get_data_yahoo(symbol, start, end)
	name = symbol.replace("^","")
	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + name
	df['Return_%s' % name] = df['AdjClose_%s' % name].pct_change()
	name = symbol.replace("^","") + ".csv"
	print "DONE..............."
	
	df.to_csv(name, sep='\t', encoding='utf-8')
	

def getStockDataFromWeb(start, end):
	# getStock('^GSPC', start, end)
	# print "Got S&P500"
	# getStock('^IXIC', start, end)
	# print "Got nasdaq"
	# getStock('^DJI', start, end)
	# print "Got djia"
	# getStock('^GDAXI', start, end)
	# print "Got frankfurt"
	# getStock('^FTSE', start, end)
	# print "Got london"
	# getStock('^FCHI', start, end)
	# print "Got paris"
	# getStock('^HSI', start, end)
	# print "Got hkong"
	# getStock('^N225', start, end)
	# print "Got nikkei"
	# getStock('^AXJO', start, end)
	# print "Got australia"
	# getStock('AAPL', start, end)
	# print "Got AAPL"
	# getStock('AMZN', start, end)
	# print "Got AMZN"
	# getStock('MSFT', start, end)
	# print "Got MSFT"
	getStock('RELIANCE.NS', start, end)
	print "Got RELIANCE.NS"
	getStock('TATASTEEL.NS', start, end)
	print "Got TATASTEEL.NS"
	



	# out =  pandas.io.data.get_data_yahoo(fout, start, end)
	# print "Got out"
	# out.columns.values[-1] = 'AdjClose'
	# out.columns = out.columns + '_Out'
	# out['Return_Out'] = out['AdjClose_Out'].pct_change()

	# out.to_csv("out.csv", sep='\t', encoding='utf-8')

	
getStockDataFromWeb('01-01-2010', '03-06-2016')
