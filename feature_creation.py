from __future__ import division

import numpy as np
import pandas as pd

def listsum(numList):
	theSum = 0
	for i in numList:
		theSum = theSum + i
	return theSum

def create_features(stock, symbol):
	
	open_ = stock.columns[0]
	high = stock.columns[1]
	low =  stock.columns[2]
	volume =  stock.columns[4]
	adj = stock.columns[5]
	return_ =  stock.columns[6]

	# All Technical Indicators
	MA6 = symbol + "_MA6"
	MA14 = symbol + "_MA14"
	EMA6 = symbol + "_EMA6"
	EMA14 = symbol + "_EMA14"
	MACD = symbol + "_MACD"
	MoM6 = symbol + "_MoM6"
	MoM14 = symbol + "_MoM14"
	K_per6 = symbol + "_K%6"
	WIILR6 = symbol + "_WIILR6"
	K_per14 = symbol + "_K%14"
	WIILR14 = symbol + "_WIILR14"
	RoC6 = symbol + "_RoC6"
	RoC14 = symbol + "_RoC14"
	RSI6 = symbol + "_RSI6"
	RSI14 = symbol + "_RSI14"
	OBV = symbol + "_OBV"

	# Rate of Change, 6 and 14 days
	stock[RoC6] = stock[adj].pct_change(6)
	stock[RoC14] = stock[adj].pct_change(14)

	# Moving Average, 6 and 14 days
	stock[MA6] = pd.rolling_mean(stock[adj], 6)
	stock[MA14] = pd.rolling_mean(stock[adj], 14)

	shape_stock = stock.shape

	# Momentum, 6 and 14 days
	# print shape_stock
	stock[MoM6] = np.nan
	stock[MoM14] = np.nan
	for i in range(shape_stock[0]-6):
		stock[MoM6][i+6] = stock[adj][i+6] - stock[adj][i]
	for i in range(shape_stock[0]-14):
		stock[MoM14][i+14] = stock[adj][i+14] - stock[adj][i]

	# Exponential Moving Average, 6 and 14 days
	stock[EMA6] = np.nan
	stock[EMA14] = np.nan
	stock[EMA6][0] = stock[adj][0]
	stock[EMA14][0] = stock[adj][0]

	multi_6 = 2/7
	multi_14 = 2/15
	for i in range(shape_stock[0]-1):
		stock[EMA6][i+1] = stock[EMA6][i] + multi_6 * (stock[adj][i] - stock[EMA6][i])
		stock[EMA14][i+1] = stock[EMA14][i] + multi_14 * (stock[adj][i] - stock[EMA14][i])
	
	# print max(stock[high][0:14])
	# print min(stock[low][0:14])

	# William %R
	stock[WIILR6] = np.nan
	stock[WIILR14] = np.nan
	for i in range(shape_stock[0]-6):
		stock[WIILR6][i+6] = (max(stock[high][i:i+6]) - stock[adj][i+6])/(max(stock[high][i:i+6]) - min(stock[low][i:i+6]))
	
	for i in range(shape_stock[0]-14):
		stock[WIILR14][i+14] = (max(stock[high][i:i+14]) - stock[adj][i+14])/(max(stock[high][i:i+14]) - min(stock[low][i:i+14]))
		
	# %K
	stock[K_per6] = np.nan
	stock[K_per14] = np.nan
	for i in range(shape_stock[0]-6):
		stock[K_per6][i+6] = (stock[adj][i+6] - min(stock[low][i:i+6]))/(max(stock[high][i:i+6]) - min(stock[low][i:i+6]))
	
	for i in range(shape_stock[0]-14):
		stock[K_per14][i+14] = (stock[adj][i+14] - min(stock[low][i:i+14]))/(max(stock[high][i:i+14]) - min(stock[low][i:i+14]))
	
	# OBV
	stock[OBV] = np.nan
	stock[OBV][0] = stock[volume][0]
	for i in range(shape_stock[0]-1):
		if stock[adj][i+1] > stock[adj][i]:
			stock[OBV][i+1] = stock[OBV][i] + stock[volume][i+1]
		if stock[adj][i+1] < stock[adj][i]:
			stock[OBV][i+1] = stock[OBV][i] - stock[volume][i+1]
		if stock[adj][i+1] == stock[adj][i]:
			stock[OBV][i+1] = stock[OBV][i]

	#RSI
	stock[RSI6]=np.nan
	stock[RSI14]=np.nan
	stock["change"] = np.nan
	stock["gain"] = float(0)
	stock["loss"] = float(0)
	stock["avgGain6"] = float(0)
	stock["avgLoss6"] = float(0)
	stock["avgGain14"] = float(0)
	stock["avgLoss14"] = float(0)
	# print stock[adj][0]
	# stock["change"][0] = 0
	for i in range(shape_stock[0]-1):	
		stock["change"][i+1] = stock[adj][i+1] - stock[adj][i]
		if stock["change"][i+1] > 0:
			stock["gain"][i+1] = float(stock["change"][i+1])
		else:
			stock["loss"][i+1] = float(stock["change"][i+1])*(-1)

	for i in range(shape_stock[0]-6):
		stock["avgGain6"][i+6] = listsum(stock["gain"][i:i+6])/6
		stock["avgLoss6"][i+6] = listsum(stock["loss"][i:i+6])/6

	for i in range(shape_stock[0]-14):
		stock["avgGain14"][i+14] = listsum(stock["gain"][i:i+14])/14
		stock["avgLoss14"][i+14] = listsum(stock["loss"][i:i+14])/14

	for i in range(shape_stock[0]-6):
		stock[RSI6][i+6] = stock["avgGain6"][i+6]/(stock["avgGain6"][i+6] + stock["avgLoss6"][i+6])
		
	for i in range(shape_stock[0]-14):
		stock[RSI14][i+14] = stock["avgGain14"][i+14]/(stock["avgGain14"][i+14] + stock["avgLoss14"][i+14])
		
	stock = stock.drop("change", 1)
	stock = stock.drop("gain", 1)
	stock = stock.drop("loss", 1)
	stock = stock.drop("avgGain6", 1)
	stock = stock.drop("avgLoss6", 1)
	stock = stock.drop("avgGain14", 1)
	stock = stock.drop("avgLoss14", 1)

	stock = stock.fillna(stock.mean())

	# print stock.iloc[1:15, :]

	name = symbol + "_with_features.csv"
	stock.to_csv(name, sep='\t', encoding='utf-8')
	print "Done... ", name


if __name__ == "__main__":
	# sp500 = pd.read_csv("GSPC.csv", sep = "\t", index_col="Date")
	# nasdaq = pd.read_csv("IXIC.csv", sep = "\t", index_col="Date")
	# frankfurt = pd.read_csv("GDAXI.csv", sep = "\t", index_col="Date")
	# djia = pd.read_csv("DJI.csv", sep = "\t", index_col="Date")
	# london = pd.read_csv("FTSE.csv", sep = "\t", index_col="Date")
	# paris = pd.read_csv("FCHI.csv", sep = "\t", index_col="Date")
	# hkong = pd.read_csv("HSI.csv", sep = "\t", index_col="Date")
	# australia = pd.read_csv("AXJO.csv", sep = "\t", index_col="Date")
	# nikkei = pd.read_csv("N225.csv", sep = "\t", index_col="Date")
	# apple = pd.read_csv("AAPL.csv", sep = "\t", index_col="Date")
	# amazon = pd.read_csv("AMZN.csv", sep = "\t", index_col="Date")
	# microsoft = pd.read_csv("MSFT.csv", sep = "\t", index_col="Date")
	reliance = pd.read_csv("RELIANCE.NS.csv", sep = "\t", index_col="Date")
	tata = pd.read_csv("TATASTEEL.NS.csv", sep = "\t", index_col="Date")

	
	# create_features(sp500, "sp500")
	# create_features(nasdaq, "nasdaq")
	# create_features(frankfurt, "frankfurt")
	# create_features(djia, "djia")
	# create_features(london, "london")
	# create_features(paris, "paris")
	# create_features(hkong, "hkong")
	# create_features(australia, "australia")
	# create_features(nikkei, "nikkei")
	# create_features(apple, "apple")
	# create_features(amazon, "amazon")
	# create_features(microsoft, "microsoft")
	create_features(reliance, "reliance")
	create_features(tata, "tata")
	