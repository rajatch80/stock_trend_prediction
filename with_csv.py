import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
# from sklearn.qda import QDA
import re
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
# from dateutil import parser
# from backtest import Strategy, Portfolio


def addFeatures(dataframe, adjclose, returns, n):
	"""
	operates on two columns of dataframe:
	- n >= 2
	- given Return_* computes the return of day i respect to day i-n. 
	- given AdjClose_* computes its moving average on n days

	"""
	# print adjclose
	if adjclose[9] == "^":
		return_n = adjclose[10:] + "_Time_" + str(n)
		dataframe[return_n] = dataframe[adjclose].pct_change(n)
		# print return_n
		roll_n = returns[8:] + "_RollMean_" + str(n)
		dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)
	else:
		return_n = adjclose[9:] + "_Time_" + str(n)
		dataframe[return_n] = dataframe[adjclose].pct_change(n)
		# print return_n
		roll_n = returns[7:] + "_RollMean_" + str(n)
		dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

def applyRollMeanDelayedReturns(datasets, delta):
	"""
	applies rolling mean and delayed returns to each dataframe in the list
	"""
	i=1
	for dataset in datasets:
		columns = dataset.columns    
		adjclose = columns[-2]
		returns = columns[-1]
		for n in delta:			
			addFeatures(dataset, adjclose, returns, n)

		# f = str(i)+".csv"
		# dataset.to_csv(f, sep='\t', encoding='utf-8')
		# i=i+1
	return datasets

def mergeDataframes(datasets, index, cut):
	"""
	merges datasets in the list 
	"""
	subset = []
	# print datasets[0]
	# dff = datasets[0].index[2]
	# dff1 = datasets[0].iloc[:, index:]
	# print dff1
	subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
	# print subset[0]

	first = subset[0].join(subset[1:], how = 'outer')
	# print first['^DJITime_5']
	finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
	# print finance['^DJITime_5'][1:7]
	finance = finance[finance.index > cut]
	# print finance['^DJITime_5'][1:13]
	return finance

def applyTimeLag(dataset, lags, delta):
	"""
	apply time lag to return columns selected according  to delta.
	Days to lag are contained in the lads list passed as argument.
	Returns a NaN free dataset obtained cutting the lagged dataset
	at head and tail
	"""

	dataset.Return_Out = dataset.Return_Out.shift(-1)
	maxLag = max(lags)

	columns = dataset.columns[::(2*max(delta)-1)]
	# print columns
	for column in columns:
		for lag in lags:
			newcolumn = column.replace("^","") + "_"+str(lag)
			dataset[newcolumn] = dataset[column].shift(lag)

	# print dataset
	return dataset.iloc[maxLag:-1,:]

def prepareDataForClassification(dataset):
	"""
	generates categorical output column, attach to dataframe 
	label the categories and split into train and test
	"""
	le = preprocessing.LabelEncoder()

	dataset['UpDown'] = dataset['Return_Out']
	dataset.UpDown[dataset.UpDown >= 0] = 'Up'
	dataset.UpDown[dataset.UpDown < 0] = 'Down'
	dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)

	features = dataset.columns[1:-1]
	X = dataset[features]    
	y = dataset.UpDown
	# print y
	print X.shape
	# print y.shape
	# for i in range(len(X.columns)):  
	# 	print X.columns[i] 
	# X.to_csv("X.csv", sep='\t', encoding='utf-8')
	# y.to_csv("y.csv", sep='\t', encoding='utf-8')
	# print X.iloc[2:5, 78:84]	
	# X = X.fillna(X.mean())
	# print X.iloc[2:5, 78:84]
	# print X.index

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# print X_train.iloc[2:5, 78:84]
	X_train = X_train.fillna(X_train.mean())
	X_test = X_test.fillna(X_test.mean())
	# print X_train.iloc[2:5, 78:84]
	# X_train.to_csv("X_train.csv", sep='\t', encoding='utf-8')
	# y_train.to_csv("y_train.csv", sep='\t', encoding='utf-8')

	# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	# imp.fit(X_train)
	# # imp.fit(y_train)
	# imp.fit(X_test)
	# # imp.fit(y_test)
	# X_train = imp.fit_transform(X_train)
	# # y_train = imp.fit_transform(y_train)
	# X_test = imp.fit_transform(X_test)
	# # y_test = imp.fit_transform(y_test)

	# imp = Imputer(missing_values=0, strategy='mean', axis=0)
	# imp.fit(X_train)
	# # imp.fit(y_train)
	# imp.fit(X_test)
	# # imp.fit(y_test)
	# X_train = imp.fit_transform(X_train)
	# # y_train = imp.fit_transform(y_train)
	# X_test = imp.fit_transform(X_test)
	# y

	return X_train, y_train, X_test, y_test 


def performRFClass(X_train, y_train, X_test, y_test, fout, savemodel):
	"""
	Random Forest Binary Classification
	"""

	clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
	clf.fit(X_train, y_train)

	# if savemodel == True:
	# 	fname_out = '{}-{}.pickle'.format(fout, datetime.now())
	# 	with open(fname_out, 'wb') as f:
	# 		cPickle.dump(clf, f, -1)    

	accuracy = clf.score(X_test, y_test)

	return accuracy

def performKNNClass(X_train, y_train, X_test, y_test, fout, savemodel):
	"""
	KNN binary Classification
	"""
	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train, y_train)

	# if savemodel == True:
	# 	fname_out = '{}-{}.pickle'.format(fout, datetime.now())
	# 	with open(fname_out, 'wb') as f:
	# 		cPickle.dump(clf, f, -1)    

	accuracy = clf.score(X_test, y_test)

	print "KNN: ", accuracy

def performSVMClass(X_train, y_train, X_test, y_test, fout, savemodel):
	"""
	SVM binary Classification
	"""
	# c = parameters[0]
	# g =  parameters[1]
	clf = SVC()
	clf.fit(X_train, y_train)

	# if savemodel == True:
	# 	fname_out = '{}-{}.pickle'.format(fout, datetime.now())
	# 	with open(fname_out, 'wb') as f:
	# 		cPickle.dump(clf, f, -1)    

	accuracy = clf.score(X_test, y_test)

	return accuracy

def performAdaBoostClass(X_train, y_train, X_test, y_test, fout, savemodel):
	"""
	Ada Boosting binary Classification
	"""
	# n = parameters[0]
	# l =  parameters[1]
	clf = AdaBoostClassifier()
	clf.fit(X_train, y_train)

	# if savemodel == True:
	# 	fname_out = '{}-{}.pickle'.format(fout, datetime.now())
	# 	with open(fname_out, 'wb') as f:
	# 		cPickle.dump(clf, f, -1)    

	accuracy = clf.score(X_test, y_test)

	print "AdaBoost: ", accuracy

def performGTBClass(X_train, y_train, X_test, y_test, fout, savemodel):
	"""
	Gradient Tree Boosting binary Classification
	"""
	clf = GradientBoostingClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	# if savemodel == True:
	# 	fname_out = '{}-{}.pickle'.format(fout, datetime.now())
	# 	with open(fname_out, 'wb') as f:
	# 		cPickle.dump(clf, f, -1)    

	accuracy = clf.score(X_test, y_test)

	print "GTBClass: ", accuracy

def performQDAClass(X_train, y_train, X_test, y_test, fout, savemodel):
	"""
	Quadratic Discriminant Analysis binary Classification
	"""
	def replaceTiny(x):
		if (abs(x) < 0.0001):
			x = 0.0001

	X_train = X_train.apply(replaceTiny)
	X_test = X_test.apply(replaceTiny)

	clf = QDA()
	clf.fit(X_train, y_train)

	# if savemodel == True:
	# 	fname_out = '{}-{}.pickle'.format(fout, datetime.now())
	# 	with open(fname_out, 'wb') as f:
	# 		cPickle.dump(clf, f, -1)    

	accuracy = clf.score(X_test, y_test)

	return accuracy


def performFeatureSelection(maxdeltas, maxlags):
	"""
	Performs Feature selection for a specific algorithm
	"""
	list_acc =[]
	for maxlag in range(3, maxlags + 2):
		lags = range(2, maxlag) 
		print ''
		print '============================================================='
		print 'Maximum time lag applied', max(lags)
		print ''
		for maxdelta in range(3, maxdeltas + 2):
			# datasets = getStockDataFromWeb('^GSPC', '01-01-1993', '03-07-2016')
			delta = range(2, maxdelta) 
			print 'Delta days accounted: ', max(delta)
			print "delta: ", delta
			print "lags: ", lags
			l = []
			stocks = ["out", "IXIC", "HSI", "N225", "AXJO"]
			# dataset = pd.read_csv("AXJO.csv", sep = "\t", index_col="Date")

			# print dataset.iloc[:8, 6:]
			for stock in stocks:
				stock = stock+".csv"
				df = pd.read_csv(stock, sep = "\t", index_col="Date")
				l.append(df)

			l1=[]
			l1 = applyRollMeanDelayedReturns(l, delta)
			# print "APPLY ROLLMEAN DONE......."
			l2=[]
			l2 = mergeDataframes(l, 6, '1998-05-05')
			print 'Size of data frame: ', l2.shape
			# print 'Number of NaN after merging: ', count_missing(finance)
			# finance = finance.interpolate(method='linear')
			# print 'Number of NaN after time interpolation: ', count_missing(finance)
			# finance = finance.fillna(finance.mean())
			# print 'Number of NaN after mean interpolation: ', count_missing(finance)    
			l3 = applyTimeLag(l2, lags, delta)
			
			# print 'Number of NaN after temporal shifting: ', count_missing(finance)
			print 'Size of data frame after feature creation: ', l3.shape
			X_train, y_train, X_test, y_test  = prepareDataForClassification(l3)
			print "DATA PREPARED..."
			accu = performSVMClass(X_train, y_train, X_test, y_test, "^IXIC", True)
			print "accuracy: ", accu
			list_acc.append(accu)
			# print performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
			# print '' 
	sum = 0
	# print list_acc
	for s in list_acc:
		sum= sum + s
	avg = sum/len(list_acc)
	print "================================================================"
	print " "
	print "Average: ", avg
	print " "


# def getPredictionFromBestModel(bestdelta, bestlags, fout, cut, start_test, path_datasets, best_model):
#     """
#     returns array of prediction and score from best model.
#     """
#     lags = range(2, bestlags + 1) 
#     l = []
# 	stocks = ["out", "IXIC", "DJI", "GDAXI", "FTSE", "FCHI", "HSI", "N225", "AXJO"]
# 	# dataset = pd.read_csv("AXJO.csv", sep = "\t", index_col="Date")

# 	# print dataset.iloc[:8, 6:]
# 	for stock in stocks:
# 		stock = stock+".csv"
# 		df = pd.read_csv(stock, sep = "\t", index_col="Date")
# 		l.append(df)

#     delta = range(2, bestdelta + 1) 
#     datasets = applyRollMeanDelayedReturns(l, delta)
#     finance = mergeDataframes(datasets, 6, cut)
#     finance = finance.interpolate(method='linear')
#     # finance = finance.fillna(finance.mean())    
#     finance = applyTimeLag(finance, lags, delta)
#     X_train, y_train, X_test, y_test  = prepareDataForClassification(finance)    
#     # with open(best_model, 'rb') as fin:
#     #     model = cPickle.load(fin)        
        
#     return model.predict(X_test), model.score(X_test, y_test)


####  MAIN  ############
# l = []
# # stocks = ["out", "IXIC", "DJI", "GDAXI", "FTSE", "FCHI", "HSI", "N225", "AXJO"]
# stocks = ["out", "IXIC", "DJI", "GDAXI"]

# # dataset = pd.read_csv("AXJO.csv", sep = "\t", index_col="Date")

# # print dataset.iloc[:8, 6:]
# for stock in stocks:
# 	stock = stock+".csv"
# 	df = pd.read_csv(stock, sep = "\t", index_col="Date")
# 	l.append(df)

# # deltas = [2,3,4,5,6,7,8,9]
# # lags = [2,3,4,5,6,7,8,9]
# deltas = [2,3]
# lags = [2,3]

# l1 = applyRollMeanDelayedReturns(l, deltas)
# # print l1[1][1:5][2:5]
# l2 = mergeDataframes(l1, 6, '1996-05-05')
# # # l1.to_csv("l1.csv", sep='\t', encoding='utf-8')
# # # print l2
# l3 = applyTimeLag(l2, lags, deltas)
# # print l3["^DJITime_5"]

# X_train, y_train, X_test, y_test  = prepareDataForClassification(l3)
# # print X_train
# print y_train


# performRFClass(X_train, y_train, X_test, y_test, "^IXIC", True)
# performKNNClass(X_train, y_train, X_test, y_test, "^IXIC", True)
# performSVMClass(X_train, y_train, X_test, y_test, "^IXIC", True)
# performAdaBoostClass(X_train, y_train, X_test, y_test, "^IXIC", True)
# performGTBClass(X_train, y_train, X_test, y_test, "^IXIC", True)

performFeatureSelection(9, 9)