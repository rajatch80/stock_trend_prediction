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


def getStock(symbol, start, end):
	"""
	Downloads Stock from Yahoo Finance.
	Computes daily Returns based on Adj Close.
	Returns pandas dataframe.
	"""
	df =  pandas.io.data.get_data_yahoo(symbol, start, end)

	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + symbol
	df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()

	return df

def getStockDataFromWeb(fout, start, end):
	"""
	Collects predictors data from Yahoo Finance and Quandl.
	Returns a list of dataframes.
	"""
	#start = parser.parse(start_string)
	#end = parser.parse(end_string)

	nasdaq = getStock('^IXIC', start, end)
	# print "Got nasdaq"
	djia = getStock('^DJI', start, end)
	# print "Got djia"
	frankfurt = getStock('^GDAXI', start, end)
	# print "Got frankfurt"
	london = getStock('^FTSE', start, end)
	# print "Got london"
	paris = getStock('^FCHI', start, end)
	# print "Got paris"
	hkong = getStock('^HSI', start, end)
	# print "Got hkong"
	nikkei = getStock('^N225', start, end)
	# print "Got nikkei"
	australia = getStock('^AXJO', start, end)
	# print "Got australia"


	out =  pandas.io.data.get_data_yahoo(fout, start, end)
	print "Got out"
	out.columns.values[-1] = 'AdjClose'
	out.columns = out.columns + '_Out'
	out['Return_Out'] = out['AdjClose_Out'].pct_change()

	return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]

def addFeatures(dataframe, adjclose, returns, n):
	"""
	operates on two columns of dataframe:
	- n >= 2
	- given Return_* computes the return of day i respect to day i-n. 
	- given AdjClose_* computes its moving average on n days

	"""

	return_n = adjclose[9:] + "Time_" + str(n)
	dataframe[return_n] = dataframe[adjclose].pct_change(n)

	roll_n = returns[7:] + "RollMean_" + str(n)
	dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

def applyRollMeanDelayedReturns(datasets, delta):
	"""
	applies rolling mean and delayed returns to each dataframe in the list
	"""
	for dataset in datasets:
		columns = dataset.columns    
		adjclose = columns[-2]
		returns = columns[-1]
		for n in delta:
			addFeatures(dataset, adjclose, returns, n)

	return datasets

def mergeDataframes(datasets, index, cut):
	"""
	merges datasets in the list 
	"""
	subset = []
	subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
	# print subset[0]

	first = subset[0].join(subset[1:], how = 'outer')
	# print first
	finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
	# print finance
	finance = finance[finance.index > cut]
	# print finance
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
			newcolumn = column + str(lag)
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

	# print X.shape
	# print y.shape   

	# print X.index

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	imp.fit(X_train)
	# imp.fit(y_train)
	imp.fit(X_test)
	# imp.fit(y_test)
	X_train = imp.fit_transform(X_train)
	# y_train = imp.fit_transform(y_train)
	X_test = imp.fit_transform(X_test)
	# y_test = imp.fit_transform(y_test)

	imp = Imputer(missing_values=0, strategy='mean', axis=0)
	imp.fit(X_train)
	# imp.fit(y_train)
	imp.fit(X_test)
	# imp.fit(y_test)
	X_train = imp.fit_transform(X_train)
	# y_train = imp.fit_transform(y_train)
	X_test = imp.fit_transform(X_test)
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

	print "RF: ", accuracy

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

	print "SVM: ", accuracy

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

	for maxlag in range(3, maxlags + 2):
		lags = range(2, maxlag) 
		print ''
		print '============================================================='
		print 'Maximum time lag applied', max(lags)
		print ''
		for maxdelta in range(3, maxdeltas + 2):
			datasets = getStockDataFromWeb('^GSPC', '01-01-1993', '03-07-2016')
			delta = range(2, maxdelta) 
			print 'Delta days accounted: ', max(delta)
			datasets = applyRollMeanDelayedReturns(datasets, delta)
			print "APPLY ROLLMEAN DONE......."
			finance = mergeDataframes(datasets, 6, '1995-02-01')
			print 'Size of data frame: ', finance.shape
			# print 'Number of NaN after merging: ', count_missing(finance)
			# finance = finance.interpolate(method='linear')
			# print 'Number of NaN after time interpolation: ', count_missing(finance)
			# finance = finance.fillna(finance.mean())
			# print 'Number of NaN after mean interpolation: ', count_missing(finance)    
			finance = applyTimeLag(finance, lags, delta)
			# print 'Number of NaN after temporal shifting: ', count_missing(finance)
			print 'Size of data frame after feature creation: ', finance.shape
			X_train, y_train, X_test, y_test  = prepareDataForClassification(finance)
			print "DATA PREPARED..."
			performRFClass(X_train, y_train, X_test, y_test, "^IXIC", True)
			# print performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
			# print '' 

####  MAIN  ############
l = getStockDataFromWeb('^GSPC', '1993-01-01', '2016-03-07')

deltas = [3,4,5,6,7,8]
lags = [1,2,3,4,5]

l1 = applyRollMeanDelayedReturns(l, deltas)
print "done Apply"
l2 = mergeDataframes(l1, 6, '1995-02-01')
print "Done merging"
l3 = applyTimeLag(l2, lags, deltas)
# l3.to_csv("l3.csv", sep='\t', encoding='utf-8')
print "done apply time lag"
# print l3
X_train, y_train, X_test, y_test  = prepareDataForClassification(l3)

# print X_train.iloc[2:5, 78:84]

# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit(X_train.iloc[2:5, 78:84])
# Z = imp.fit_transform(X_train.iloc[2:5, 78:84])

# print Z

performRFClass(X_train, y_train, X_test, y_test, "^IXIC", True)
performKNNClass(X_train, y_train, X_test, y_test, "^IXIC", True)
performSVMClass(X_train, y_train, X_test, y_test, "^IXIC", True)
performAdaBoostClass(X_train, y_train, X_test, y_test, "^IXIC", True)
performGTBClass(X_train, y_train, X_test, y_test, "^IXIC", True)

# performFeatureSelection(9, 9)

# print "Accuracy: ", accuracy