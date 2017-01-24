from __future__ import division

import numpy as np
import pandas as pd
import datetime
import sklearn
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

import subprocess
from sklearn.cross_validation import KFold
# from sklearn.lda import LDA

def performKNNClass(X_train, y_train, X_test, y_test):

	clf = neighbors.KNeighborsClassifier(n_neighbors=50)
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)

	return accuracy

def performSVMClass(X_train, y_train, X_test, y_test):

	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
	rbf_svc.fit(X_train, y_train)

	accuracy = rbf_svc.score(X_test, y_test)

	return accuracy


def performRFClass(X_train, y_train, X_test, y_test):

	clf = RandomForestClassifier(n_estimators=50, n_jobs=1)
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)

	return accuracy

def mergeDataFrames(datasets):
	new_df = datasets[0].join(datasets[1], how='outer')
	new_df1 = new_df.join(datasets[2], how='outer')

	return new_df1

def predict_nextDay_trend(stockToPredict, symbol):
	l = [sp500, nasdaq, stockToPredict]
	df = mergeDataFrames(l)

	dataset = df.fillna(df.mean())
	dataset = (dataset - dataset.mean()) / (dataset.max() - dataset.min())

	return_ = stockToPredict.columns[6]

	le = preprocessing.LabelEncoder()
	dataset['UpDown'] = dataset[return_]
	# print dataset['UpDown'][2:10]
	dataset.UpDown[dataset.UpDown >= 0] = 'Up'
	dataset.UpDown[dataset.UpDown < 0] = 'Down'
	# print dataset['UpDown'][2:10]

	dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)

	acc_svm = 0
	acc_KNN = 0
	acc_RF = 0
	num_folds = 5

	y = dataset.UpDown
	labels = list(set(y))
	y = np.array([labels.index(x) for x in y])

	features = dataset.iloc[:,:-1]
	X = np.array(features)

	kf = KFold(len(y), n_folds=num_folds)
	acc_svm=0
	acc_rcf=0
	acc_knn=0

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		acc_svm = acc_svm + performSVMClass(X_train, y_train, X_test, y_test)
		acc_knn = acc_knn + performKNNClass(X_train, y_train, X_test, y_test)

	print ""
	print "Final SVM Accuracy:  ", acc_svm/num_folds
	print "Final KNN Accuracy:  ", acc_knn/num_folds

def predict_next6Day_trend(stockToPredict, symbol):
	# print stockToPredict
	l = [sp500, nasdaq, stockToPredict]
	df = mergeDataFrames(l)

	dataset = df.fillna(df.mean())
	dataset = (dataset - dataset.mean()) / (dataset.max() - dataset.min())
	shape = stockToPredict.shape
	dataset['UpDown'] = np.nan
	MA = symbol + "_MA6"
	for i in range(shape[0]-6):
		if stockToPredict[MA][i+6] >= stockToPredict[MA][i]:
			dataset['UpDown'][i+6] = 1
		else:
			dataset['UpDown'][i+6] = 0

	y = dataset.UpDown
	y = y[6:-40]
	y = y.fillna(0)
	labels = list(set(y))
	y = np.array([labels.index(x) for x in y])

	features = dataset.iloc[:,:-1]
	features = features.iloc[6:-40, :]
	X = np.array(features)

	num_folds = 10
	kf = KFold(len(y), n_folds=num_folds)
	acc_svm=0
	acc_rcf=0
	acc_knn=0

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		acc_svm = acc_svm + performSVMClass(X_train, y_train, X_test, y_test)
		acc_knn = acc_knn + performKNNClass(X_train, y_train, X_test, y_test)

	print ""
	print "Final SVM Accuracy:  ", acc_svm/num_folds
	print "Final KNN Accuracy:  ", acc_knn/num_folds


sp500 = pd.read_csv("./csv_with_features/sp500_with_features.csv", sep = "\t", index_col="Date")
nasdaq = pd.read_csv("./csv_with_features/nasdaq_with_features.csv", sep = "\t", index_col="Date")
frankfurt = pd.read_csv("./csv_with_features/frankfurt_with_features.csv", sep = "\t", index_col="Date")
london = pd.read_csv("./csv_with_features/london_with_features.csv", sep = "\t", index_col="Date")
hkong = pd.read_csv("./csv_with_features/hkong_with_features.csv", sep = "\t", index_col="Date")
paris = pd.read_csv("./csv_with_features/paris_with_features.csv", sep = "\t", index_col="Date")
australia = pd.read_csv("./csv_with_features/australia_with_features.csv", sep = "\t", index_col="Date")
djia = pd.read_csv("./csv_with_features/djia_with_features.csv", sep = "\t", index_col="Date")

print "===============Predicting Next Day Trend====================="
print ""
print "Predicting frankfurt's trend"
predict_nextDay_trend(frankfurt, "frankfurt")
print ""
print "Predicting london's trend"
predict_nextDay_trend(london, "london")
print ""
print "Predicting hkong's trend"
predict_nextDay_trend(hkong, "hkong")
print ""
print "Predicting paris's trend"
predict_nextDay_trend(paris, "paris")
print ""
print "Predicting australia's trend"
predict_nextDay_trend(australia, "australia")
print ""
print "Predicting djia's trend"
predict_nextDay_trend(djia, "djia")
print ""
print "===============Predicting Next 6 Days Trend====================="
print ""
print "Predicting frankfurt's trend"
predict_next6Day_trend(frankfurt, "frankfurt")
print ""
print "Predicting london's trend"
predict_next6Day_trend(london, "london")
print ""
print "Predicting hkong's trend"
predict_next6Day_trend(hkong, "hkong")
print ""
print "Predicting paris's trend"
predict_next6Day_trend(paris, "paris")
print ""
print "Predicting australia's trend"
predict_next6Day_trend(australia, "australia")
print ""
print "Predicting djia's trend"
predict_next6Day_trend(djia, "djia")

print "===============THE END============================"
