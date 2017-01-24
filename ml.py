from __future__ import division

import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import neighbors
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
# from sklearn.lda import LDA

import matplotlib
from matplotlib import pyplot as plt


def visualize_tree(tree, feature_names):
	with open("dt.dot", 'w') as f:
		export_graphviz(tree, out_file=f, feature_names=feature_names)

	command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
	subprocess.check_call(command)
   

def performKNNClass(X_train, y_train, X_test, y_test):
	
	clf = neighbors.KNeighborsClassifier(n_neighbors=1000)
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)

	return accuracy

def performSVMClass(X_train, y_train, X_test, y_test):
	C = 1.0
	
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
	rbf_svc.fit(X_train, y_train)

	accuracy = rbf_svc.score(X_test, y_test)
	
	return accuracy
	

def performRFClass(X_train, y_train, X_test, y_test):
	
	clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
	clf.fit(X_train, y_train)  

	accuracy = clf.score(X_test, y_test)

	return accuracy

def mergeDataFrames(datasets):
	new_df1 = datasets[0].join(datasets[1], how='outer')
	# new_df1 = new_df.join(datasets[2], how='outer')

	return new_df1

def predict_nextDay_trend(base, stockToPredict):
	l = [base, stockToPredict]
	df = mergeDataFrames(l)

	dataset = df.fillna(df.mean())
	return_ = stockToPredict.columns[6]
	# print return_

	le = preprocessing.LabelEncoder()
	dataset['UpDown'] = dataset[return_]
	# print dataset['UpDown'][2:10]
	dataset.UpDown[dataset.UpDown >= 0] = 'Up'
	dataset.UpDown[dataset.UpDown < 0] = 'Down'
	# print dataset['UpDown'][2:10]

	dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
	# print dataset['UpDown'][2:10]

	features = dataset.columns[:]
	# print features["UpDown"][3:10]
	X = dataset[features]    
	y = dataset.UpDown
	# print X["UpDown"][3:10]
	# print y[1:10]
	features = list(dataset.columns[:])
	# print features

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	acc_svm=0
	acc_KNN=0
	num_folds = 10
	subset_size = len(X)//num_folds
	# print "size: ", subset_size
	# print X.shape
	for i in range(num_folds):
		# print X.iloc[10:20, 4:6]
		X_test = X.iloc[i*subset_size:(i+1)*subset_size, :]
		# print X_test.iloc[10:20, 4:6]
		if i==0:
			X_train = X.iloc[(i+1)*subset_size:-1, :]
		elif i==9:
			X_train = X.iloc[0:i*subset_size, :]
		else:
			X_train = X.iloc[:i*subset_size, :].append(X.iloc[(i+1)*subset_size:-1, :])

		# print y[1:4]
		y_test = y.iloc[i*subset_size:(i+1)*subset_size]
		# print X_test.iloc[10:20, 4:6]
		if i==0:
			y_train = y.iloc[(i+1)*subset_size:-1]
		elif i==9:
			y_train = y.iloc[0:i*subset_size]
		else:
			y_train = y.iloc[:i*subset_size].append(y.iloc[(i+1)*subset_size:-1])
		# # print "next................."
		# # print X_train.shape
		# # print X_test.shape
		acc1 = performSVMClass(X_train, y_train, X_test, y_test)
		# print "SVM: " + str(i), acc1
		# acc2 = performKNNClass(X_train, y_train, X_test, y_test)
		# print "KNN: " + str(i), acc2
		acc_svm = acc_svm + acc1
		# acc_KNN = acc_KNN + acc2
		
	print "Accuracy ", acc_svm/10
	# print "Accuracy KNN", acc_KNN/10
	# performKNNClass(X_train, y_train, X_test, y_test)
	# performSVMClass(X_train, y_train, X_test, y_test)
	# performAdaBoostClass(X_train, y_train, X_test, y_test)
	# performGTBClass(X_train, y_train, X_test, y_test)
	# performRFClass(X_train, y_train, X_test, y_test)

def predict_next6Day_trend(base, stockToPredict, symbol):
	# print stockToPredict
	l = [base, stockToPredict]
	df = mergeDataFrames(l)

	dataset = df.fillna(df.mean())
	shape = stockToPredict.shape
	dataset['UpDown'] = np.nan
	MA = symbol + "_MA6"
	for i in range(shape[0]-6):
		if stockToPredict[MA][i+6] >= stockToPredict[MA][i]:
			dataset['UpDown'][i+6] = 1
		else:
			dataset['UpDown'][i+6] = 0
	

	# print dataset["UpDown"][0:10]
	features = dataset.columns[:]
	# print features["UpDown"][3:10]
	X = dataset[features]  
	X = X.iloc[6:, :]
	X = X.fillna(0)
	y = dataset.UpDown
	y = y[6:]
	y = y.fillna(0)
	# print X["UpDown"][3:10]
	# print y[1:10]
	features = list(dataset.columns[:])
	# print features

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	acc_svm=0
	acc_KNN=0
	num_folds = 10
	subset_size = len(X)//num_folds
	print "size: ", subset_size
	# print X.shape
	for i in range(num_folds):
		# print X.iloc[10:20, 4:6]
		X_test = X.iloc[i*subset_size:(i+1)*subset_size, :]
		# print X_test.iloc[10:20, 4:6]
		if i==0:
			X_train = X.iloc[(i+1)*subset_size:-1, :]
		elif i==9:
			X_train = X.iloc[0:i*subset_size, :]
		else:
			X_train = X.iloc[:i*subset_size, :].append(X.iloc[(i+1)*subset_size:-1, :])

		# print y[1:4]
		y_test = y.iloc[i*subset_size:(i+1)*subset_size]
		# print X_test.iloc[10:20, 4:6]
		if i==0:
			y_train = y.iloc[(i+1)*subset_size:-1]
		elif i==9:
			y_train = y.iloc[0:i*subset_size]
		else:
			y_train = y.iloc[:i*subset_size].append(y.iloc[(i+1)*subset_size:-1])
		# # print "next................."
		# # print X_train.shape
		# # print X_test.shape
		# print X_train
		acc1 = performSVMClass(X_train, y_train, X_test, y_test)
		# print "SVM: " + str(i), acc1
		# acc2 = performKNNClass(X_train, y_train, X_test, y_test)
		# print "KNN: " + str(i), acc2
		acc_svm = acc_svm + acc1
		# acc_KNN = acc_KNN + acc2
		
	print "Accuracy "+ symbol +" ", acc_svm/10


sp500 = pd.read_csv("sp500_with_features.csv", sep = "\t", index_col="Date")
# nasdaq = pd.read_csv("nasdaq_with_features.csv", sep = "\t", index_col="Date")
# frankfurt = pd.read_csv("frankfurt_with_features.csv", sep = "\t", index_col="Date")
# london = pd.read_csv("london_with_features.csv", sep = "\t", index_col="Date")
hkong = pd.read_csv("hkong_with_features.csv", sep = "\t", index_col="Date")
# paris = pd.read_csv("paris_with_features.csv", sep = "\t", index_col="Date")
australia = pd.read_csv("australia_with_features.csv", sep = "\t", index_col="Date")
djia = pd.read_csv("djia_with_features.csv", sep = "\t", index_col="Date")

print "===============Predicting Next Day Trend====================="
print ""
print "Predicting base: sp500, market index: hkong"
predict_nextDay_trend(sp500, hkong)
print ""
print "Predicting base: sp500, market index: australia"
predict_nextDay_trend(sp500, australia)
print ""
print "Predicting base: sp500, market index: djia"
predict_nextDay_trend(sp500, djia)
print ""
print "Predicting base: hkong, market index: sp500"
predict_nextDay_trend(hkong, sp500)
print ""
print "Predicting base: australia, market index: sp500"
predict_nextDay_trend(australia, sp500)
print ""
print "Predicting base: djia, market index: sp500"
predict_nextDay_trend(djia, sp500)
print ""
print "Predicting base: hkong, market index: australia"
predict_nextDay_trend(hkong, australia)
print ""
print "Predicting base: hkong, market index: djia"
predict_nextDay_trend(hkong, djia)
print ""
print "Predicting base: australia, market index: djia"
predict_nextDay_trend(australia, djia)
print ""
print "Predicting base: australia, market index: hkong"
predict_nextDay_trend(australia, hkong)
print ""
print "Predicting base: djia, market index: australia"
predict_nextDay_trend(djia, australia)
print ""
print "Predicting base: djia, market index: hkong"
predict_nextDay_trend(djia, hkong)
print ""

print "===============Predicting Next 6 Days Trend====================="
print ""
print "Predicting base: sp500, market index: hkong"
predict_next6Day_trend(sp500, hkong, "hkong")
print ""
print "Predicting base: sp500, market index: australia"
predict_next6Day_trend(sp500, australia, "australia")
print ""
print "Predicting base: sp500, market index: djia"
predict_next6Day_trend(sp500, djia, "djia")
print ""
print "Predicting base: hkong, market index: sp500"
predict_next6Day_trend(hkong, sp500, "sp500")
print ""
print "Predicting base: australia, market index: sp500"
predict_next6Day_trend(australia, sp500, "sp500")
print ""
print "Predicting base: djia, market index: sp500"
predict_next6Day_trend(djia, sp500, "sp500")
print ""
print "Predicting base: hkong, market index: australia"
predict_next6Day_trend(hkong, australia, "australia")
print ""
print "Predicting base: hkong, market index: djia"
predict_next6Day_trend(hkong, djia, "djia")
print ""
print "Predicting base: australia, market index: djia"
predict_next6Day_trend(australia, djia, "djia")
print ""
print "Predicting base: australia, market index: hkong"
predict_next6Day_trend(australia, hkong, "hkong")
print ""
print "Predicting base: djia, market index: australia"
predict_next6Day_trend(djia, australia, "australia")
print ""
print "Predicting base: djia, market index: hkong"
predict_next6Day_trend(djia, hkong, "hkong")
print ""

print "===============THE END============================"