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
	
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1)
	rbf_svc.fit(X_train, y_train)

	accuracy = rbf_svc.score(X_test, y_test)
	
	return accuracy
	

def performRFClass(X_train, y_train, X_test, y_test):
	
	clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
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
	return_ = stockToPredict.columns[6]
	# print return_
	return_sp500 = sp500.columns[6]
	# print return_sp500
	return_nasdaq = nasdaq.columns[6]
	# print return_nasdaq
	# print return_
	shape = dataset.shape
	dataset = (dataset - dataset.mean()) / (dataset.max() - dataset.min())
	# return 0
	# dataset = (dataset - dataset.mean()) / (dataset.max() - dataset.min())
	# for i in range(shape[1]):
	# 	col = dataset.columns[i]
	# 	if col != return_ and col != return_sp500 and col != return_nasdaq :
	# 		# print col
	# 		mini = min(dataset[col])
	# 		# print mini
	# 		maxi = max(dataset[col])
	# 		# print maxi
	# 		for j in range(shape[0]):
	# 			datasetset[col][j] = (dataset[col][j]- mini) / (maxi-mini)

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
	# print X
	
	# print X.iloc[4500:4510,1:5]
	# return 0
	# X_scaled = preprocessing.scale(X)
	# print X_scaled
	# a = X.iloc[0:10,0:1]
	# col = a.columns[0]
	# print max(a[col])
	# print min(a[col])
	# print max(a[col]) - min(a[col])
	print ""
	# for i in range(10):
	# 	print (a[col][i]-min(a[col]))/ (max(a[col])-min(a[col]))
	# print len(X.columns)
	# print shape[1]

	# return 0
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

		labels = list(set(y_train))
		print labels
		y_Train = np.array([labels.index(x) for x in y_train])
		X_Train = np.array(X_train)

		labels1 = list(set(y_test))
		y_Test = np.array([labels.index(x) for x in y_test])
		X_Test = np.array(X_test)


		acc1 = performSVMClass(X_Train, y_Train, X_Test, y_Test)
		print "SVM: " + str(i), acc1
		# acc2 = performKNNClass(X_train, y_train, X_test, y_test)
		# print "KNN: " + str(i), acc2
		acc_svm = acc_svm + acc1
		# acc_KNN = acc_KNN + acc2
		
	print "Accuracy "+ symbol +" ", acc_svm/10
	# print "Accuracy KNN", acc_KNN/10
	# performKNNClass(X_train, y_train, X_test, y_test)
	# performSVMClass(X_train, y_train, X_test, y_test)
	# performAdaBoostClass(X_train, y_train, X_test, y_test)
	# performGTBClass(X_train, y_train, X_test, y_test)
	# performRFClass(X_train, y_train, X_test, y_test)

def predict_next6Day_trend(stockToPredict, symbol):
	# print stockToPredict
	l = [sp500, nasdaq, stockToPredict]
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
		print "SVM: " + str(i), acc1
		# acc2 = performKNNClass(X_train, y_train, X_test, y_test)
		# print "KNN: " + str(i), acc2
		if i < 4:
			acc_svm = acc_svm + acc1
		# acc_KNN = acc_KNN + acc2
		
	print "Accuracy "+ symbol +" ", acc_svm/4


sp500 = pd.read_csv("sp500_with_features.csv", sep = "\t", index_col="Date")
nasdaq = pd.read_csv("nasdaq_with_features.csv", sep = "\t", index_col="Date")
frankfurt = pd.read_csv("frankfurt_with_features.csv", sep = "\t", index_col="Date")
london = pd.read_csv("london_with_features.csv", sep = "\t", index_col="Date")
hkong = pd.read_csv("hkong_with_features.csv", sep = "\t", index_col="Date")
# paris = pd.read_csv("paris_with_features.csv", sep = "\t", index_col="Date")
# australia = pd.read_csv("australia_with_features.csv", sep = "\t", index_col="Date")
# djia = pd.read_csv("djia_with_features.csv", sep = "\t", index_col="Date")

print "===============Predicting Next Day Trend====================="
print ""
# print "Predicting frankfurt's trend"
# predict_nextDay_trend(frankfurt, "frankfurt")
# print ""
print "Predicting london's trend"
predict_nextDay_trend(london, "london")
print ""
# print "Predicting hkong's trend"
# predict_nextDay_trend(hkong, "hkong")
# print ""
# print "Predicting paris's trend"
# predict_nextDay_trend(paris, "paris")
# print ""
# print "Predicting australia's trend"
# predict_nextDay_trend(australia, "australia")
# print ""
# print "Predicting djia's trend"
# predict_nextDay_trend(djia, "djia")
print ""
print "===============Predicting Next 6 Days Trend====================="
# print ""
# print "Predicting frankfurt's trend"
# predict_next6Day_trend(frankfurt, "frankfurt")
# print ""
# print "Predicting london's trend"
# predict_next6Day_trend(london, "london")
# print ""
# print "Predicting hkong's trend"
# predict_next6Day_trend(hkong, "hkong")
# print ""
# print "Predicting paris's trend"
# predict_next6Day_trend(paris, "paris")
# print ""
# print "Predicting australia's trend"
# predict_next6Day_trend(australia, "australia")
# print ""
# print "Predicting djia's trend"
# predict_next6Day_trend(djia, "djia")

print "===============THE END============================"