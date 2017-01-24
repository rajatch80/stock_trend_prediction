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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	# tick_marks = np.arange(len(iris.target_names))
	# plt.xticks(tick_marks, iris.target_names, rotation=45)
	# plt.yticks(tick_marks, iris.target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def plot_data(dataset):
	# print dataset.columns
	# print list(dataset.index[2:4])
	# print dataset.iloc[2:4, 50:51]
	d1 = dataset.iloc[:, 5:6]
	d2 = dataset.iloc[:, 50:51]
	d3 = dataset.iloc[:, -1:]
	d4 = d1.join(d2, how='outer')
	d5 = d4.join(d3, how='outer')
	cols = d5.columns
	# print d5.iloc[2:4,:]
	fig, ax = plt.subplots()
	colors = {1:'red', 0:'blue'}
	ax.scatter(d5[cols[0]], d5[cols[1]], c=d5[cols[2]].apply(lambda x: colors[x]))
	plt.show()

def performKNNClass(X_train, y_train, X_test, y_test):

	clf = neighbors.KNeighborsClassifier(n_neighbors=10)
	clf.fit(X_train, y_train)
	results = clf.predict(X_test)
	# print results

	accuracy = clf.score(X_test, y_test)
	# print "KNN accuracy (%): ", accuracy, "%"

	return accuracy*100

def performSVMClass(X_train, y_train, X_test, y_test):
	classifier = svm.SVC()
	classifier.fit(X_train, y_train)
	results = classifier.predict(X_test)

	# colors = {1:'red', 0:'blue'}
	# df = pd.DataFrame(dict(adj=X_test[:,5], return_=X_test[:,50], label=results))

	# fig, ax = plt.subplots()
	# colors = {1:'red', 0:'blue'}
	# ax.scatter(df['adj'],df['return_'], c=df['label'].apply(lambda x: colors[x]))
	# # ax.scatter(X_test[:,5], X_test[:,50], c=y_test_list.apply(lambda x: colors[x]))
	# plt.show()
	# print y_pred
	# cm = confusion_matrix(y_test, results)
	# print cm
	# plt.figure()
	# plot_confusion_matrix(cm)
	# plt.show()

	num_correct = (results == y_test).sum()
	recall = num_correct / len(y_test)
	# print "SVM model accuracy (%): ", recall * 100, "%"

	return recall*100


def performRFClass(X_train, y_train, X_test, y_test):

	clf = RandomForestClassifier(n_estimators=10, n_jobs=1)
	clf.fit(X_train, y_train)

	results = clf.predict(X_test)
	print y_test
	print results
	# print y_test
	num_correct = (results == y_test).sum()
	recall = num_correct / len(y_test)
	print "Random Forest model accuracy (%): ", recall * 100, "%"

	return recall*100

def mergeDataFrames(datasets):
	new_df1 = datasets[0].join(datasets[1], how='outer')

	return new_df1

def predict_nextDay_trend(base, stockToPredict):
	l = [base, stockToPredict]
	df = mergeDataFrames(l)

	dataset = df.fillna(df.mean())
	dataset = (dataset - dataset.mean()) / (dataset.max() - dataset.min())
	# print dataset.iloc[1:9, 62:]

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
	# plot_data(dataset)
	# return 0

	y = dataset.UpDown
	labels = list(set(y))
	y = np.array([labels.index(x) for x in y])

	features = dataset.iloc[:,:-1]
	X = np.array(features)

	kf = KFold(len(y), n_folds=5)
	acc_svm=0
	acc_rcf=0
	acc_knn=0

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		acc_svm = acc_svm + performSVMClass(X_train, y_train, X_test, y_test)
		acc_knn = acc_knn + performKNNClass(X_train, y_train, X_test, y_test)
		# acc_rcf = acc_rcf + performRFClass(X_train, y_train, X_test, y_test)
	print ""
	print "Final SVM Accuracy:  ", acc_svm/5
	print "Final KNN Accuracy:  ", acc_knn/5
	# print "Final RCF 	Accuracy:  ", acc_rcf/5
	# performRFClass(X_train, y_train, X_test, y_test)


def predict_next6Day_trend(base, stockToPredict, symbol):
	l = [base, stockToPredict]
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
	# features = features.fillna(0)
	features = features.iloc[6:-40, :]
	X = np.array(features)

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	kf = KFold(len(y), n_folds=5)
	acc_svm=0
	acc_rcf=0
	acc_knn=0

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		acc_svm = acc_svm + performSVMClass(X_train, y_train, X_test, y_test)
		acc_knn = acc_knn + performKNNClass(X_train, y_train, X_test, y_test)
		# acc_rcf = acc_rcf + performRFClass(X_train, y_train, X_test, y_test)
	print ""
	print "Final SVM Accuracy:  ", acc_svm/5
	print "Final KNN Accuracy:  ", acc_knn/5
	# performKNNClass(X_train, y_train, X_test, y_test)
	print ""


sp500 = pd.read_csv("sp500_with_features.csv", sep = "\t", index_col="Date")
nasdaq = pd.read_csv("nasdaq_with_features.csv", sep = "\t", index_col="Date")
frankfurt = pd.read_csv("frankfurt_with_features.csv", sep = "\t", index_col="Date")
london = pd.read_csv("london_with_features.csv", sep = "\t", index_col="Date")
hkong = pd.read_csv("hkong_with_features.csv", sep = "\t", index_col="Date")
paris = pd.read_csv("paris_with_features.csv", sep = "\t", index_col="Date")
australia = pd.read_csv("australia_with_features.csv", sep = "\t", index_col="Date")
djia = pd.read_csv("djia_with_features.csv", sep = "\t", index_col="Date")
apple = pd.read_csv("apple_with_features.csv", sep = "\t", index_col="Date")
amazon = pd.read_csv("amazon_with_features.csv", sep = "\t", index_col="Date")
microsoft = pd.read_csv("microsoft_with_features.csv", sep = "\t", index_col="Date")

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
