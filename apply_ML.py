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
	classifier = svm.SVC(kernel='linear', gamma=0.1, C=100)
	classifier.fit(X_train, y_train)
	results = classifier.predict(X_test)

	acc = classifier.score(X_test, y_test)

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

	# num_correct = (results == y_test).sum()
	# recall = num_correct / len(y_test)
	# print "SVM model accuracy (%): ", recall * 100, "%"
	
	# return recall*100
	return acc*100
	

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
	new_df = datasets[0].join(datasets[1], how='outer')
	new_df1 = new_df.join(datasets[2], how='outer')

	return new_df1

def predict_nextDay_trend(sp500, nasdaq, stockToPredict, symbol):
	l = [sp500, nasdaq, stockToPredict]
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

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# classifier = svm.SVC()
	# classifier.fit(X_train, y_train)
	# results = classifier.predict(X_test)

	# df = pd.DataFrame(dict(adj=X_test[:,5], return_=X_test[:,50], label=results))

	# fig, ax = plt.subplots()
	# colors = {1:'red', 0:'blue'}
	# ax.scatter(df['adj'],df['return_'], c=df['label'].apply(lambda x: colors[x]))
	# plt.show()
	kf = KFold(len(y), n_folds=5)
	acc_svm=0
	acc_rcf=0
	acc_knn=0

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		acc_svm = acc_svm + performSVMClass(X_train, y_train, X_test, y_test)
		# acc_knn = acc_knn + performKNNClass(X_train, y_train, X_test, y_test)
		# acc_rcf = acc_rcf + performRFClass(X_train, y_train, X_test, y_test)
	print ""
	print "Final SVM Accuracy:  ", acc_svm/5
	# print "Final KNN Accuracy:  ", acc_knn/5
	# print "Final RCF 	Accuracy:  ", acc_rcf/5
	# performRFClass(X_train, y_train, X_test, y_test)


def predict_next6Day_trend(sp500, nasdaq, stockToPredict, symbol):
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
	# print y[6:-40]
	# print y.isnull().values.any()
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
# frankfurt = pd.read_csv("frankfurt_with_features.csv", sep = "\t", index_col="Date")
# london = pd.read_csv("london_with_features.csv", sep = "\t", index_col="Date")
# hkong = pd.read_csv("hkong_with_features.csv", sep = "\t", index_col="Date")
# paris = pd.read_csv("paris_with_features.csv", sep = "\t", index_col="Date")
# australia = pd.read_csv("australia_with_features.csv", sep = "\t", index_col="Date")
# djia = pd.read_csv("djia_with_features.csv", sep = "\t", index_col="Date")
apple = pd.read_csv("apple_with_features.csv", sep = "\t", index_col="Date")
# amazon = pd.read_csv("amazon_with_features.csv", sep = "\t", index_col="Date")
# microsoft = pd.read_csv("microsoft_with_features.csv", sep = "\t", index_col="Date")
# reliance = pd.read_csv("reliance_with_features.csv", sep = "\t", index_col="Date")
# tata = pd.read_csv("tata_with_features.csv", sep = "\t", index_col="Date")

print "===============Predicting Next Day Trend====================="
print ""
print "Predicting apple's trend"
predict_nextDay_trend(sp500, nasdaq, apple, "apple")
print ""
# print "Predicting amazon's trend"
# predict_nextDay_trend(sp500, nasdaq, amazon, "amazon")
# print ""
# print "Predicting microsoft's trend"
# predict_nextDay_trend(sp500, nasdaq, microsoft, "microsoft")
# # print ""
# print "Predicting london's trend"
# predict_nextDay_trend(sp500, nasdaq,london, "london")
# print ""
# print "Predicting hkong's trend"
# predict_nextDay_trend(sp500, nasdaq,hkong, "hkong")
# print ""
# print "Predicting paris's trend"
# predict_nextDay_trend(sp500, nasdaq,paris, "paris")
# print ""
# print "Predicting australia's trend"
# predict_nextDay_trend(sp500, nasdaq,australia, "australia")
# print ""
# print "Predicting djia's trend"
# predict_nextDay_trend(sp500, nasdaq,djia, "djia")
# print ""
# print "Predicting reliance's trend"
# predict_nextDay_trend(sp500, nasdaq,reliance, "reliance")
# print ""
# print "Predicting tata's trend"
# predict_nextDay_trend(sp500, nasdaq,tata, "tata")

# print "===============Predicting Next 6 Days Trend====================="
# # print ""
# print "Predicting apple's trend"
# predict_next6Day_trend(sp500, nasdaq, apple, "apple")
# print ""
# print "Predicting amazon's trend"
# predict_next6Day_trend(sp500, nasdaq, amazon, "amazon")
# print ""
# print "Predicting microsoft's trend"
# predict_next6Day_trend(sp500, nasdaq, microsoft, "microsoft")
# print ""
# print "Predicting london's trend"
# predict_next6Day_trend(sp500, nasdaq, london, "london")
# print ""
# print "Predicting hkong's trend"
# predict_next6Day_trend(sp500, nasdaq, hkong, "hkong")
# print ""
# print "Predicting paris's trend"
# predict_next6Day_trend(sp500, nasdaq, paris, "paris")
# print ""
# print "Predicting australia's trend"
# predict_next6Day_trend(sp500, nasdaq, australia, "australia")
# print ""
# print "Predicting djia's trend"
# predict_next6Day_trend(sp500, nasdaq, djia, "djia")
print ""
# print "Predicting reliance's trend"
# predict_next6Day_trend(sp500, nasdaq, reliance, "reliance")
# print ""
# print "Predicting tata's trend"
# predict_next6Day_trend(sp500, nasdaq, tata, "tata")

# import some data to play with

# X = sp500.iloc[:, :6]  # we only take the first two features.
# # Y = sp500.target
# cols = X.columns

# # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# # plt.figure(2, figsize=(8, 6))
# # plt.clf()

# # Plot the training points
# fig, ax = plt.subplots()

# ax.scatter(X[cols[4]], X[cols[5]])
# plt.show()

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())

# # To getter a better understanding of interaction of the dimensions
# # plot the first three PCA dimensions
# fig = plt.figure(1, figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
#            cmap=plt.cm.Paired)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])

# plt.show()

print "===============THE END============================"