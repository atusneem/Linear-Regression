import math
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression

TRAIN_DATA_FILE = "reg_train.csv"
SPLIT_PERC = 0.2

#read the train file and return the data as two lists (ind and dep variables)
def readData(fname):
	names = ["feature", "prediction"]
	df = pandas.read_csv(fname, names = names)
	x = df[[names[0]]].to_numpy()
	x = x.reshape(x.shape[0],1)

	y = df[[names[1]]].to_numpy()
	y = y.reshape(x.shape[0],1)

	if (len(x) == len(y)):
		print("yes")
	else:
		print("u suck")

	return train_test_split(x, y, test_size=SPLIT_PERC)

def predict(params, x):
	#TODO Fill in prediction functionality
	return params.predict(x);

def calculate(x, y, params):
	prediction = predict(x, params)
	return ((prediction - Y)**2).mean()/2

def printParams(params):
	print("The value of B0 (intercept) is: ", params[0])
	print("The value of B1 (slope) is: ", params[1])

#The linear regression algorithm. Takes a list of lists as input
def lreg(x,y):
	params = 0
	n = 700
	alpha = 0.0001

	b0 = (n,1)
	b1 = (n,1)

	while(params < 1000):
		y_temp = b0 + b1 * x
		error = y_temp - y
		mean_sq_er = np.sum(error**2)
		mean_sq_er = mean_sq_er/n
		b0 = b0 - alpha * 2 * np.sum(error)/n
		b1 = b1 - alpha * 2 * np.sum(error *x)/n
		params += 1
		if(params%10 == 0):
			print(mean_sq_er)

	print("Mean squared error (mpg)^2: %.5f" % (mean_sq_er))
	print("Root Mean Squared Error mpg: %.5f" % (math.sqrt(mean_sq_er)))

	y_prediction = b0 + b1 * x
	y_plot = []
	for i in range(100):
	    y_plot.append(b0 + b1 * i)
	plt.figure(figsize=(10,10))
	plt.scatter(x,y,color='red',label='GT')
	plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
	plt.legend()
	plt.show()

    #TODO estimate the linear regression parameters (B0 and B1) here
	#DO NOT USE SciKit Learn functionality here

	return params
	#params.append(b0)
	#params.append(b1)
	#return params

#this is the main routine of the program. You should not have to modify anything here
if __name__ == "__main__":
	xTrain, xTest, yTrain, yTest = readData(TRAIN_DATA_FILE)
	parameters = lreg(xTrain,yTrain)
	#printParams(parameters)

	#TODO Validation metrics and visualization
	#I did this part in linreg because I kept getting errors
