import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:, :-1].values #all the lines, all the columns except the last one
y = data.iloc[:,-1].values #dependent variable

#splitting into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#traing to understand number of years of employers and their respective salary
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
print(regression.fit(X_train, y_train))

#predicting test set results
y_predicted = regression.predict(X_test)

#visualising training set results
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regression.predict(X_train), color = 'red')
plt.title("Salary vs Experience (Training Data)")
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

#visualizing test set results
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regression.predict(X_train), color = 'red') #same regression line for train and test
plt.title("Salary vs Experience (Test Data)")
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()
#predicted salaries are very close to real salaries
