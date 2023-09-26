from sklearn import datasets
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
import pandas as pd



# module for spiliting your data into test and train
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = pd.DataFrame(diabetes['data'])
y = pd.DataFrame(diabetes['target'])


#spiliting into test and tain
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#building model, trainning it using the x train and predict both x train and y train
model = LinearRegression()
model.fit(X_train,Y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

#1 test -  residual analysis: 
# visualizing both predications and compare test to train 
plt.scatter(y_train_predict, y_train_predict-Y_train, c = 'blue')
plt.scatter(y_test_predict, y_test_predict - Y_test, c = 'red')
plt.show()

#2 test -  Mean-Squared error (MSE):
from sklearn.metrics import mean_squared_error
print("MSE of train data: ", mean_squared_error(Y_train, y_train_predict))
print("MSE of test data: ", mean_squared_error(Y_test, y_test_predict))

#3 test: R^2:
from sklearn.metrics import r2_score
print("R2 of train data: ", r2_score(Y_train, y_train_predict))
print("R2 of test data: ", r2_score(Y_test, y_test_predict))




