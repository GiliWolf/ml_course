import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression, RANSACRegressor

diabetes = datasets.load_diabetes()
# print(list(diabetes['feature_names']))
data = diabetes['data']
x = data[:,2].reshape(-1,1)
# print(x)
y = diabetes['target']
# print(y)
model = LinearRegression()
model.fit(x,y)
# print(model.coef_)
# x_new = pd.Series(x.flatten())
sns.regplot(x = x,y = y)
plt.show()
