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


ransac = RANSACRegressor()
ransac.fit(x,y)
in_values = ransac.inlier_mask_
out_values = np.logical_not(in_values)
plt.scatter(x[in_values], y[in_values], c = 'blue')
plt.scatter(x[out_values], y[out_values], c = 'red')
plt.show()