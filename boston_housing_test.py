import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_url = "http://lib.stat.cmu.edu/datasets/boston"
# Define the list of values to be treated as NaN
na_values_list = ['NA', 'N/A', 'missing', 'NaN']  # Customize this list based on your data

# Read the CSV file while specifying the na_values parameter
df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
df.dropna()
data = np.hstack([df.values[::2, :], df.values[1::2, :2]])
target = df.values[1::2, 2]
# print(df[5])
print(df[10])
x = df[5].dropna().values.reshape(-1,1)
y = df[10].dropna().values
model = LinearRegression()
model.fit(x,y)
print(model.coef_)