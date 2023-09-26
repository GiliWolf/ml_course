import statsmodels.api as sm
import pandas as pd


# using the statsmodles.api in order to run OLS test and getting a wide statics test summary

# adding a column of the values 1 for all values in order to get a constent value (c)
# to the model, otherwise it affects the slope values(m)
# y = mX --> y = mX +c
X_constant = sm.add_constant(X)

pd.DataFrame(X_constant)
model = sm.OLS(y, X_constant)
lr = model.fit()
lr.summary()