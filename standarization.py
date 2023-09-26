from sklearn.preprocessing import StandardScaler  
from sklearn.pipeline import make_pipeline  

#standarization in order to minimise the coeffients' range 
scaler = StandardScaler()  
standard_coefficient_linear_reg = make_pipeline(scaler, model)
standard_coefficient_linear_reg.fit(X,y)


