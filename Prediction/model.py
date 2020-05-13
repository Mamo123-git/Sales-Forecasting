
#import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Sales.csv')

data.fillna('0', inplace = True)

data.fillna(data['sales_in_first_month'].mean(),inplace = True)

X = data.iloc[:,:3]

y = data.iloc[:,-1]

# linear regression

regressor = LinearRegression()

regressor.fit(X,y)

#regressor.score(X,y)


pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


