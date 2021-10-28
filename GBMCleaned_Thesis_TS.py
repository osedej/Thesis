import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from scipy import stats

df= pd.read_excel('Compilation.xlsx')

df1=df[['Diameter', 'Length', 'Temperature']]
df2=df['Ratio']
Y= df2

#scale the data before selection
X = preprocessing.StandardScaler().fit_transform(df1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

GBM = GradientBoostingRegressor(n_estimators=100000, learning_rate=0.05, max_features=3, max_depth=3, random_state=0)
GBM.fit(X_train, Y_train)

print('accuracy train:', GBM.score(X_train, Y_train))
print('accuracy testing:', GBM.score(X_test, Y_test))

#correlation Matrix
correlations = df[['Diameter', 'Length', 'Temperature', 'Ratio']].corr(method='spearman')
