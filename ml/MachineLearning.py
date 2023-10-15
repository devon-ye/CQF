
import sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

# selecting the estimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import k_means

# selecting the metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
# Import required libraries - data manipulation
import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')

# 设置 Pandas 显示选项
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 显示所有单元格的内容


from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()


boston_data= load_wine()
print(boston_data.keys())

#type(boston_data)

boston_data.data

#Define feature and target
X= boston_data['data']  #feature
y= boston_data['target'] #label



print(boston_data['DESCR'])

boston = pd.DataFrame(boston_data['data'],columns=boston_data['feature_names'])
print(boston.head())

#print(boston.shape)

#print(boston.isnull().sum())


#print(pd.DataFrame(y).isnull().sum())

print(boston.describe().T)

scaler.fit(X)
XT=scaler.transform(X)

stats = np.vstack((X.mean(axis=0),X.var(axis=0),XT.mean(axis=0),XT.var(axis=0))).T
feature_names=boston_data['feature_names']
columns=['unscaled mean','unscaled variance','scaled mean','scaled variance']

df=pd.DataFrame(stats,index=feature_names,columns=columns)


print(df)





