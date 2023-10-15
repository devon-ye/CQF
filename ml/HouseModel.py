import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
# 设置 Pandas 显示选项
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 显示所有单元格的内容
# 读取数据集
data = pd.read_csv('../house_data.csv')
#获取表头
header = data.columns


# 数据清洗
# 检测和处理缺失值
data.dropna(inplace=True)
# 处理重复行
data.drop_duplicates(inplace=True)

for column in data.columns:
    if data[column].dtype == 'object' and header[8] != column:
        # 对于字符串列，将空串替换为NaN
        data[column].replace('', np.nan, inplace=True)
        # 将非数字值替换为NaN
        data[column] = pd.to_numeric(data[column], errors='coerce')
    if column == header[1] or column == header[3]:
        # 使用列的中位数替换NaN值
        data[column].fillna(data[column].median(), inplace=True)
    if column == header[5] or column == header[6] or column == header[7]:
        # 使用列的均值替换NaN值
        data[column].fillna(data[column].mean(), inplace=True)
    if header[8] != column and (data[column] < 0).any():
        # 将负数替换为中位数
        data.loc[data[column] < 0, column] =data[column].median()


cleaned_data= data
# 特征工程
#todo

# 类别变量编码
encoder = OneHotEncoder(sparse=False, drop='first')
#地址 都热编码
encoded_data= pd.get_dummies(cleaned_data, columns=[header[8]])
transform_data = encoded_data.astype(int)

# 特征缩放
minMaxScaler = MinMaxScaler()
numerical_features = cleaned_data.head(0)
transform_data[numerical_features] = minMaxScaler.fit_transform(transform_data[numerical_features])


# 划分数据集
X = transform_data.drop(header[7], axis=1)
y = transform_data[header[7]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"R平方值 (R2): {r2}")
