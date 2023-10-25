# Import data manipulation libraries
import pandas as pd
import numpy as np

# Import yahoo finance library
import yfinance as yf

# Import cufflinks for visualization
import cufflinks as cf
cf.set_config_file(offline=True)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 显示所有单元格的内容

#help(yf)

#df1 = yf.download('SPY', period='5d', progress=False)
df2 = yf.download('SPY', start='2020-01-01', end='2023-10-17', progress=False)
#df3 = yf.download('SPY', period='ytd', progress=False)
#print(df3)
pd.DataFrame(df2).to_csv('20200101_SPY.csv')





