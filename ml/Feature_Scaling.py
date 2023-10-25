import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)
from matplotlib.font_manager import FontProperties

# 设置中文字体路径
myFont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

# Create the dataset
data = {
    '身高 (cm)': [170, 175, 168, 180, 172],
    '体重 (kg)': [65, 72, 58, 80, 68],
    '年龄': [25, 30, 28, 35, 29],
    '脚长 (cm)': [25, 26, 25, 27, 26],
    '收缩压 (mmHg)': [120, 125, 118, 128, 121],
    '胆固醇 (mg/dL)': [190, 200, 185, 210, 195],
    '心率 (bpm)': [70, 72, 68, 75, 71]
}

df = pd.DataFrame(data)

# Define a unique color for each feature
colors = plt.cm.Accent(np.linspace(0, 1, len(df.columns)))

# Scaling methods
scalars = {
    '原始数据': None,
    '最小-最大 缩放器': MinMaxScaler(),
    '标准缩放器': StandardScaler(),
    '鲁棒缩放器': RobustScaler(),
    '最大绝对值缩放器': MaxAbsScaler(),
    '分位数转换器': QuantileTransformer(n_quantiles=5),
    '幂转换器': PowerTransformer(method='yeo-johnson')
}

# Determine the number of rows and columns for the subplots
n_features = len(df.columns)
n_scalars = len(scalars)
n_rows = n_features
n_cols = n_scalars

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 4 * n_features))

for row, feature in enumerate(df.columns):
    for col, (name, scaler) in enumerate(scalars.items()):
        ax = axes[row, col]

        if scaler:
            scaled_data = scaler.fit_transform(df[[feature]])
        else:
            scaled_data = df[[feature]].values

        ax.hist(scaled_data, bins=10, color=colors[row], edgecolor='black')
        if row == 0:
            ax.set_title(name, fontproperties=myFont)
        if col == 0:
            ax.set_ylabel(feature, fontproperties=myFont, rotation=0, labelpad=60, ha='right')

fig.text(0.5, 0.01, 'Value', ha='center', fontproperties=myFont)
fig.text(0.01, 0.5, 'Frequency', va='center', rotation='vertical', fontproperties=myFont)
# Adjust layout
plt.tight_layout(pad=1.0)
plt.subplots_adjust(top=0.90,left=0.12, wspace=0.4, hspace=0.5,bottom=0.08)
plt.suptitle("各特征在不同缩放方法下的分布", fontproperties=myFont)
plt.show()
