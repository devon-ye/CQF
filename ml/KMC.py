import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ========== 数据加载 ==========
# 1. 加载数据集
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# ========== 数据探索 (EDA) ==========
# 打印数据的头部和描述性统计
print(df.head())
print(df.describe())

# 使用pairplot查看数据的分布和特征间的关系
sns.pairplot(df)
plt.show()

# 检查数据中是否有缺失值
print(df.isnull().sum())

# ========== 数据预处理 ==========
# 使用标准化器标准化数据（由于KMeans对特征的尺度敏感）
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# ========== 特征工程 ==========
# 使用PCA进行特征降维
pca = PCA(n_components=2)
data_2d = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data_2d, columns=['PC1', 'PC2'])

# ========== 模型训练 ==========
# 进行K-Means聚类
kmeans = KMeans(n_clusters=3)
df_pca['cluster'] = kmeans.fit_predict(df_scaled)

# ========== 可视化结果 ==========
# 使用散点图可视化PCA后的数据
plt.figure(figsize=(12, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['cluster'], cmap='viridis', s=50)
plt.title('PCA of Iris Dataset after K-Means Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

# 使用并行坐标轴进行可视化
plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(df.assign(cluster=df_pca['cluster']), 'cluster', colormap='viridis')
plt.title('Parallel Coordinates Plot of Iris Dataset after K-Means Clustering')
plt.show()

# ========== 模型评估 ==========
# 计算不同K值的WCSS来选择最佳K值
wcss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(k_values, wcss, '-o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()

# 使用轮廓系数评估聚类效果
k_values = range(2, 10)
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_scaled)
    labels = kmeans.labels_
    score = silhouette_score(df_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()

# 显示每个群集的特征均值
cluster_means = df.assign(cluster=df_pca['cluster']).groupby('cluster').mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means.T, cmap='coolwarm', annot=True)
plt.title('Feature Means by Cluster')
plt.show()
