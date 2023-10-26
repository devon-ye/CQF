import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. 数据加载

# Define the column names for the dataset
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Load the dataset from UCI machine learning repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_data = pd.read_csv(url, header=None, names=column_names, na_values="?")

# Drop rows with missing values for simplicity
heart_data.dropna(inplace=True)

print(heart_data)

# Now, you can use the KNN workflow code provided previously, replacing the data loading part with the above lines.

X, y = heart_data, heart_data.target

# 2. 数据清洗 (数据集已经是处理过的，这里可以跳过)

# 3. 特征工程
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. 定义变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 6. K值选择
neighbors = list(range(1, 50, 2))
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring="accuracy")
    cv_scores.append(scores.mean())
optimal_k = neighbors[cv_scores.index(max(cv_scores))]

# 7. 模型优化和评估
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)

print(f"Optimal K value: {optimal_k}")
print(f"Optimized Model Accuracy: {score:.4f}")

# 可视化
plt.figure(figsize=(10,6))
plt.plot(neighbors, cv_scores, label="CV Average Score")
plt.axvline(optimal_k, color="red", linestyle="--", label="Optimal K")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("Model accuracy with respect to K")
plt.legend()
plt.show()


# 使用 age 和 thalach 两个特征进行可视化
scaler = StandardScaler()
X_visualize = heart_data[["age", "thalach"]].values
y_visualize = heart_data["target"].values
X_visualize = scaler.fit_transform(X_visualize)

# 根据训练集创建网格来显示决策边界
x_min, x_max = X_visualize[:, 0].min() - 1, X_visualize[:, 0].max() + 1
y_min, y_max = X_visualize[:, 1].min() - 1, X_visualize[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_visualize,y_visualize)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 使用蓝色和红色分别表示没有心脏病和有心脏病的数据点
plt.contourf(xx, yy, Z, alpha=0.4)
scatter = plt.scatter(X_visualize[:, 0], X_visualize[:, 1], c=y_visualize, cmap=plt.cm.RdBu_r, s=20)
plt.xlabel("Age (normalized)")
plt.ylabel("Max Heart Rate (normalized)")
plt.title("Decision Boundary for Heart Disease Classification")
plt.colorbar(scatter)
plt.legend(handles=scatter.legend_elements()[0], labels=["No Heart Disease", "Heart Disease"])
plt.show()