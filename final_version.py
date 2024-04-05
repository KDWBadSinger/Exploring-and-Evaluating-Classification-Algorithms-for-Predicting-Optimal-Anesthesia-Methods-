# 处理UserWarning: KMeans is known to have a memory leak on Windows with MKL
import os

os.environ["OMP_NUM_THREADS"] = '15'

from sklearn import svm
from sklearn.metrics import accuracy_score

# K临近
from sklearn.neighbors import KNeighborsClassifier

# 逻辑回归
# 训练逻辑回归分类器
from sklearn.linear_model import LogisticRegression

# 决策树分类器
# 训练决策树分类器
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

import seaborn as sns
from scipy.stats import norm

df = pd.read_csv('Data.csv', sep=',', header=0)
data = df[df['Label'] != 2]
data_array = data.values

# # 绘制高斯曲线
# # 绘制带有核密度估计的直方图
# sns.histplot(data_array.reshape(-1, 1), kde=True)
# plt.show()
#
# mu, std = norm.fit(data_array)  # 估计数据的均值和标准差
# x = np.linspace(np.min(data_array), np.max(data_array), 100)  # 创建一系列等间距的点
# pdf = norm.pdf(x, mu, std)  # 计算高斯分布的概率密度函数值
# plt.plot(x, pdf, 'r-', label='Gaussian PDF')  # 绘制概率密度函数曲线
# plt.legend()
# plt.show()


# from scipy.stats import shapiro

# # 进行Shapiro-Wilk测试
# statistic, p_value = shapiro(data_array)
#
# # 打印测试结果
# print("Shapiro-Wilk Test:")
# print("Statistic:", statistic)
# print("p-value:", p_value)
#
# # 根据p-value进行判断
# alpha = 0.05  # 设置显著性水平
# if p_value > alpha:
#     print("数据符合高斯分布")
# else:
#     print("数据不符合高斯分布")

# ------------------------------------------------------------------------------

data = data.drop('Patient index', axis=1)
X = data.iloc[:, 0:-1]
scaler = StandardScaler()
dataset = scaler.fit_transform(X)

n_components = 10
pca = PCA(n_components=n_components)

after_pca = pca.fit_transform(dataset)

reduced = pd.DataFrame(after_pca)
reduced['Label'] = data['Label'].values
reduced.to_csv('reduced10.csv', index=False)

df = pd.read_csv('reduced10.csv')

X = data.iloc[:, :-1]  # 特征数据
y = data.iloc[:, -1]  # 标签数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 看看训练集大小
# print(X_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
X_train_removed_bias = rf.apply(X_train)
X_test_removed_bias = rf.apply(X_test)

# SVM
# 训练SVM分类器
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
# 测试精确度
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of SVM is:", accuracy)

# 逻辑回归
clf = LogisticRegression()
clf.fit(X_train, y_train)
# 测试精确度
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of Logistic Regression is:", accuracy)

# 决策树
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
# 测试精确度
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of Decision Tree is:", accuracy)

# K临近
clf = KNeighborsClassifier(n_neighbors=38)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of KNN is:", accuracy)

# 交叉验证评估SVM分类器
svm_clf = svm.SVC(kernel='linear')
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=5)  # 5折交叉验证
svm_accuracy = svm_scores.mean()
print("The Accuracy of SVM with Cross-Validation is:", svm_accuracy)

# 交叉验证评估逻辑回归分类器
lr_clf = LogisticRegression()
lr_scores = cross_val_score(lr_clf, X_train, y_train, cv=5)
lr_accuracy = lr_scores.mean()
print("The Accuracy of Logistic Regression with Cross-Validation is:", lr_accuracy)

# 交叉验证评估决策树分类器
dt_clf = DecisionTreeClassifier()
dt_scores = cross_val_score(dt_clf, X_train, y_train, cv=5)
dt_accuracy = dt_scores.mean()
print("The Accuracy of Decision Tree with Cross-Validation is:", dt_accuracy)

# 交叉验证评估KNN分类器
knn_clf = KNeighborsClassifier(n_neighbors=38)
knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=5)
knn_accuracy = knn_scores.mean()
print("The Accuracy of KNN with Cross-Validation is:", knn_accuracy)

data = df
data = data.drop('Label', axis=1)

# 使用肘部法选择聚类数
inertia = []
for n_clusters in range(1, 11):
    classifier = KMeans(n_clusters=n_clusters, random_state=0)
    classifier.fit(X_train)
    inertia.append(classifier.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()  # 拐点在2处， n_cluster = 2

# 创建一个空列表来存储每个聚类数对应的评分
silhouette_scores = []

for i in range(2, 11):
    n_clusters = i
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    # print(labels)
    score = silhouette_score(data, labels)
    # print('The Silhouette Score of K-means is:', score, " with i = ", i)
    silhouette_scores.append(score)

# 绘制折线图
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('K-means Clustering Results')
plt.show()

print(silhouette_scores)

cluster_labels = kmeans.fit_predict(data)

# 可视化聚类结果
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(data)
plt.scatter(X_2d[cluster_labels == 0, 0], X_2d[cluster_labels == 0, 1], label="cluster 0")
plt.scatter(X_2d[cluster_labels == 1, 0], X_2d[cluster_labels == 1, 1], label="cluster 1")
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.title('PCA Visualization of Clustering Results')
plt.show()