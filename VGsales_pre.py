#!usr/bin/env python
# coding:utf-8

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder  # 编码
from sklearn.model_selection import train_test_split  # 对测试集和训练集进行划分

### 引入各种分类器的包
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn import tree  # 决策树
from sklearn.svm import SVC # 支持向量机

### 评价方法
from sklearn.metrics import  r2_score, explained_variance_score
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.DataFrame(pd.read_csv('vgsales.csv'))

# 查看数据集缺失值
print(df.isnull().sum())


# 这里采用年份平均数填充缺失的年份
def impute_median(series):
    return series.fillna(series.median())


df.Year = df['Year'].transform(impute_median)

# 输出最多的游戏发布商
print(df['Publisher'].mode())
# 这里采用发布商众数填充缺失的发布商
df['Publisher'].fillna(str(df['Publisher'].mode().values[0]), inplace=True)
print(df.isnull().sum())

# 去除游戏发布年份超过2016年的数据
df = df[df.Year < 2016]
print(df.head())

# 对标称属性进行编码
# 将标签存储在内存中，稍后将使用这些标签单独从需要的特性和标签构建数据样式，利用标签编码所有分类标签
categorical_labels = ['Rank', 'Name', 'Platform', 'Genre', 'Publisher']
numerical_labels = ['Year', 'NA_Sales', 'Other_Sales', 'Global_Sales']
enc = LabelEncoder()
encoded_df = pd.DataFrame(columns=['Rank', 'Name', 'Platform', 'Genre', 'Publisher',
                                   'Year', 'NA_Sales', 'Other_Sales', 'Global_Sales'])

for label in categorical_labels:
    temp_column = df[label]

    encoded_temp_col = enc.fit_transform(temp_column)

    encoded_df[label] = encoded_temp_col

for label in numerical_labels:
    temp_column = df[label]

    encoded_temp_col = enc.fit_transform(temp_column)

    encoded_df[label] = encoded_temp_col
    #encoded_df[label] = df[label].values

# 输出编码后的数据格式
print(encoded_df.head())

# 分割测试集和训练集
train, test = train_test_split(encoded_df, test_size=0.1, random_state=1)


def data_splitting(df):
    x = df.drop(['NA_Sales', 'Other_Sales', 'Global_Sales'], axis=1)
    y = df['Global_Sales']
    return x, y


x_train, y_train = data_splitting(train)
x_test, y_test = data_splitting(test)

### 线性回归 LinearRegression
log = LinearRegression()
log.fit(x_train, y_train)

log_train = log.score(x_train, y_train)
log_test = log.score(x_test, y_test)

print("=" * 70)
print("Training score :", log_train)
print("Testing score :", log_test)
print("=" * 70)


print("=" * 70)

### 随机森林分类器 Random Forest Classifier
clf = RandomForestClassifier(n_estimators=8)
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)

# 对分类器进行评价
r2 = r2_score(y_test, clf_pred)
evs = explained_variance_score(y_test, clf_pred)

print(f'Random Forest R2 Coeff: {r2}')
print(f'Random Forest Explained_Variance_Score: {evs}')
print("Random Forest Test Accuracy is {:.2f}%".format(accuracy_score(y_test, clf_pred) * 100.0))

### KNN Classifier
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)

# 对分类器进行评价
r2 = r2_score(y_test, clf_pred)
evs = explained_variance_score(y_test, clf_pred)

print(f'\nKNN R2 Coeff: {r2}')
print(f'KNN Explained_Variance_Score: {evs}')
print("KNN Test Accuracy is {:.2f}%".format(accuracy_score(y_test, clf_pred) * 100.0))

### 决策树Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)

# 对分类器进行评价
r2 = r2_score(y_test, clf_pred)
evs = explained_variance_score(y_test, clf_pred)

print(f'\nDecision Tree R2 Coeff: {r2}')
print(f'Decision Tree Explained_Variance_Score: {evs}')
print("Decision Tree Test Accuracy is {:.2f}%".format(accuracy_score(y_test, clf_pred) * 100.0))

###  SVM Classifier
clf = SVC(kernel='rbf', probability=True)
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)

# 对分类器进行评价
r2 = r2_score(y_test, clf_pred)
evs = explained_variance_score(y_test, clf_pred)

print(f'\nSVM R2 Coeff: {r2}')
print(f'SVM Explained_Variance_Score: {evs}')
print("SVM Test Accuracy is {:.2f}%".format(accuracy_score(y_test, clf_pred) * 100.0))

print("=" * 70)