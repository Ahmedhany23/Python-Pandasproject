import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('forproject.xlsx')

d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
series = pd.DataFrame(d)
print(series)

# Adding a new column

series['three'] = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(series)
# Adding a new colum with the existing column
series['four'] = series['one'] + series['three']
print(series)
column = df.drop(columns="Country")
print(column)
twocolumn = df.drop(labels=["Year", "Rank"], axis=1)
print(twocolumn)
#  df.drop(["Year","Rank"],axis=1,inplace=True)
dfaxis = df.drop(labels=0, axis=0)
print(dfaxis)
multilabels = df.drop(labels=[1, 2, 3, 4], axis=0)
print(multilabels)
first = df.drop(0)
print(first)
# df.drop(labels=range(1,5),inplace=True)
second = df.drop(range(1, 5))
print(second)
# delete rows based on index value
data = df.set_index("Country").head()
data.drop("Yemen", inplace=True)
data = data.drop(data.index[[1, 2, 3]])
print(data)
print(df.shape)
print("---------table shape---------")
df.drop(df.index[range(5)], inplace=True)
df.drop("Country", axis=1, inplace=True)
print(df.shape)

print("-----------using loc---------------")
dataloc = df.loc[(df["C1: Security Apparatus"] == 9.7)]
print(dataloc.shape)
print("--------------rename------------")
dataR = {'Name': ['John', 'Doe', 'Paul'], 'age': [22, 13, 15]}

ss = pd.DataFrame(dataR)
print(ss)
ssR = ss.rename(columns={'Name': 'FirstName', 'age': 'AGe'})
print(ssR)
print(df.head())
print(df.dtypes)
print(df.isna().sum())
datadrop = df.dropna()
print(datadrop.isna().sum())

print("--------MEAN---------")
x1 = df["X1: External Intervention"].mean()
mean = df["X1: External Intervention"].fillna(x1)
print(mean)
print("--------MEDIAN---------")
x2 = df["X1: External Intervention"].median()
median = df["X1: External Intervention"].fillna(x2)
print(median)
print("--------MODE---------")
x3 = df["X1: External Intervention"].mode()
mode = df["X1: External Intervention"].fillna(x3)
print(mode)

print("----------DUPLICATES--------")

print(df.duplicated().astype(int))
print(df.drop_duplicates(inplace=True))
# dummies = pd.get_dummies(df.Country).astype(int)
# print(dummies)
# merged = pd.concat([df,dummies],axis=1)
# print(merged)
# final = merged.drop(['Country'],axis=1)
# print(final)


datadum = pd.DataFrame({'temperature': ['Hot', 'Cold', 'Warm', 'Cold'], })
print(datadum.head())
dss = pd.get_dummies(datadum).astype(int)
print(dss)
mergeddata = pd.concat([datadum, dss], axis=1)
print(mergeddata)
final = mergeddata.drop(['temperature'], axis=1)
print(final)

x = np.array([[-1000.1], [-200.2], [500.5], [600.6], [9000.9]])
scaler = preprocessing.StandardScaler()
standardized = scaler.fit_transform(x)
print(standardized)

y = pd.DataFrame([[1], [5], [20], [111]])
scaler1 = MinMaxScaler()
stock_df = scaler1.fit_transform(y)
print(stock_df)

dataZ = {'weight': [300, 250, 800], 'price': [3, 2, 5]}

z = pd.DataFrame(dataZ)

print(z)

scalerZ = MinMaxScaler()
Normalized_data = scalerZ.fit_transform(z)

print(Normalized_data)

normalized_data = pd.DataFrame(Normalized_data, columns=z.columns)
print(normalized_data)
