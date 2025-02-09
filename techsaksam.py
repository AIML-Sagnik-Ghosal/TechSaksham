# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:04:39 2025

@author: SAGNIK GHOSHAL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("C:\\Users\SAGNIK GHOSHAL\Downloads\dataset\dataset\\shopping_trends_updated.csv")
df_8=df.groupby(['Payment Method','Gender'])['Customer ID'].count()
print(df_8)
sns.barplot(df,x='Payment Method' , y = 'Purchase Amount (USD)',estimator='sum',hue='Shipping Type')
plt.savefig("Techsaksham/1.jpg")
plt.show()
df_9  = df.groupby(['Gender','Discount Applied'])['Purchase Amount (USD)'].sum().reset_index()
print(df_9)
plt.pie(list(df_9['Purchase Amount (USD)']), labels = ['Female and Not Applied','Male and Not Applied','Male and Applied'],)
plt.savefig("Techsaksham/2.jpg")
plt.show()
df_12=df.groupby('Category')['Shipping Type'].value_counts()
print(df_12)
df_t=df.copy()
df_t['Age_category'] = pd.cut(df['Age'], bins= [ 18 ,24, 35 , 50 , 70] , labels= ['teen' , 'Young Adults' ,'Middle-Aged Adults', 'old'])
age_count=df_t.groupby('Age_category')['Customer ID'].count()
print(age_count)
df_10 = df_t.groupby('Frequency of Purchases')['Age_category'].value_counts()
print(df_10)
d=['Fortnightly', 'Weekly', 'Annually', 'Quarterly', 'Bi-Weekly', 'Monthly', 'Every 3 Months']
v=[26,52,1,3,104,12,4]
df_t=df_t.replace(d,v)
df_10=df_t.groupby('Age_category')['Frequency of Purchases'].mean()
print(df_10)
plt.bar(['teen' , 'Young Adults' ,'Middle-Aged Adults', 'old'],df_10)
plt.xlabel('Age Category')
plt.ylabel('Average visits per year')
plt.savefig("Techsaksham/3.jpg")
plt.show()
df_13 = df.groupby(['Discount Applied','Gender'])['Purchase Amount (USD)'].sum().reset_index()
print(df_13)
sns.barplot(df, x = 'Category' , y = 'Purchase Amount (USD)',estimator='sum',hue='Size')
plt.savefig("Techsaksham/4.jpg")
plt.show()
df_17 = df.groupby('Category')['Purchase Amount (USD)'].sum().reset_index()
print(df_17)
sns.barplot(df ,y = 'Purchase Amount (USD)' , hue= 'Category',estimator='sum',x='Season')
plt.savefig("Techsaksham/5.jpg")
plt.show()
df_t=df.copy()
c=[i for i in df_t.columns if df_t.dtypes[i]=='object']
for i in c:
    ls=LabelEncoder()
    df_t[i]=ls.fit_transform(df_t[i])
sns.heatmap(df_t.corr(),annot=False)
plt.savefig("Techsaksham/6.jpg")
plt.show()
X = df_t.drop(columns=['Purchase Amount (USD)', 'Customer ID','Promo Code Used','Color'])  # Remove target and unique ID
y = df_t['Purchase Amount (USD)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2 Score": r2}
    print(f"\n{name} Performance:")
    for i in results[name]:
        print(f"\t{i}  -  {results[name][i]}")