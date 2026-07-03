# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 19:38:12 2025

@author: Aqueak
"""

#Kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble  import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,  r2_score


#Veri setini yükleme
df=yf.download("AAPL", start="2025-01-01", end="2025-11-14")
df.head()

#Özellik üretimi(Moving Averages)
df["MA5"]=df["Close"].rolling(5).mean()
df["MA20"]=df["Close"].rolling(20).mean()
df["MA50"]=df["Close"].rolling(50).mean()
df=df.dropna()

#X ve y



X=df[["Open","Low","High","Volume","MA5","MA20","MA50"]]
y=df["Close"].values.ravel()


#Eğitim ve test
X_train, X_test, y_train, y_test=train_test_split(
    X, y,  test_size=0.2 ,shuffle=False)

#Normalizasyon
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Model Olulşturma
model=RandomForestRegressor(n_estimators=300,random_state=42)
model.fit(X_train_scaled,y_train)

#Tahmin yap
y_pred=model.predict(X_test_scaled)


#Başarı Oranı
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(mae)
print(r2)

plt.figure(figsize=(12,6))
plt.plot(y_test,label="Gerçek fiyat",color="blue")
plt.plot(y_pred,label="Tahmin",color="red")
plt.title("Gerçek v Tahmin")
plt.xlabel("Gün")
plt.ylabel("Fiyat")
plt.legend()
plt.show()
