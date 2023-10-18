#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('house3.csv')


X = data[['bedrooms', 'bathrooms', 'sqft_living']]  
y = data['price']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


new_data = np.array([[4,3, 2200]])
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price[0])


# In[4]:


print(data.head())


# In[5]:


print(data.tail())


# In[6]:


import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
plt.bar(["MSE"], [mse], color='skyblue')
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.title("Mean Squared Error (MSE)")
plt.show()


plt.figure(figsize=(8, 6))
plt.bar(["R-squared"], [r2], color='lightcoral')
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.title("R-squared (R2) Score")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_pred, bins=20, color='lightgreen')
plt.xlabel("Predicted Prices")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Prices")
plt.show()


# In[ ]:




