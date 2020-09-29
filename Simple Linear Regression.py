#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# ## My First Program in Python

# In[18]:


import os
#working with operating system
os.getcwd()
#get the working directory


# In[22]:


os.chdir("C:\\Users\\DELL\\Desktop\\DSP25\\28th March 2020")


# In[23]:


import numpy as np
#dealing with multi dimensional array
import pandas as pd
#import the data file & manipulate with datas
import seaborn as sns
#Statistics
import matplotlib.pyplot as plt
#visualization purpose


# In[25]:


#Read Datafile
df=pd.read_csv('C:\\Users\\DELL\\Desktop\\DSP25\\28th March 2020\\Salary_Data.csv')


# In[26]:


df


# In[41]:


#Data preprocessing
#Slicing the data
x=df.iloc[:,:-1].values
#coverting to array format using .iloc


# In[42]:


#IDV
x


# In[39]:


#DV(output y)
y=df.iloc[:,1].values


# In[40]:


y


# In[54]:


y_test


# In[ ]:


#if you have 2.7 version go with below
#from sklearn.cross_validation 


# In[44]:


#if you have 3.7 version
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#test size by default train - 75% & test 25%


# In[45]:


len(x_train)


# In[46]:


len(x_test)


# In[47]:


len(y_train)


# In[48]:


len(y_test)


# In[49]:


#Model Building - import machine learning algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[50]:


#Model Prediction
y_pred = regressor.predict(x_test)


# In[51]:


y_pred


# In[52]:


y_test


# In[53]:


#co_efficient
regressor.coef_


# In[55]:


#Experience = 3 <-- assumption
#y=co_eff * IDV + Intercept Value(constant) -- simple regression
#y=co_eff * IDV +co_eff * IDV+co_eff * IDV+co_eff * IDV+...... Intercept Value(constant) -- Multiple regression
salary = 9898.12403493*3+112635.


# In[62]:


salary


# In[63]:


regressor.intercept_


# In[60]:


salary1 = 9898.12403493*2+23101.596361240066


# In[61]:


salary1


# In[64]:


#Accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[68]:


#Visualization
plt.scatter(x_train,y_train, color = "red")
plt.plot(x_train,regressor.predict(x_train),color="blue")


# In[ ]:




