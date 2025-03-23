#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas-datareader')


# In[2]:


pip install pandas-datareader


# In[5]:


import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split 
from pandas_datareader import data
from sklearn import metrics 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes


# In[6]:


diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data)


# In[7]:


df.columns = diabetes.feature_names
df["age"] = diabetes.target


# In[8]:


df.head()


# In[11]:


X = df.iloc[:,0:5]
y = df.iloc[:,5]


# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[16]:


RT = DecisionTreeRegressor(criterion="mse",max_depth=5)


# In[18]:


from sklearn.tree import DecisionTreeRegressor

# Corrected model initialization
RT = DecisionTreeRegressor(criterion="squared_error", random_state=42)

# Fit the model
RT.fit(X_train, y_train)


# In[19]:


y_pred = RT.predict(X_test)


# In[20]:


r2_score(y_test,y_pred)


# In[26]:


param_grid = {
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}


# In[27]:


reg = GridSearchCV(DecisionTreeRegressor(),param_grid=param_grid)


# In[28]:


reg.fit(X_train,y_train)


# In[29]:


reg.best_score_


# In[30]:


reg.best_params_

