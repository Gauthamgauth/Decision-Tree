#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("archive (6).zip")


# In[3]:


df.head()


# In[4]:


df.shape


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[11]:


X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


# In[12]:


rf = RandomForestClassifier(oob_score=True)


# In[14]:


rf.fit(X_train,y_train)


# In[21]:


rf.oob_score_


# In[18]:


y_pred = rf.predict(X_test)
accuracy_score(y_pred,y_test)

