#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets  import make_classification 
from sklearn.tree import plot_tree
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
import matplotlib.pyplot as plt 


# In[3]:


X,y = make_classification(n_features=5,n_redundant=0,n_informative =5, n_clusters_per_class=1)


# In[5]:


df = pd.DataFrame(X,columns=["col1","col2","col3","col4","col5"])
df["target"] = y
print(df.shape)
df.head()


# In[8]:


bag = BaggingClassifier(max_features=2)


# In[12]:


bag.fit(df.iloc[:,:5],df.iloc[:,-1])


# In[14]:


plt.figure(figsize=(12,12))
plot_tree(bag.estimators_[0])
plt.show()


# In[15]:


rf = RandomForestClassifier(max_features=2)


# In[16]:


rf.fit(df.iloc[:,:5],df.iloc[:,-1])


# In[17]:


plt.figure(figsize=(12,12))
plot_tree(rf.estimators_[4])
plt.show()

