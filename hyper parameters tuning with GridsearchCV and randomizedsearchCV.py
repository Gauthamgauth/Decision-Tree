#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[25]:


df = pd.read_csv("archive (6).zip")


# In[26]:


df.head()


# In[27]:


df.shape


# In[28]:


X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[30]:


print(X_train.shape)
print(y_train.shape)


# In[31]:


rf = RandomForestClassifier()


# In[40]:


gb = GradientBoostingClassifier()
svm = SVC()
lr = LogisticRegression()


# In[33]:


rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[38]:


gb.fit(X_train,y_train)
y_pred = gb.predict(X_test)
accuracy_score(y_test,y_pred)


# In[41]:


svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
accuracy_score(y_test,y_pred)


# In[42]:


lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
accuracy_score(y_test,y_pred)


# In[50]:


rf = RandomForestClassifier(max_samples=0.75,random_state=42)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
accuracy_score(y_pred,y_test)


# In[47]:


from sklearn.model_selection import cross_val_score


# In[49]:


np.mean(cross_val_score(LogisticRegression(),X,y,cv=10,scoring="accuracy"))


# In[53]:


np.mean(cross_val_score(RandomForestClassifier(max_samples=0.75),X,y,cv=10,scoring="accuracy"))


# GridSearchCV

# In[54]:


# number of trees in random forest 
n_estimators = [20,60,100,120]

# number opf fearure split to consider at every split 
max_features = [0.2,0.6,1.0]

# maximum number of levels in tree 
max_depth = [2,8,None]

# number of samples 
max_samples = [0.5,0.75,1.0]


# In[65]:


param_grid = {"n_estimators":n_estimators,
             "max_features":max_features,
             "max_depth":max_depth,
             "max_samples":max_samples}
print(param_grid)


# In[66]:


rf = RandomForestClassifier()


# In[67]:


from sklearn.model_selection import GridSearchCV


# In[68]:


rf_grid = GridSearchCV(estimator=rf,
                      param_grid = param_grid,
                      cv = 5,
                      verbose = 2,
                      n_jobs = -1)


# In[69]:


rf_grid.fit(X_train,y_train)


# In[70]:


rf_grid.best_params_


# In[71]:


rf_grid.best_score_


# RandomSearchCV

# In[81]:


# number of trees in random forest 
n_estimators = [20,60,100,120]

# number opf fearure split to consider at every split 
max_features = [0.2,0.6,1.0]

# maximum number of levels in tree 
max_depth = [2,8,None]

# number of samples 
max_samples = [0.5,0.75,1.0]

# bootstraps 
bootstrap = [True,False]

# minumun no of smaples to split the node
min_samples_split = [2,5]

min_samples_leaf = [1,2]


# In[86]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [0.6, 0.8, 'sqrt'],
    'max_samples': [1.0, 0.8],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]  # Corrected parameter name
}


# In[87]:


from sklearn.model_selection import RandomizedSearchCV


# In[88]:


rf_grid1 = RandomizedSearchCV(estimator=rf,
                            param_distributions=param_grid,
                            cv=5,
                            verbose=2,
                            n_jobs=-1)


# In[89]:


rf_grid1.fit(X_train,y_train)


# In[91]:


rf_grid1.best_score_

