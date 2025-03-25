#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd

import os
for dirname,_, filenames in os.walk("/kaggle/input"):
    for fileneame in filenames:
        print(os.path.join(dirname,filename))


# In[30]:


pip install kaggle 


# In[32]:


df  = pd.read_csv("archive (11).zip")


# In[33]:


df.head()


# In[34]:


from  sklearn.preprocessing import LabelEncoder


# In[35]:


encoder = LabelEncoder()


# In[36]:


df = df[df["Species"]!=0][["SepalWidthCm","PetalLengthCm","Species"]]


# In[37]:


df.head()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[46]:


import matplotlib.pyplot as plt

# Convert 'Species' column into numeric values
df["Species_numeric"] = pd.factorize(df["Species"])[0]

# Create scatter plot
plt.scatter(df["SepalWidthCm"], df["PetalLengthCm"], c=df["Species_numeric"], cmap="winter")

# Add labels
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Iris Dataset Scatter Plot")
plt.colorbar(label="Species")  # Shows color mapping

plt.show()


# In[47]:


# 10 rows for training 

df = df.sample(100)
df_train = df.iloc[:60,:].sample(10)
df_val = df.iloc[60:80,:].sample(5)
df_test = df.iloc[80:,:].sample(5)


# In[48]:


df_train


# In[50]:


df_test


# In[51]:


df_val


# In[52]:


X_test = df_val.iloc[:,0:2].values
y_test = df_val.iloc[:,-1].values


# In[53]:


# bagging case  1 

df_bag = df_train.sample(8,replace=True)

X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

df_bag


# In[55]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score


# In[56]:


df_bag1 = DecisionTreeClassifier()


# In[60]:


df_bag1 = DecisionTreeClassifier()  # Create the model
df_bag1.fit(X, y)  # Train the model

# Now it should work
evaluate(df_bag1, X, y)


# In[61]:


from sklearn.metrics import accuracy_score

def evaluate(model, X, y):
    """Evaluates the model using accuracy score."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    return acc

# Example usage:
# evaluate(df_bag1, X, y)  # Ensure df_bag1 is a trained model


# In[62]:


evaluate(df_bag1,X,y)

