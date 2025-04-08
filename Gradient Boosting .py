#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


np.random.seed(42)
X = np.random.rand(100,1)-0.5
y = 3*X[:,0]**2 + 0.05 * np.random.randn(100)


# In[3]:


df = pd.DataFrame()


# In[4]:


df["X"] = X.reshape(100)
df["y"] = y 


# In[5]:


df


# In[10]:


plt.scatter(df["X"],df["y"])
plt.xlabel = X 
plt.ylabel = y 
plt.show


# In[11]:


df["pred1"] = df["y"].mean() 


# In[12]:


df


# In[13]:


df["res1"]= df["y"] - df["pred1"]


# In[14]:


df


# In[19]:


plt.scatter(df["X"],df["y"])
plt.plot(df["X"],df["pred1"],color="red")


# In[20]:


from sklearn.tree import DecisionTreeRegressor


# In[22]:


tree1 = DecisionTreeRegressor(max_leaf_nodes=8)


# In[25]:


tree1.fit(df["X"].values.reshape(100,1),df["res1"].values)


# In[27]:


from sklearn.tree import plot_tree
plot_tree(tree1)
plt.show()


# In[28]:


# generating Xtest 
X_test = np.linspace(-0.5,0.5,500)


# In[29]:


y_pred = 0.265458 + tree1.predict(X_test.reshape(500,1))


# In[38]:


plt.figure(figsize=(14,4))
plt.subplot(121)
plt.plot(X_test,y_pred,linewidth=2
         ,color="red")
plt.scatter(df["X"],df["y"])


# In[40]:


df["pred2"] = 0.265458 + tree1.predict(df["X"].values.reshape(100,1))


# In[41]:


df


# In[42]:


df["res2"] = df["y"] - df["pred2"]


# In[43]:


df


# In[44]:


tree2 = DecisionTreeRegressor(max_leaf_nodes=8)


# In[45]:


tree2.fit(df["X"].values.reshape(100,1),df["res2"].values)


# In[46]:


y_pred = 0.265458 + sum(regressor.predict(X_test.reshape(-1,1)) for regressor in [tree1,tree2])


# In[48]:


plt.figure(figsize=(14,4))
plt.subplot(121)
plt.plot(X_test,y_pred,linewidth=2,color="red")
plt.scatter(df["X"],df["y"])
plt.title("X vs y ")


# In[53]:


def gradient_boost(X,y,number,lr,count=1,regs=[],foo=None):
    
    if number == 0 :
        return 
    else:
        # gradient boosting
        
        if count > 1 :
            y = y -regs[-1].predict(X)
            
        else:
            foo=y
            
        tree_reg = DecisionTreeRegressor(max_depth=5,random_state=42)
        tree_reg.fit(X,y)
        
        regs.append(tree_reg)
        
        x1 = np.linspace(-0.5,0.5,500)
        y_pred = sum(lr*regressor.predict(x1.reshape(-1,1))for regressor in regs)

        print(number)
        plt.figure()
        plt.plot(x1,y_pred,linewidth=2)
        plt.plot(X[:,0],foo,"r.")
        plt.show()
        
        gradient_boost(X,y,number-1,lr,count+1,regs,foo=foo)


# In[54]:


np.random.seed(42)
X = np.random.rand(100,1) - 0.5
y = 3*X[:,0]**2 + 0.05 * np.random.randn(100)
gradient_boost(X,y,5,lr=1)

