#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


data=pd.read_csv("archive (14).zip")
data.head()


# In[3]:


data


# In[4]:


data["play"].value_counts()


# In[21]:


Py = 9/14
Pn = 5/14


# In[6]:


print(py)
print(pn)


# In[7]:


#outlook
pd.crosstab(data["outlook"],data["play"])


# In[20]:


#temp
Pon=0
Prn=2/5
Psn=3/5

Poy=4/9
Pry=3/9
Psy=2/9


# In[11]:


# temperature
pd.crosstab(data["temp"],data["play"])


# In[12]:


PCoolNo=1/5
PHotNo=2/5
PMildNo=2/5

PCoolNo=3/9
PHotYes=2/9
PMildYes=4/9


# In[13]:


#humidity
pd.crosstab(data["humidity"],data["play"])


# In[14]:


PHighNo=4/5
PNormalNo=1/5

PHighYes=3/9
PNormalYes=6/9


# In[16]:


#wind 
pd.crosstab(data["windy"],data["play"])


# In[17]:


PFalseNo=2/5
PTrueNo=3/5

PFalseYes=6/9
PTrueYes=3/9


# In[23]:


# problems 
#Outlook=sunny temp=hot, humididty=high, wimd=true

Pyes=Py*Psy*PHotYes*PHighYes*PFalseYes
print(Pyes)


# In[25]:


Pno=Pn*PHotNo*PHighNo*PTrueNo
print(Pno)

