#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import re


# In[3]:


#open and parse files for necessary data --> testing for 1 .out file to make sure everything can run
#utilized Derek's parse code and made adjustments
with open('./0.out') as f:
    lines = f.read().splitlines()

    edge_weight_array = []
    time_array = []

    for line in lines:
        edge_weight_pat = "tw: Edge From Node (\d{3}) to Node (\d{3}) ; Edge Speed : 0.5000000000000000000000 ; Edge Weight: ([\d\.-]+) ; Current Time: ([\d\.]+)"
        matches = re.search(edge_weight_pat, line)
        if matches:
            #matches in order of info is extracted
            match_list1 = (float(matches.group(3)))
            match_list2 = (float(matches.group(4)))
            
            #adds elements from .out file to the empty list
            edge_weight_array.append(match_list1)
            time_array.append(match_list2)
            
print(edge_weight_array)
print(time_array)
print(len(edge_weight_array))


# In[22]:


#turning into array
ta1 = np.asarray(time_array)
edw1 = np.asarray(edge_weight_array)

#converting elements into float data type
ta = ta1.astype(float)
edw = edw1.astype(float)

#equating edge weights from a specific time frame (90 weights for connection of 1 specific node to another per time point)
y0 = pd.Series(edw[ta==0.00])
print(y0) #testing for sanity


# In[29]:


y1 = pd.Series(edw[ta==2.0000000000000004])
print(y1)

#scale the index properly - from Vivek's and Vikash's code!
y0.reset_index(inplace=True, drop=True)
y1.reset_index(inplace=True, drop=True)

dat = y1-y0
#getting range by finding the min and max
dat_min1 = min(dat)
dat_max1 = max(dat)

#sanity check
print(dat_min)
print(dat_max)

#FINALLY! :D
plt.hist(dat, density = True, range = (dat_min1, dat_max1), bins = 50)


# In[33]:


y3 = pd.Series(edw[ta==0.9999999999999999])
print(y3)
y3.reset_index(inplace=True, drop=True)

dat2 = y3-y0
dat_min3 = min(dat2)
dat_max3 = max(dat2)
plt.hist(dat2, density = True, range = (dat_min3, dat_max3), bins = 50)


# In[34]:


y3 = pd.Series(edw[ta==1.5000000000000002])
print(y2)
y2.reset_index(inplace=True, drop=True)

dat3 = y2-y0
dat_min2 = min(dat1)
dat_max2 = max(dat1)
plt.hist(dat3, density = True, range = (dat_min2, dat_max2), bins = 50)


# In[35]:


y4 = pd.Series(edw[ta==0.1])
print(y4)
y4.reset_index(inplace=True, drop=True)

dat4 = y4-y0
dat_min2 = min(dat4)
dat_max2 = max(dat4)
plt.hist(dat4, density = True, range = (dat_min2, dat_max2), bins = 50)


# In[ ]:




