#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import re


# In[41]:


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
            
#print(edge_weight_array)
print(time_array)
print(len(edge_weight_array))


# In[116]:


y0 = np.asarray(edge_weight_array[0:89])
y1 = np.asarray(edge_weight_array[90:179])
y2 = np.asarray(edge_weight_array[180:269])
y3 = np.asarray(edge_weight_array[270:359])

print(y0)

print('y0 min: ', min(y0))
print('y0 max: ', max(y0))

#t_array = np.asarray(time_array)
#t_float = t_array.astype(np.float)

twf = y0.astype(np.float)
twf1 = y1.astype(np.float)
twf2 = y2.astype(np.float)
twf3 = y3.astype(np.float)

t0 = twf[0]
t1 = twf[88] 

#t1 = t_array[90] --> Then get t1-t0, but that gave an error
#t0 = t_array[0]



#range == [0 0.1] positional argument follows keyword argument error message


#std0 = np.std(y0)
#m0 = np.mean(y0)


#nor_hist0 = (y0 - m0)/std0

#plot last time point first for scaling bins properly - w(t2) - w(t0) --> for all 10,000 what is the biggest range of the dataset?
plt.hist(y1-y0, density = True, range = (t1, t0), bins = 4)
#need to figure out how to make the edge


# In[104]:


#t2 = t_float[180]

#range = (t0, t2)

plt.hist(y2-y0, density = True, range = (t1, t0), bins = 4)
#normed = 1 crashed, error message: rectangle object has no property normed


# In[108]:


#t3 = t_float[270] #Gives sum of 34 ish instead of 100

#range = (t0, t3)

plt.hist(y3-y0, density = True, range = (t1, t0), bins = 4)


# In[ ]:




