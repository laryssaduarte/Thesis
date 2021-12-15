#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import category_encoders as ce
from datetime import datetime as dt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns


# In[3]:


"""GET THE DATA, THAT IS PICKLED"""
dataset = pd.DataFrame.from_dict(pd.read_pickle("device1.pkl"))


# In[4]:


dataset.rename(columns = {'device_name': 'V1', '@timestamp': 'V2', 'event_id': 'V3','sap_dc_name': 'V4','event_creation_time': 'V5','event_severity': 'severity','last_response_time': 'response_time','event_source': 'V6','event_timestamp_updated': 'timestamp', 'event_timestamp_started': 'V7','last_test_time': 'V8',}, inplace = True)


# In[23]:


dataset.dtypes


# In[12]:


"""CONVERT THESE COLUMNS FROM OBJECT DTYPE TO INT64, THEY ARE ACTUALLY EPOCH"""
dataset[['timestamp', 'response_time']] = dataset[['timestamp', 'response_time']].apply(pd.to_numeric, axis = 1)


# In[19]:


"""CONVERT THESE COLUMNS FROM OBJECT DTYPE TO INT64, THEY ARE ACTUALLY EPOCH"""
dataset[['V5','severity','V7','V8']] = dataset[['V5','severity','V7','V8']].apply(pd.to_numeric, axis = 1)

"""CONVERT COLUMNS FROM OBJECT DTYPE TO DATETIME"""
dataset['V2'] = dataset['V2'].astype('datetime64')

"""CONVERT THESE COLUMNS FROM OBJECT DTYPE TO CATEGORY"""
#unpiclked_df['event_status'] = unpiclked_df['event_status'].astype('category')


# In[21]:


#heatmap
data = dataset
corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[16]:


dataset['response_time'].plot(figsize = (12,6))


# In[15]:


sns.boxplot(dataset.response_time)


# In[22]:


dataset.info()

