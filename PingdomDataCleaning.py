#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import category_encoders as ce
from datetime import datetime as dt


# In[2]:


"""GET THE DATA, THAT IS PICKLED"""
unpiclked_df = pd.DataFrame.from_dict(pd.read_pickle("HistoriesDevice18.pkl"))


# In[15]:


"""GET THE DATA, THAT IS PICKLED"""
dataset = pd.DataFrame.from_dict(pd.read_pickle("device1.pkl"))

dataset.rename(columns = {'device_name': 'V1', '@timestamp': 'V2', 'event_id': 'V3','sap_dc_name': 'V4','event_creation_time': 'V5','event_severity': 'severity','last_response_time': 'response_time','event_source': 'V6','event_timestamp_updated': 'timestamp', 'event_timestamp_started': 'V7','last_test_time': 'V8',}, inplace = True)


# In[23]:


unpiclked_df = dataset = pd.DataFrame.from_dict(pd.read_pickle("device1.pkl"))


# In[17]:


""" HAVE AN INSIGHT ABOUT THE DATA TO TRANSFORM"""
unpiclked_df.describe()


# In[7]:


unpiclked_df.isnull().sum()


# In[24]:


unpiclked_df.info()


# In[86]:


""" CHECK IF COLUMNS ARE IDENTICAL """
unpiclked_df['event_creation_time'].equals(unpiclked_df['event_timestamp_started'])


# In[4]:


""" DROP COLUMNS BC DONT ADD USEFUL DATA """
unpiclked_df.drop(['device_name','event_source', '@timestamp','event_source','last_test_time'], axis=1, inplace=True)


# In[6]:


""" DROP LINES TO FIT THE LEAST POPULOUS COLUMN  """
unpiclked_df.dropna(subset=['event_timestamp_started'],inplace=True)


# In[26]:


""" CHECK COLUMNS TYPE"""
unpiclked_df.dtypes


# In[27]:


"""CONVERT THESE COLUMNS FROM OBJECT DTYPE TO INT64, THEY ARE ACTUALLY EPOCH"""
unpiclked_df[['event_creation_time', 'event_timestamp_started','event_timestamp_updated','last_response_time']] = unpiclked_df[['event_creation_time', 'event_timestamp_started','event_timestamp_updated','last_response_time']].apply(pd.to_numeric, axis = 1)


# In[34]:


"""CONVERT VALUES FROM EPOCH TO DATETIME"""

"""Create a function to apply to each row of the data frame"""
def epochToDatetime(value):
	"""Converts epoch value to datetime """
	return dt.fromtimestamp(value)

# Apply that function to every row of the columns : 'event_creation_time', 'event_timestamp_started','event_timestamp_updated','last_test_time'
unpiclked_df['event_creation_time_DT']=unpiclked_df['event_creation_time'].apply(epochToDatetime)
unpiclked_df['event_timestamp_started_DT']=unpiclked_df['event_timestamp_started'].apply(epochToDatetime)
unpiclked_df['event_timestamp_updated_DT']=unpiclked_df['event_timestamp_updated'].apply(epochToDatetime)

# Check the data output
unpiclked_df.head()


# In[38]:


"""CONVERT THESE COLUMNS FROM OBJECT DTYPE TO INT64, THEY ARE ACTUALLY EPOCH"""
unpiclked_df[['event_timestamp_updated','last_response_time']] = unpiclked_df[['event_timestamp_updated','last_response_time']].apply(pd.to_numeric, axis = 1)

def epochToDatetime(value):
	"""Converts epoch value to datetime """
	return dt.fromtimestamp(value)

unpiclked_df['event_timestamp_updated_DT']=unpiclked_df['event_timestamp_updated'].apply(epochToDatetime)


# In[32]:


unpiclked_df.head()


# In[31]:


unpiclked_df['event_timestamp_updated_DT'].plot(figsize = (12,6))


# In[65]:


"""CONVERT COLUMNS FROM OBJECT DTYPE TO DATETIME"""
unpiclked_df['@timestamp'] = unpiclked_df['@timestamp'].astype('datetime64')


# In[21]:


"""CONVERT THESE COLUMNS FROM OBJECT DTYPE TO CATEGORY"""
unpiclked_df['status'] = unpiclked_df['event_status'].astype('category')


# In[94]:


unpiclked_df.dtypes


# In[68]:


unpiclked_df['sap_dc_name'].isna().sum()


# In[100]:


""" LABEL ENCODING - ASSIGN THE ENCODED VARIABLES TO A NEW COLUMN""" 
unpiclked_df["event_status_Categ"] = unpiclked_df["event_status"].cat.codes


# In[70]:


#"""CONVERT COLUMN FROM OBJECT DTYPE TO BOOL"""
#unpiclked_df['change_in_status'] = unpiclked_df['change_in_status'].astype('bool')


# In[71]:


"""DROP UNKNOWN STATUS"""
unpiclked_df.drop(unpiclked_df[unpiclked_df['event_status'] == "unknown"].index, inplace = True)


# In[72]:


"""DROP unconfirmed_down STATUS"""
unpiclked_df.drop(unpiclked_df[unpiclked_df['event_status'] == "unconfirmed_down"].index, inplace = True)


# In[13]:


unpiclked_df['event_status'].value_counts()


# In[98]:


unpiclked_df[unpiclked_df['event_status'] == "unknown"]


# In[14]:


""" CHANGE STATUS UP/DOWN TO NUMERIC """
unpiclked_df["event_status_01"]=unpiclked_df['event_status'].replace({"up": 3, "down": 0, "unconfirmed_down": 1, "unknown": 2})


# In[15]:


"""CHANGE event_severity PATTERN TO HAVE MEANING 1(LOW) - 4(HIGH) 
    OLD: 1(HIGH, down),3(unconfirmed_down),4(unknown),5(LOW, up)
    NEW: 1(LOW, up),2(unconfirmed_down),3(unknown),4(HIGH, up)"""

unpiclked_df["event_severity_1to4"]=unpiclked_df["event_severity"].replace({1: 4, 3: 3,4: 2,5:1})


# In[102]:


unpiclked_df['event_severity_1to4'].value_counts()


# In[20]:


unpiclked_df.index = unpiclked_df.event_timestamp_updated_DT
unpiclked_df.drop(['event_timestamp_updated_DT'], axis=1, inplace=True)
unpiclked_df = unpiclked_df.sort_index()
unpiclked_df.head()


# In[21]:


"""REDUCE THE TIMELINE"""
unpiclked_df['last_response_time'].plot(figsize = (12,6))


# In[22]:


df1 = unpiclked_df[(unpiclked_df.index >= '2021-10-26')]


# In[24]:


df1 = df1[['last_response_time','event_status_01','event_severity_1to4']]
df1.columns = ['response_time','status','severity']


# In[28]:


"""REDUCE THE TIMELINE"""
df1['response_time'].plot(figsize = (12,6))


# In[77]:


#"""ENCODE HOSTNAME"""
#Create an object for Base N Encoding
#encoder= ce.BaseNEncoder(cols=['hostname'],return_df=True,base=8)
#Fit and Transform Data
#host_encoded=encoder.fit_transform(unpiclked_df)
#host_encoded


# In[78]:


#"""CALCULATE EVENT DURATION"""
# Calculate difference between lastdownend_DT and lastdownstart_DT
#host_encoded['duration'] = (unpiclked_df.lastdownend_DT - unpiclked_df.lastdownstart_DT)
#host_encoded['duration'].head()


# In[79]:


#"""CONVERT DURATION TO MINUTES"""
# Create a function to apply to each row of the data frame
#def timedeltaToMinutes(duration):
#	"""CONVERT TIMEDELTA TO MINUTES """
#	return duration.total_seconds() / 60

# Apply that function to every row of the column
#host_encoded['duration_minutes']=host_encoded['duration'].apply(timedeltaToMinutes)

# Check the data output
#host_encoded.head()


# In[26]:


"""RESULTS"""
df1.info()


# In[27]:


"""SAVE THE DATA IN A NEW FILE"""
unpiclked_df.to_pickle("orderedDatasetDevice18.pkl")

