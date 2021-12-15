#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns


# In[13]:


rng = np.random.RandomState(42)


# In[43]:


"""IMPORT DATA"""
df = pd.DataFrame.from_dict(pd.read_pickle("device1Cleaned.pkl"))


# In[34]:


""" subsample to have a different timeline than its twim-sample (validation)"""
df1 = df.loc['2021-10-26':'2021-11-15',:]


# In[56]:


metrics_df = df[['event_timestamp_updated_DT','event_severity_1to4','event_status_Categ','event_creation_time','event_timestamp_updated','last_response_time']]


# In[75]:


dataForPlot = df[['event_severity_1to4','event_status_Categ','event_creation_time_DT','event_timestamp_updated_DT','last_response_time']]


# In[59]:


metrics_df.index = metrics_df.event_timestamp_updated_DT


# In[63]:


metrics_df = metrics_df.sort_index()
metrics_df.head()


# In[64]:


metrics_df.drop(['event_timestamp_updated_DT'], axis=1, inplace=True)


# In[76]:


dataForPlot.index = dataForPlot.event_timestamp_updated_DT
dataForPlot = dataForPlot.sort_index()
dataForPlot.head()


# In[67]:


metrics_df = metrics_df.loc['2021-10-26':'2021-11-15',:]
dataForPlot = dataForPlot.loc['2021-10-26':'2021-11-15',:]


# In[69]:


metrics_df


# In[143]:





# In[68]:


#MODEL

clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.05),                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(metrics_df)
pred = clf.predict(metrics_df)
metrics_df['anomaly']=pred
outliers=metrics_df.loc[metrics_df['anomaly']==-1]
outlier_index=list(outliers.index)
#Find the number of anomalies and normal points. Points classified -1 are anomalous
print(metrics_df['anomaly'].value_counts())


# In[70]:


sns.scatterplot(data=metrics_df, x= metrics_df.index, y="last_response_time", hue="anomaly", palette="deep")


# In[83]:


"""FUNCTION TO APPLY TO EACH ROW AND CALCULATE MEAN OF LAST_RESPONSE_TIME FROM LAST 24H"""
def getMean(currentTime,responseTime):
    """Get all records from last 24h """
    last24h = dataForPlot.loc[(dataForPlot['event_timestamp_updated_DT'] >= (currentTime - timedelta(hours=24))) & (dataForPlot['event_timestamp_updated_DT'] <= currentTime)]
    return last24h['last_response_time'].mean()

"""Select the columns that are involved in the calculation as a subset of the original data frame, and use the apply function to it.
In the apply function,the parameter axis=1 indicates that the x in the lambda represents a row, so we can unpack the x with *x and pass it to getMean"""

dataForPlot["mean_responseTime24h"] = dataForPlot[["event_timestamp_updated_DT", "last_response_time"]].apply(lambda x : getMean(*x), axis=1)


# In[84]:


df_filtered = dataForPlot[['last_response_time','mean_responseTime24h']]


# In[82]:


dataForPlot


# In[85]:


clf2=IsolationForest(n_estimators=100, max_samples=10, contamination=float(.05),                         max_features=2, bootstrap=True, n_jobs=-1, random_state=42, verbose=0)
clf2.fit(df_filtered)
pred = clf2.predict(df_filtered)
df_filtered['anomaly_isolation']=pred
outliers=df_filtered.loc[df_filtered['anomaly_isolation']==-1]
outlier_index=list(outliers.index)
#Find the number of anomalies and normal points. Points classified -1 are anomalous
print(df_filtered['anomaly_isolation'].value_counts())


# In[90]:


df_filtered1 = df_filtered.loc['2021-10-26':'2021-11-15',:]


# In[160]:


df_filtered1


# In[91]:


sns.scatterplot(data=df_filtered1, x= df_filtered1.index, y="last_response_time", hue="anomaly_isolation", palette="deep")


# In[177]:


(~s).cumsum()[s].value_counts().max()


# In[173]:


zoomDF = df_filtered1.loc['2021-10-30':'2021-11-01',:]


# In[239]:


df_filtered1.to_pickle("toCheckOutlier.pkl")


# In[174]:


sns.scatterplot(data = zoomDF, x= zoomDF.index, y="last_response_time", hue="anomaly_isolation", palette="deep")


# In[159]:


sns.boxplot(df_filtered1.mean_responseTime24h)


# In[229]:


""" BASELINE PERFORMANCE:
AS WE CAN SEE FROM THE PLOT, THE MEAN_RESPONSE ABOVE 830 AND BELLOW 780 ARE CONSIDERED OUTLIERS
"""

"""FUNCTION TO APPLY TO EACH ROW AND CALCULATE IF OUTLIER OR NOT"""
def checkOutlier(meanResponseTime):
    outlier = False
    if ((meanResponseTime >= 830.0) and (meanResponseTime < 780.0)):
        outlier = True
    return outlier


# In[236]:


df_filtered1["baseline_outlier_Mean"] = df_filtered1["mean_responseTime24h"].apply(checkOutlier)


# In[233]:


542.000000 >= 830.0


# In[231]:


df_filtered1["mean_responseTime24h"]


# In[230]:


df_filtered1["baseline_outlier_Mean"]


# In[235]:


df_filtered1["baseline_outlier_Mean"].value_counts()


# In[176]:


zoomDF['mean_responseTime24h'].plot(figsize = (12,6))


# In[175]:


zoomDF['last_response_time'].plot(figsize = (12,6))


# In[163]:


df_filtered1['last_response_time'].plot(figsize = (12,6))


# In[92]:


normalValues = df_filtered1.loc[df_filtered1['anomaly_isolation'] == 1]


# In[131]:


normalValues["timestamp"] = normalValues.index


# In[132]:


normalValues


# In[96]:


Outliers = df_filtered1.loc[df_filtered1['anomaly_isolation'] == -1]


# In[133]:


Outliers["timestamp"] = Outliers.index


# In[134]:


Outliers


# In[149]:


# plot the line, the samples, and the nearest vectors to the plane

xx, yy = np.meshgrid(np.linspace(0, 20000, 50), np.linspace(0, 20000, 50)) 
Z = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = clf.decision_function(train)
Z = Z.reshape(xx.shape)


# In[150]:


plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(normalValues.iloc[:,0], normalValues.iloc[:, 1], c="green", s=20, edgecolor="k")
b2 = plt.scatter(Outliers.iloc[:,0], Outliers.iloc[:, 1], c="red", s=20, edgecolor="k")
plt.axis("tight")
plt.xlim((200, 2000))
plt.ylim((500, 1000))
plt.legend(
    #[b1, b2, c],
    [b1, b2],
    #["training observations", "new regular observations", "new abnormal observations"],
    ["normal observations", "anomalies"],
    loc="upper left",
)
plt.show()


# In[157]:


# plot the line, the samples, and the nearest vectors to the plane

xx, yy = np.meshgrid(np.linspace(0, 2000, 50), np.linspace(0, 1500, 50)) 
Z = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(normalValues.iloc[:,0], normalValues.iloc[:, 1], c="green", s=20, edgecolor="k")
b2 = plt.scatter(Outliers.iloc[:,0], Outliers.iloc[:, 1], c="red", s=20, edgecolor="k")
plt.axis("tight")
plt.xlim((200, 2000))
plt.ylim((500, 1000))
plt.legend(
    [b1, b2],
    ["normal observations", "anomalies"],
    loc="upper left",
)
plt.show()


# In[196]:


df = pd.DataFrame.from_dict(pd.read_pickle("device1Cleaned.pkl"))
df


# In[200]:


""" subsample to have a different timeline than its twim-sample (validation)"""

df = pd.DataFrame.from_dict(pd.read_pickle("device1Cleaned.pkl"))

metrics_df = df[['event_timestamp_updated_DT','event_severity_1to4','event_status_Categ','event_timestamp_updated','last_response_time']]

dataForPlot = df[['event_severity_1to4','event_status_Categ','event_creation_time_DT','event_timestamp_updated_DT','last_response_time']]


# In[203]:


metrics_df.index = metrics_df.event_timestamp_updated_DT
dataForPlot.index = dataForPlot.event_timestamp_updated_DT
metrics_df = metrics_df.sort_index()
dataForPlot = dataForPlot.sort_index()

metrics_df = metrics_df.loc['2021-10-26':'2021-11-15',:]
dataForPlot = dataForPlot.loc['2021-10-26':'2021-11-15',:]

metrics_df.head()


# In[206]:


dataUpDown = dataForPlot.drop(dataForPlot[dataForPlot['event_status_Categ'] == 1].index, inplace = False)


# In[207]:


dataUpDown = dataUpDown.drop(dataUpDown[dataUpDown['event_status_Categ'] == 2].index, inplace = False)


# In[208]:


dataUpDown['event_status_Categ'].value_counts()


# In[215]:


s= dataUpDown['event_status_Categ']


# In[218]:


(~s).cumsum()[s].value_counts().max()


# In[209]:


a = dataUpDown['event_status_Categ'].values
b = pd.factorize((~a).cumsum())[0]


# In[210]:


np.bincount(b[a]).max()


# In[211]:


np.bincount(b[a]).mean()

