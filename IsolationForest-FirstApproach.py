#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


# In[9]:


rng = np.random.RandomState(42)


# In[10]:


"""IMPORT DATA"""
df = pd.DataFrame.from_dict(pd.read_pickle("orderedDatasetDevice1.pkl"))


# In[11]:


""" subsample to come from a different duration than its twim-sample (validation)"""
df1 = df.loc['2021-10-26':'2021-11-15',:]


# In[12]:


df1.info()


# In[13]:


"""VISUALIZE DATA"""
sns.boxplot(df1.response_time)
#Points outside of whiskers will be inferred as an outliers. 


# In[14]:


"""1ST EXPERIMENT"""


# In[15]:


""" BASELINE PERFORMANCE:
AS WE CAN SEE FROM THE PLOT, THE RESPONSE_TIME ABOVE 2000 ARE CONSIDERED OUTLIERS
"""

"""FUNCTION TO APPLY TO EACH ROW AND CALCULATE IF OUTLIER OR NOT"""
def checkOutlier(responseTime):
    outlier = False
    if responseTime > 2000:
        outlier = True
    return outlier

df1["baseline_outlier"] = df1["response_time"].apply(checkOutlier)    


# In[16]:


df1["baseline_outlier"].value_counts()


# In[17]:


"""TRAIN MODEL"""
clf1=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.03),                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf1.fit(df1)
pred = clf1.predict(df1)
df1['anomaly']=pred
#Find the number of anomalies and normal points. Points classified -1 are anomalous
print(df1['anomaly'].value_counts())


# In[18]:


"""VISUALIZE AND COMPARE RESULTS"""
f, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=df1, x= df1.index, y="response_time", hue="anomaly", ax=axs[0])
sns.scatterplot(data=df1, x= df1.index, y="response_time", hue="baseline_outlier", ax=axs[1])
f.tight_layout()


# In[38]:


a = df1.loc[df1['anomaly']==-1]
a['response_time'].min()


# In[40]:


b = df1.loc[df1['anomaly']==1]
b['response_time'].max()


# In[19]:


"""2ND EXPERIMENT"""


# In[20]:


df1


# In[50]:


"""SPLIT TRAIN AND TEST DATA"""

X = df1.drop(['anomaly','baseline_outlier'], axis=1)
y = df1['anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[48]:





# In[51]:


X_train.shape, y_train.shape


# In[47]:


X_test.shape, y_test.shape


# In[52]:


"""USE GRIDSEARCHCV TO TUNE DE MODEL BY FINDIG THE BEST COMBINATION FOR THE PARAMETERS"""

model = IsolationForest(random_state=47)

param_grid = {'n_estimators': [100,1000, 1500], 
              'max_samples': [10,20], 
              'contamination': ['auto', 0.003, 0.04,0.05], 
              'max_features': [0, 3], 
              'bootstrap': [True,False], 
              'n_jobs': [-1]}

grid_search = model_selection.GridSearchCV(model, 
                                           param_grid,
                                           scoring="neg_mean_squared_error", 
                                           refit=True,
                                           cv=10, 
                                           return_train_score=True)
grid_search.fit(X_train, y_train)

best_model = grid_search.fit(X_train, y_train)
print('Optimum parameters', best_model.best_params_)

"""Optimum parameters {'bootstrap': True, 'contamination': 0.04, 'max_features': 3, 'max_samples': 10, 'n_estimators': 100, 'n_jobs': -1}"""


# In[24]:


df2 = df1.drop(['anomaly','baseline_outlier'], axis=1)


# In[25]:


df2


# In[26]:


"""TRAIN MODEL WITH TUNED PARAMETERS"""
clf2=IsolationForest(n_estimators=100, max_samples=10, contamination=float(.04),                         max_features=3, bootstrap=True, n_jobs=-1, random_state=42, verbose=0)
clf2.fit(df2)
pred = clf2.predict(df2)
df2['anomaly_isolation']=pred
#Find the number of anomalies and normal points. Points classified -1 are anomalous
print(df2['anomaly_isolation'].value_counts())


# In[27]:


"""VISUALIZE RESULTS"""
f, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=df2, x= df2.index, y="response_time", hue="anomaly_isolation", ax=axs[0], palette="deep")
sns.scatterplot(data=df1, x= df1.index, y="response_time", hue="baseline_outlier", ax=axs[1],palette="deep")
f.tight_layout()


# In[41]:


"""VALIDATION: Twin-Sample Validation"""
"""IMPORT DATA"""
"""DATA COMING FROM DIFFERENT DEVICE"""
twin = pd.DataFrame.from_dict(pd.read_pickle("orderedDatasetDevice18.pkl"))


# In[42]:


twin.info()


# In[29]:


"""select data from a different timeline than its twin sample"""
twin1 = twin.loc['2021-11-15':,:]


# In[30]:


"""VALIDATE MODEL WITH NEW DATA"""

pred = clf2.predict(twin1)
twin1['anomaly_isolation']=pred
#Find the number of anomalies and normal points. Points classified -1 are anomalous
print(twin1['anomaly_isolation'].value_counts())


# In[ ]:



sns.scatterplot(data=twin1, x= twin1.index, y="last_response_time", hue="anomaly_isolation", palette="deep")


# In[ ]:


twin1


# In[ ]:


# plot the line, the samples, and the nearest vectors to the plane

xx, yy = np.meshgrid(np.linspace(0, 2000, 50), np.linspace(0, 1500, 50)) 
Z = clf6.decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = clf.decision_function(train)
Z = Z.reshape(xx.shape)

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

