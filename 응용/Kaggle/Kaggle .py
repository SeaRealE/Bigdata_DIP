#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[2]:


train = pd.read_csv('./train.csv', parse_dates=['datetime'],
                    index_col=0,)
test = pd.read_csv('./test.csv', parse_dates=['datetime'],
                   index_col=0,)


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train['count'].plot()


# In[8]:


train['temp'].plot()


# In[9]:


train['season'].plot(kind='hist')


# In[10]:


sns.boxplot(x='season', y='count', data=train)


# In[11]:


train['month'] = train.index.month
train['day'] = train.index.day
train['wday'] = train.index.week
train['year'] = train.index.year.astype(str)
train['hour'] = train.index.hour

test['month'] = test.index.month
test['day'] = test.index.day
test['wday'] = test.index.week
test['year'] = test.index.year.astype(str)
test['hour'] = test.index.hour


# In[12]:


train.head()


# In[13]:


sns.boxplot(x='wday',y='count', data=train)


# In[14]:


sns.boxplot(x='hour',y='count', data=train)


# In[15]:


plt.hist(train['count'][train['year'] == '2011'], alpha=0.5, label='2011')
plt.hist(train['count'][train['year'] == '2012'], alpha=0.5, label='2012')


# In[16]:


sns.boxplot(x='season',y='windspeed',data=train, palette='winter')


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


train.columns


# In[19]:


X = train[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered',  'month',
       'day', 'wday', 'year', 'hour']]
Y = train['count']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 5000,random_state= 42)
clf.fit(X_train, y_train)


# In[ ]:


clf.score(X_train, y_train), clf.score(X_test, y_test)


# In[ ]:


pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn import metrics
print('MSE:', metrics.mean_squared_error(y_test, pred))


# In[ ]:


import numpy as np
plt.scatter(y_test, pred)
plt.plot(np.arange(1000),color='r')


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
clf2 = DecisionTreeRegressor()
clf2.fit(X_train, y_train)


# In[ ]:


pred = clf2.predict(X_test)


# In[ ]:


print('MSE:', metrics.mean_squared_error(y_test, pred))


# In[ ]:


plt.scatter(y_test,pred)


# In[ ]:




