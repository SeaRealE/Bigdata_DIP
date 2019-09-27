#!/usr/bin/env python
# coding: utf-8

# # Getting Started with pandas

# In[ ]:


import pandas as pd


# In[ ]:


from pandas import Series, DataFrame


# In[ ]:


import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)


# ## Introduction to pandas Data Structures

# ## Introduction to pandas Data Structures

# ### Series

# In[ ]:


obj = pd.Series([4, 7, -5, 3])
obj


# In[ ]:


obj.values
obj.index  # like range(4)


# In[ ]:


obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index


# In[ ]:


obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]


# In[ ]:


obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)


# In[ ]:


'b' in obj2
'e' in obj2


# In[ ]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3


# In[ ]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4


# In[ ]:


pd.isnull(obj4)
pd.notnull(obj4)


# In[ ]:


obj4.isnull()


# In[ ]:


obj3
obj4
obj3 + obj4


# In[ ]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# In[ ]:


obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj


# ### DataFrame

# In[ ]:


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)


# In[ ]:


frame


# In[ ]:


frame.head()


# In[ ]:


pd.DataFrame(data, columns=['year', 'state', 'pop'])


# In[ ]:


frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
frame2
frame2.columns


# In[ ]:


frame2['state']
frame2.year


# In[ ]:


frame2.loc['three']


# In[ ]:


frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(6.)
frame2


# In[ ]:


val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2


# In[ ]:


frame2['eastern'] = frame2.state == 'Ohio'
frame2


# In[ ]:


del frame2['eastern']
frame2.columns


# In[ ]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}


# In[ ]:


frame3 = pd.DataFrame(pop)
frame3


# In[ ]:


frame3.T


# In[ ]:


pd.DataFrame(pop, index=[2001, 2002, 2003])


# In[ ]:


pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)


# In[ ]:


frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3


# In[ ]:


frame3.values


# In[ ]:


frame2.values


# ### Index Objects

# In[ ]:


obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]


# index[1] = 'd'  # TypeError

# In[ ]:


labels = pd.Index(np.arange(3))
labels
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
obj2.index is labels


# In[ ]:


frame3
frame3.columns
'Ohio' in frame3.columns
2003 in frame3.index


# In[ ]:


dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels


# ## Essential Functionality

# ### Reindexing

# In[ ]:


obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj


# In[ ]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# In[ ]:


obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
obj3.reindex(range(6), method='ffill')


# In[ ]:


frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2


# In[ ]:


states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)


# In[ ]:


frame.loc[['a', 'b', 'c', 'd'], states]


# ### Dropping Entries from an Axis

# In[ ]:


obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])


# In[ ]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data


# In[ ]:


data.drop(['Colorado', 'Ohio'])


# In[ ]:


data.drop('two', axis=1)
data.drop(['two', 'four'], axis='columns')


# In[ ]:


obj.drop('c', inplace=True)
obj


# ### Indexing, Selection, and Filtering

# In[ ]:


obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]


# In[ ]:


obj['b':'c']


# In[ ]:


obj['b':'c'] = 5
obj


# In[ ]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]


# In[ ]:


data[:2]
data[data['three'] > 5]


# In[ ]:


data < 5
data[data < 5] = 0
data


# #### Selection with loc and iloc

# In[ ]:


data.loc['Colorado', ['two', 'three']]


# In[ ]:


data.iloc[2, [3, 0, 1]]
data.iloc[2]
data.iloc[[1, 2], [3, 0, 1]]


# In[ ]:


data.loc[:'Utah', 'two']
data.iloc[:, :3][data.three > 5]


# ### Integer Indexes

# ser = pd.Series(np.arange(3.))
# ser
# ser[-1]

# In[ ]:


ser = pd.Series(np.arange(3.))


# In[ ]:


ser


# In[ ]:


ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]


# In[ ]:


ser[:1]
ser.loc[:1]
ser.iloc[:1]


# ### Arithmetic and Data Alignment

# In[ ]:


s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=['a', 'c', 'e', 'f', 'g'])
s1
s2


# In[ ]:


s1 + s2


# In[ ]:


df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2


# In[ ]:


df1 + df2


# In[ ]:


df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})
df1
df2
df1 - df2


# #### Arithmetic methods with fill values

# In[ ]:


df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                   columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                   columns=list('abcde'))
df2.loc[1, 'b'] = np.nan
df1
df2


# In[ ]:


df1 + df2


# In[ ]:


df1.add(df2, fill_value=0)


# In[ ]:


1 / df1
df1.rdiv(1)


# In[ ]:


df1.reindex(columns=df2.columns, fill_value=0)


# #### Operations between DataFrame and Series

# In[ ]:


arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
arr - arr[0]


# In[ ]:


frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame
series


# In[ ]:


frame - series


# In[ ]:


series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame + series2


# In[ ]:


series3 = frame['d']
frame
series3
frame.sub(series3, axis='index')


# ### Function Application and Mapping

# In[ ]:


frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
np.abs(frame)


# In[ ]:


f = lambda x: x.max() - x.min()
frame.apply(f)


# In[ ]:


frame.apply(f, axis='columns')


# In[ ]:


def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)


# In[ ]:


format = lambda x: '%.2f' % x
frame.applymap(format)


# In[ ]:


frame['e'].map(format)


# ### Sorting and Ranking

# In[ ]:


obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()


# In[ ]:


frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame.sort_index()
frame.sort_index(axis=1)


# In[ ]:


frame.sort_index(axis=1, ascending=False)


# In[ ]:


obj = pd.Series([4, 7, -3, 2])
obj.sort_values()


# In[ ]:


obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()


# In[ ]:


frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_values(by='b')


# In[ ]:


frame.sort_values(by=['a', 'b'])


# In[ ]:


obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()


# In[ ]:


obj.rank(method='first')


# In[ ]:


# Assign tie values the maximum rank in the group
obj.rank(ascending=False, method='max')


# In[ ]:


frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
                      'c': [-2, 5, 8, -2.5]})
frame
frame.rank(axis='columns')


# ### Axis Indexes with Duplicate Labels

# In[ ]:


obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj


# In[ ]:


obj.index.is_unique


# In[ ]:


obj['a']
obj['c']


# In[ ]:


df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df
df.loc['b']


# ## Summarizing and Computing Descriptive Statistics

# In[ ]:


df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df


# In[ ]:


df.sum()


# In[ ]:


df.sum(axis='columns')


# In[ ]:


df.mean(axis='columns', skipna=False)


# In[ ]:


df.idxmax()


# In[ ]:


df.cumsum()


# In[ ]:


df.describe()


# In[ ]:


obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()


# ### Correlation and Covariance

# conda install pandas-datareader

# In[ ]:


price = pd.read_pickle('examples/yahoo_price.pkl')
volume = pd.read_pickle('examples/yahoo_volume.pkl')


# import pandas_datareader.data as web
# all_data = {ticker: web.get_data_yahoo(ticker)
#             for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
# 
# price = pd.DataFrame({ticker: data['Adj Close']
#                      for ticker, data in all_data.items()})
# volume = pd.DataFrame({ticker: data['Volume']
#                       for ticker, data in all_data.items()})

# In[ ]:


returns = price.pct_change()
returns.tail()


# In[ ]:


returns['MSFT'].corr(returns['IBM'])
returns['MSFT'].cov(returns['IBM'])


# In[ ]:


returns.MSFT.corr(returns.IBM)


# In[ ]:


returns.corr()
returns.cov()


# In[ ]:


returns.corrwith(returns.IBM)


# In[ ]:


returns.corrwith(volume)


# ### Unique Values, Value Counts, and Membership

# In[ ]:


obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])


# In[ ]:


uniques = obj.unique()
uniques


# In[ ]:


obj.value_counts()


# In[ ]:


pd.value_counts(obj.values, sort=False)


# In[ ]:


obj
mask = obj.isin(['b', 'c'])
mask
obj[mask]


# In[ ]:


to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = pd.Series(['c', 'b', 'a'])
pd.Index(unique_vals).get_indexer(to_match)


# In[ ]:


data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],
                     'Qu2': [2, 3, 1, 2, 3],
                     'Qu3': [1, 5, 2, 4, 4]})
data


# In[ ]:


result = data.apply(pd.value_counts).fillna(0)
result


# ## Conclusion

# In[ ]:


pd.options.display.max_rows = PREVIOUS_MAX_ROWS

