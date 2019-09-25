#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []               # 최고 기온 데이터를 저장할 리스트 생성

for row in data:
    if row[-1] != '':    # 최고 기온 데이터 값이 존재하면
        result.append(float(row[-1])) # result 리스트에 최고 기온 값 추가

print(result)
print(len(result))
f.close()


# In[5]:


import csv
import matplotlib.pyplot as plt

f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []               # 최고 기온 데이터를 저장할 리스트 생성

for row in data:
    if row[-1] != '':    # 최고 기온 데이터 값이 존재하면
        result.append(float(row[-1])) # result 리스트에 최고 기온 값 추가

plt.figure(figsize=(13, 2)) # 그래프 가로 세로 크기(13, 2 인치)
                            # 아래 코드보다 먼저 와야 함
plt.plot(result,'r')        # result 리스트의 데이터를 red  그래프로 그리기
plt.show()

f.close()


# In[10]:


import csv
import matplotlib.pyplot as plt

f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = []
low = []

for row in data:
    if row[-1] != '' and row[-2] != '':
        if 1983 <= int(row[0].split('-')[0]):
            if row[0].split('-')[1] == '02' and row[0].split('-')[2] == '16':
                high.append(float(row[-1]))
                low.append(float(row[-2]))
plt.figure(figsize = (10, 2))
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.title('맑은 고딕')
plt.plot(high, 'hotpink', label = 'high')
plt.plot(low, 'skyblue', label = 'low')
plt.legend()
plt.show()
f.close()


# In[12]:


import csv
import matplotlib.pyplot as plt

f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data:
    if row[-1] != '' and row[0].split('-')[1] == '08':
        result.append(float(row[-1]))

plt.hist(result, bins=100, color='r')
plt.show()
f.close()


# In[15]:


import csv
import matplotlib.pyplot as plt

f = open('seoul.csv')
data = csv.reader(f)
next(data)
jan = []
aug = []

for row in data:
    if row[-1] != '' :
        if row[0].split('-')[1] == '01':
            jan.append(float(row[-1]))
        if row[0].split('-')[1] == '08':
            aug.append(float(row[-1]))
            
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.hist(jan, bins=100, color='b', label = 'Jan')
plt.hist(aug, bins=100, color='r', label = 'Aug')
plt.legend(loc=2)
plt.title('1907~2019: 1월, 8월 최고온도')
plt.show()

f.close()


# In[16]:


plt.boxplot(jan)
plt.boxplot(aug)
plt.show()


# In[ ]:




