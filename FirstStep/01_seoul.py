#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
f = open('seoul.csv', 'r', encoding='cp949')
data = csv.reader(f,delimiter=',')
print(data)
f.close()


# In[2]:


import csv
f = open('seoul.csv')
data = csv.reader(f)

for row in data:
    print(row)
    
f.close()


# In[4]:


import csv
f = open('seoul.csv')
data = csv.reader(f)

header = next(data)
#print(header)

for row in data:
    print(row)
    
f.close()


# In[5]:


import csv
f = open('seoul.csv')
data = csv.reader(f)
header = next(data)
for row in data:
    row[-1] = float(row[-1]) # 최고 기온을 실수로 변환
    print(row)
    
f.close()


# In[6]:


import csv
f = open('seoul.csv')
data = csv.reader(f)
header = next(data)
for row in data:
    if row[-1] == '':
        row [-1] = -999 # 빈 문자열이 있던 자리 표시
    row [-1] = float(row[-1])
    print(row)
    
f.close()


# In[7]:


import csv

max_temp = -999 # 최고 기온을 저장할 변수 초기화
max_date = ''   # 최고 기온이었던 날짜를 저장할 변수 초기화

f = open('seoul.csv')
data = csv.reader(f)
header = next(data)

for row in data:
    if row[-1] == '':
        row[-1] = -999
    row[-1] = float(row[-1])
    
    if max_temp < row[-1]:
        max_temp = row[-1]
        max_date = row[0]
print(max_date, ':', max_temp)

f.close()

