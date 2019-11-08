#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


#pd.read_excel()
s1 = pd.read_csv('./대구도시철도공사_일별시간별승하차인원_20171231.csv', 
            encoding="cp949")


# In[4]:


s2 = pd.read_csv('./대구도시철도공사_일별시간별승하차인원_20181231.csv', 
            encoding="cp949")


# In[5]:


s3 = pd.read_csv('./대구도시철도공사_일별시간별승하차인원_20190831.csv', 
            encoding="cp949", skiprows=2)


# In[6]:


s1.head(1)
# col을 잘 읽었는지 확인 하기 위해, col 오류 확인하기 위해


# In[7]:


s2.head(1)


# In[8]:


s3.head(1)


# In[9]:


s2.columns


# In[10]:


colname = ['월', '일', '역번호', '역명', '승하차', '05~06', '06~07', '07~08', '08~09',
       '09~10', '10~11', '11~12', '12~13', '13~14', '14~15', '15~16', '16~17',
       '17~18', '18~19', '19~20', '20~21', '21~22', '22~23', '23~24', '일계']


# In[11]:


s1.columns = colname
s3.columns = colname

# 승하 를 승하차로 바꾸기 


# In[12]:


s1.head(1)


# In[13]:


s1['년도'] = 2017 # [~~~~]
s2['년도'] = 2018
s3['년도'] = 2019


# In[14]:


s1['역명'].unique(), len(s1['역명'].unique())

# 환승역에는 역이 중첩된다. ex) 반월당1, 반월당2   신남2, 신남3


# In[15]:


s1.shape, s2.shape, s3.shape


# In[16]:


subway = pd.concat( [s1,s2,s3] )

# 연도별 데이터 다 합치기, index 초기화가 안되어 있다.


# In[17]:


subway.shape


# In[18]:


subway.to_csv('./대구지하철 201701~201908.csv', index=False, encoding="cp949")


# In[19]:


subway.head(1)


# In[20]:


subway.tail(1)

# index 초기화가 안된 상태라서


# In[21]:


subway = subway.reset_index()

# index 초기화 하기


# In[22]:


subway.tail(1)


# In[23]:


subway.info()

# 데이터 타입을 확인을 하자


# In[24]:


subway['역번호'].dtype


# In[25]:


subway['역번호'] = subway['역번호'].astype('str')


# In[26]:


subway['역번호'].dtype


# In[27]:


subway.isnull().sum()

# true false


# In[28]:


subway.isnull().sum().sum()

# 0 -> 결측치가 없다는 뜻
# 보간법


# In[29]:


subway['역번호'].unique() , len(subway['역번호'].unique())


# In[30]:


subway.columns


# In[31]:


t = pd.DataFrame(subway.groupby(by=['역번호', '역명',
                                    '년도', '승하차'])['일'].count())
t

# 1년치 측정값이 다 있는 2017년 2018년에서 365가 아닌 값이 나오면 결측치가 있는 것
# boolean indexing -> true 결과가 나오는 값만 가져온다.


# In[32]:


t2 = t[t['일'] < 365]

# 365가 아닌 경우 값 넣기


# In[33]:


t2 = t2.reset_index()
t2['년도'].unique()

# 2017년 2018년은 값이 다 제대로 존재한다.


# In[34]:


t = t.reset_index()
t[ (t['일'] < 365) & (t['년도'] != 2019) ] # &(and) | (or)

# 2017년 2018년은 값이 다 제대로 존재한다.


# In[35]:


t['역번호']


# In[36]:


month = pd.DataFrame(subway.groupby(by=['년도', '월', '역명'])['일계'].mean())

# 이제부터 본격적으로 데이터 분석 
# 년별, 월별, 역별


# In[37]:


month = month.reset_index()
month


# In[38]:


top10 = month.sort_values(by='일계', ascending=False).head(10)   # 내림차순
bottom10 = month.sort_values(by='일계').head(10)                 # 오름차순


# In[39]:


top10


# In[40]:


bottom10


# In[41]:


year = pd.DataFrame(subway.groupby(by=['년도','역명'])['일계'].mean())


# In[42]:


year_top10 = year.sort_values(by='일계', ascending=False).head(10)   # 내림차순
year_bottom10 = year.sort_values(by='일계').head(10)                 # 오름차순


# In[43]:


year_top10


# In[44]:


year_bottom10


# In[45]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


year_top10 = year_top10.reset_index()
year_top10 = year_top10[['역명', '일계', '년도']]
year_top10['역명_년'] =  year_top10['년도'].astype('str') + year_top10['역명']
year_top10 = year_top10.set_index('역명_년')
year_top10


# In[47]:


year_top10['일계'].plot(kind='bar')

# 한글 폰트가 깨져서 에러


# In[48]:


station = pd.read_csv('./06. 대구도시철도공사_도시철도역사정보_20190923.csv', encoding='cp949')


# In[49]:


station.head()


# In[50]:


station['역번호'].unique(), subway['역번호'].unique()


# In[51]:


station['역번호'] = station['역번호'] * 10
station['역번호'] = station['역번호'].astype('str')


# In[52]:


station['역번호'].unique(), subway['역번호'].unique()


# In[53]:


station = station[['역번호', '역위도', '역경도']]


# In[54]:


subway_merge = pd.merge(subway, station, on='역번호', how='inner')

# inner 조인을 사용해라, outer 값?


# In[55]:


subway_merge


# In[56]:


year = pd.DataFrame(subway_merge.groupby(by=['년도','역번호','역명'])['일계'].mean())


# In[57]:


year = year.reset_index()


# In[58]:


year_2017 = year[ year['년도'] == 2017]


# In[59]:


year_2017 = pd.merge(year_2017, station, on='역번호', how='inner') 


# In[60]:


year_2017


# In[61]:


lon = year_2017['역위도'].mean()
lat = year_2017['역경도'].mean()


# In[62]:


pip install folium


# In[63]:


import folium


# In[64]:


daegu = folium.Map(location= [lon, lat], zoom_start=12)
daegu


# In[65]:


for index in year_2017.index:
    lat = year_2017.loc[index, '역위도']
    lon = year_2017.loc[index, '역경도']
    
    folium.CircleMarker( [lat,lon],
                       radius = year_2017.loc[index, '일계']/ 1000,
                       popup = year_2017.loc[index, '역명'],
                       color='blue', fill= True).add_to(daegu)


# In[66]:


daegu


# In[ ]:




