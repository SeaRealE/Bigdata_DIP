#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import pandas as pd
import numpy as np

import collections

import folium
import folium.plugins as plugins
import os


style.use('seaborn-talk')
krfont = {'family':'NanumGothic','weight':'bold','size':10}
matplotlib.rc('font',**krfont)
matplotlib.rcParams['axes.unicode_minus']=False

# 클러스터링 결과 표시
def plotResult(X,y,data,title="클러스터링 결과"):
    for i in range(0,len(collections.Counter(y))):
        plt.scatter(X[y==i,0],X[y==i,1], marker='o', s=40, label="클러스터"+str(i))
    
    plt.scatter(data[:,0],data[:,1], c='black',marker='s', s=40)
    plt.title(title)
    #plt.legend()
    #plt.legend(frameon=False)
    plt.legend().remove()
    plt.show()

######################################################################

# 주요 변수 위치 데이터
bus = pd.read_csv('bus_stop.csv')
fire = pd.read_csv('fire.csv')
police = pd.read_csv('police.csv')
co = pd.read_csv('co.csv')
cctv = pd.read_csv('cctv.csv')
private = pd.read_csv("private_parking.csv")
public = pd.read_csv("public_parking.csv")

cctv.columns=["name","Latitude","Longitude"]
private.columns=["name","Latitude","Longitude"]
public.columns=["name","Latitude","Longitude"]

data_var = pd.concat([cctv])
# data_var = pd.concat([bus,fire,police, co])
# data_var = pd.concat([private, public])
# data_var = pd.concat([private])
# data_var = pd.concat([public])
# data_var

# 지도 마커 그룹 생성
# 버스정류장, 소방서, 경찰서, 관공서, CCTV, 민공영주차장
add_cctv = folium.FeatureGroup(name='cctv')
for i in range(len(cctv)) :
    folium.Marker([cctv.iat[i,1], cctv.iat[i,2]], color = 'blue', popup='cctv').add_to(add_cctv)

add_bus = folium.FeatureGroup(name='bus_stop')
for i in range(len(bus)) :
    folium.Marker([bus.iat[i,1], bus.iat[i,2]], color = 'green', popup='bus').add_to(add_bus)

add_co = folium.FeatureGroup(name='community')
for i in range(len(co)) :
    folium.Marker([co.iat[i,1], co.iat[i,2]], color = 'yellow', popup='co').add_to(add_co)

add_police = folium.FeatureGroup(name='police')
for i in range(len(police)) :
    folium.Marker([police.iat[i,1], police.iat[i,2]], color = 'yellow', popup='police').add_to(add_police)

add_fire = folium.FeatureGroup(name='fire')
for i in range(len(fire)) :
    folium.Marker([fire.iat[i,1], fire.iat[i,2]], color = 'yellow', popup='fire').add_to(add_fire)

add_private = folium.FeatureGroup(name='private')
for i in range(len(private)) :
    folium.Marker([private.iat[i,1], private.iat[i,2]], color = 'black', popup='private').add_to(add_private)

add_public =folium.FeatureGroup(name='public')
for i in range(len(public)) :
    folium.Marker([public.iat[i,1], public.iat[i,2]], color = 'black', popup='public').add_to(add_public)


# 이상치 제거
lat_mean = data_var['Latitude'].mean()
lat_std = data_var['Latitude'].std()

indexNames = data_var[data_var['Latitude'] < lat_mean - 3*lat_std].index
data_var.drop(indexNames , inplace=True)
indexNames = data_var[data_var['Latitude'] > lat_mean + 3*lat_std].index
data_var.drop(indexNames , inplace=True)


lon_mean = data_var['Longitude'].mean()
lon_std = data_var['Longitude'].std()

indexNames = data_var[data_var['Longitude'] < lon_mean - 3*lon_std].index
data_var.drop(indexNames , inplace=True)
indexNames = data_var[data_var['Longitude'] > lon_mean + 3*lon_std].index
data_var.drop(indexNames , inplace=True)

# numpy 자료형 변환
listx= list(data_var['Latitude'])
listy = list(data_var['Longitude'])

data_list = np.empty((len(listx),2),float)

for i in range(0,len(listx)):
    data_list[i,0] = listx[i]
    data_list[i,1] = listy[i]
    
data_list.shape

######################################################################

# 불법주정차 데이터
# 날짜 / 위치 인덱스 / 클러스터 인덱스 / 위도 / 경도 
data = pd.read_csv("DATA_setting")

# 인덱스 / 위치 인덱스 / 위도 / 경도 
ADD = pd.read_csv('add.csv')

# 날짜 자료형 변환 및 인덱스 설정
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data[['add', 'lat', 'lon']]

# 위치인덱스 사이즈 분류
data_freq = data.groupby('add').size()
# 오름차순
data_freq = data_freq.sort_values()
# 인덱스 설정
data_freq = pd.DataFrame(data_freq).reset_index()
# ADD(위경도 데이터) merge
data_freq = pd.merge(data_freq, ADD, on = 'add', how = 'left')
data_freq.columns = ['add','freq','lat','lon']


# 이상치 제거
lat_mean = data_freq['lat'].mean()
lat_std = data_freq['lat'].std()

indexNames = data_freq[data_freq['lat'] < lat_mean - 3*lat_std].index
data_freq.drop(indexNames , inplace=True)

indexNames = data_freq[data_freq['lat'] > lat_mean + 3*lat_std].index
data_freq.drop(indexNames , inplace=True)


lon_mean = data_freq['lon'].mean()
lon_std = data_freq['lon'].std()
indexNames = data_freq[data_freq['lon'] < lon_mean - 3*lon_std].index
data_freq.drop(indexNames , inplace=True)
indexNames = data_freq[data_freq['lon'] > lon_mean + 3*lon_std].index
data_freq.drop(indexNames , inplace=True)

# numpy 자료형으로 변환
listx = list(data_freq['lat'])
listy = list(data_freq['lon'])

result_list = np.empty((len(listx),2),float)

for i in range(0,len(listx)):
    result_list[i,0] = listx[i]
    result_list[i,1] = listy[i]
    
result_list.shape

######################################################################

# DBSCAN
# eps = 원의 반지름
# min_sample 원에 포함되는 데이터 개수의 최소값
db = DBSCAN(eps=0.00010, min_samples=2, metric='haversine')
y_db = db.fit_predict(result_list)

plotResult(result_list, y_db, data_list, title="DBSCAN 클러스터링 결과")

######################################################################

# 좌표값과 클러스터 인덱스값 merge
count_data = collections.Counter(y_db)
newData = pd.DataFrame(result_list.reshape(len(result_list),2), columns=['lat','lon'])
dbscan_data = pd.DataFrame(y_db.reshape(len(y_db),1), columns=['set'])

newData = pd.merge(newData,dbscan_data, how="outer",left_index=True, right_index=True)
#newData


# 각 클러스터마다 평균 좌표값 계산
final_data = np.empty((len(collections.Counter(y_db)),2),float)

len_set = len(collections.Counter(y_db))
len_datasize = len(result_list)

lat = 0
lon = 0

for i in range(0,len_set):
    for j in range(0, len_datasize):
        if newData['set'][j] == i:
            lat += newData['lat'][j]
            lon += newData['lon'][j]
    if lat == 0 or lon == 0 :
        continue
    lat /= count_data[i]
    lon /= count_data[i]

    final_data[i,0] = lat
    final_data[i,1] = lon

    lat = 0
    lon = 0

# 클러스터 인덱스값 -1 로 인한 이상치  
final_data2 = np.delete(final_data,1359, axis=0)

# 주요 변수 위치 비교
plt.scatter(final_data2[:,0],final_data2[:,1], c='black',marker='s', s=40)
plt.scatter(data_list[:,0],data_list[:,1], c='red',marker='s', s=40)
plt.show()


# DBSCAN로 나온 클러스터마다 위치 평균값 + 해당 클러스터 내 데이터 수 
output = pd.DataFrame(final_data2.reshape(len(final_data2),2), columns=['lat','lon'])
output.to_csv("output.csv")
#output

dbscan_count = pd.DataFrame(collections.Counter(y_db).items())
dbscan_count = dbscan_count.set_index(0)
# 클러스터 인덱스값 -1 삭제 
dbscan_count = dbscan_count.drop([-1])
dbscan_count.columns = ['freq']

final_output = pd.merge(output,dbscan_count, how="outer",left_index=True, right_index=True)
final_output


### 히트맵
data_1 = final_output
data_1 = data_1[['freq', 'lat', 'lon']]

arr = np.empty((0, 2), int)

for i in range(len(data_1)) :    
    loc = [data_1.iat[i,1],data_1.iat[i,2]]
    freq = int(data_1.iat[i,0])
    
    if freq > 0 :
        arr = np.append(arr, np.array([loc]*freq), axis = 0)               

visual = arr
Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

high_95 = folium.FeatureGroup(name='high_0.95')
plugins.HeatMap(visual).add_to(high_95)
Map.add_child(high_95)
Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)

folium.LayerControl(collapsed=False).add_to(Map)

Map.save(os.path.join('result', 'DBSCAN.html'))







