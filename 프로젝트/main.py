#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from haversine import haversine
import folium.plugins as plugins
import os


# 날짜 / 위치 인덱스 / 클러스터 인덱스 / 위도 / 경도 
data = pd.read_csv("DATA")

# 인덱스 / 위치 인덱스 / 위도 / 경도 
ADD = pd.read_csv('add.csv')

# 날짜 자료형 변환 및 인덱스 설정
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
#data.head()
data = data[['add', 'lat', 'lon']]

# 위치인덱스 사이즈 분류
data_freq = data.groupby('add').size()

# 내림차순
data_freq = data_freq.sort_values(ascending = [False])
data_freq = pd.DataFrame(data_freq).reset_index()

# ADD(위경도 데이터) merge
data_freq = pd.merge(data_freq, ADD, on = 'add', how = 'left')
data_freq.columns = ['add','freq','lat','lon']
#data_freq.head()

# 버스정류장, 소방서, 경찰서, 관공서, CCTV, 민공영주차장 위치데이터
bus = pd.read_csv('bus_stop.csv')
fire = pd.read_csv('fire.csv')
police = pd.read_csv('police.csv')
co = pd.read_csv('co.csv')
cctv = pd.read_csv('cctv.csv')
private = pd.read_csv("private_parking.csv")
public = pd.read_csv("public_parking.csv")

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

#####################################################################

# 2000개 분류 
# 위치 인덱스 / 빈도수 / 위도 / 경도
data_1 = pd.read_csv("DATA_2000")
data_1 = data_1[['add', 'freq', 'lat', 'lon']]

data_change = pd.read_csv("CHANGE_2000")
data_change = data_change[['origin', 'after']]
data_change.head()

data_del = pd.read_csv('DEL_2000')
data_del = list(data_del['add'])
#data_del

#####################################################################

# 날짜 / 위치 인덱스 / 클러스터 인덱스 / 위도 / 경도 
data = pd.read_csv("DATA")

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
data_freq = pd.DataFrame(data_freq).reset_index()

# ADD(위경도 데이터) merge
data_freq = pd.merge(data_freq, ADD, on = 'add', how = 'left')
data_freq.columns = ['add','freq','lat','lon']
#data_freq

#####################################################################

data_1 = data_freq


diff = 0.5  # 거리 500m
data_change = pd.DataFrame(columns = ['origin', 'after'])
data_del=[]

# freq 1 삭제 or 다른 곳에 병합
while len(data_1[data_1['freq']==2]) > 0 :
    x_loc = [data_1.iat[0,2],data_1.iat[0,3]]
    y_loc = [data_1.iat[1,2],data_1.iat[1,3]]
    d = haversine(x_loc, y_loc)
    ind = 1
    
    for i in range(2, len(data_1)) :
        y_loc = [data_1.iat[i,2],data_1.iat[i,3]]
        temp = haversine(x_loc, y_loc)
        if temp < d :
            d = temp
            ind = i
            
    if d > diff :
        data_del.append(data_1.iat[0,0])
        data_1 = data_1.drop(data_1.index[0])
    else :
        temp = pd.DataFrame([[data_1.iat[0,0],data_1.iat[ind,0]]], columns = ['origin', 'after'])
        data_change = data_change.append(temp)
        s = data_1.iat[0,1] + data_1.iat[ind,1]
        data_1.iat[ind,2] = (data_1.iat[0,1]*x_loc[0] + data_1.iat[ind,1]*data_1.iat[ind,2])/s
        data_1.iat[ind,3] = (data_1.iat[0,1]*x_loc[1] + data_1.iat[ind,1]*data_1.iat[ind,3])/s
        data_1.iat[ind,1] = s
        data_1 = data_1.drop(data_1.index[0])
    data_1 = data_1.sort_values(by='freq')

#####################################################################   

#data_1
data_1.to_csv("DATA_2")
#data_change.to_csv("CHANGE_2")
#pd.Series(data_del).to_csv("DEL_2")

#####################################################################

D = pd.read_csv('data_2')
D = D[['add', 'freq', 'lat', 'lon']]

D['cctv'] = D['bus']= D['police']= D['fire']= D['co']= D['private'] = D['public'] = 0
D = D[['freq', 'lat', 'lon', 'cctv', 'bus', 'police', 'fire', 'co', 'private', 'public']]

# 불법주정차 적발 위치 주변 주요변수 개수 파악
# 버스정류장, 소방서, 경찰서, 관공서, CCTV, 민공영주차장
for i in range(len(cctv)) :
    x_loc = [cctv.iat[i,1], cctv.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.1 :
            D.iat[j,3] += 1

for i in range(len(bus)) :
    x_loc = [bus.iat[i,1], bus.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.01 :
            D.iat[j,4] += 1

for i in range(len(police)) :
    x_loc = [police.iat[i,1], police.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.05 :
            D.iat[j,5] += 1

for i in range(len(fire)) :
    x_loc = [fire.iat[i,1], fire.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.05 :
            D.iat[j,6] += 1
            
for i in range(len(co)) :
    x_loc = [co.iat[i,1], co.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.05 :
            D.iat[j,7] += 1
            
for i in range(len(private)) :
    x_loc = [private.iat[i,1], private.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.5 :
            D.iat[j,8] += 1

for i in range(len(public)) :
    x_loc = [public.iat[i,1], public.iat[i,2]]
    for j in range(len(D)) :
        y_loc = [D.iat[j,1],D.iat[j,2]]
        d = haversine(x_loc, y_loc)
        if d < 0.5 :
            D.iat[j,9] += 1

#####################################################################

### 히트맵
data_1 = data_freq
data_1 = data_1[['add', 'freq', 'lat', 'lon']]

arr = np.empty((0, 2), int)
m = data_1['freq'].quantile(0.95)

for i in range(len(data_1)) :    
    loc = [data_1.iat[i,2],data_1.iat[i,3]]
    freq = int(data_1.iat[i,1]/m)
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

Map.save(os.path.join('result', 'ALL_95.html'))

#####################################################################

data_change.groupby('after').size().sort_values(ascending=False)

visual = data_1
s = visual['freq'].sum()
quan_95 = visual['freq'].quantile(0.95)
quan_99 = visual['freq'].quantile(0.99)
Map = folium.Map(location = [data['lat'].mean(), data['lon'].mean() ] , zoom_start=13)
for i in range(len(visual)) :
    col = 'blue'
    rad = visual.iat[i,1] /s *2000

    folium.Circle([visual.iat[i,2], visual.iat[i,3]], radius = rad, color = col).add_to(Map)
#Map

#####################################################################

### 히트맵, 원래 데이터 중 상위 2500 - 500
data_1 = data_freq.sort_values(by='freq', ascending=False)
data_1 = data_1[['add', 'freq', 'lat', 'lon']]

Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

for n in [2500, 2000, 1500, 1000, 500] :
    arr = np.empty((0, 2), int)

    for i in range(n) :    
        loc = [data_1.iat[i,2],data_1.iat[i,3]]
        freq = data_1.iat[i,1]
        if freq > 0 :
            arr = np.append(arr, np.array([loc]*freq), axis = 0)               

    visual = arr
    
        
    globals()[f'high_{n}'] = folium.FeatureGroup(name=str(f'high_{n}')) 
    Map.add_child(globals()[f'high_{n}'])
    plugins.HeatMap(visual).add_to(globals()[f'high_{n}'])
    
Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)

folium.LayerControl(collapsed=False).add_to(Map)
Map.save(os.path.join('result', 'ALL_high.html'))

#####################################################################

### 히트맵, 빈도 1 삭제 데이터 중 상위 2500 - 500
D = pd.read_csv("DATA_1")
D = D[['add', 'freq', 'lat', 'lon']]
data_1 = D.sort_values(by='freq', ascending=False)
data_1 = data_1[['add', 'freq', 'lat', 'lon']]

Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

for n in [2500, 2000, 1500, 1000, 500] :
    arr = np.empty((0, 2), int)

    for i in range(n) :    
        loc = [data_1.iat[i,2],data_1.iat[i,3]]
        freq = data_1.iat[i,1]
        if freq > 0 :
            arr = np.append(arr, np.array([loc]*freq), axis = 0)               

    visual = arr
    
        
    globals()[f'high_{n}'] = folium.FeatureGroup(name=str(f'high_{n}')) 
    Map.add_child(globals()[f'high_{n}'])
    plugins.HeatMap(visual).add_to(globals()[f'high_{n}'])
    
Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)

folium.LayerControl(collapsed=False).add_to(Map)
Map.save(os.path.join('result', 'delfreq1_high.html'))

#####################################################################

### 히트맵, 500개 기준
data_1 = pd.read_csv("DATA_2000")
data_1 = data_1[['add', 'freq', 'lat', 'lon']]

arr = np.empty((0, 2), int)
m = data_1['freq'].min()

for i in range(len(data_1)) :    
    loc = [data_1.iat[i,2],data_1.iat[i,3]]
    freq = int(data_1.iat[i,1]/m)
    if freq > 0 :
        arr = np.append(arr, np.array([loc]*freq), axis = 0)               

visual = arr
Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)
plugins.HeatMap(visual).add_to(Map)
Map.save(os.path.join('result', 'ALL_2000.html'))

#####################################################################

### 연도별 히트맵
for y in range(2014, 2020) :
    globals()[f'data_{y}']= data[data.index.year==y]

    data_1 = globals()[f'data_{y}'].groupby('add').size()
    data_1 = pd.DataFrame(data_1).reset_index()
    data_1 = pd.merge(data_1, ADD, on = 'add', how = 'left')
    data_1.columns = ['add','freq','lat','lon']
    data_1.sort_values(by= 'freq', ascending = False)

    arr = np.empty((0, 2), int)
    m = data_1['freq'].quantile(0.90)

    for i in range(len(data_1)) :    
        loc = [data_1.iat[i,2],data_1.iat[i,3]]
        freq = int(data_1.iat[i,1]/m)
        if freq > 0 :
            arr = np.append(arr, np.array([loc]*freq), axis = 0)  

    globals()[f'arr_{y}'] = arr


Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)


All_2014 = folium.FeatureGroup(name='2014') 
Map.add_child(All_2014)
plugins.HeatMap(arr_2014).add_to(All_2014)

All_2015 = folium.FeatureGroup(name='2015') 
Map.add_child(All_2015)
plugins.HeatMap(arr_2015).add_to(All_2015)

All_2016 = folium.FeatureGroup(name='2016') 
Map.add_child(All_2016)
plugins.HeatMap(arr_2016).add_to(All_2016)

All_2017 = folium.FeatureGroup(name='2017') 
Map.add_child(All_2017)
plugins.HeatMap(arr_2017).add_to(All_2017)

All_2018 = folium.FeatureGroup(name='2018') 
Map.add_child(All_2018)
plugins.HeatMap(arr_2018).add_to(All_2018)

All_2019 = folium.FeatureGroup(name='2019') 
Map.add_child(All_2019)
plugins.HeatMap(arr_2019).add_to(All_2019)

folium.LayerControl(collapsed=False).add_to(Map)
Map.save(os.path.join('result', 'Year.html'))


#####################################################################

### 월별
for m in range(1,13) :
    globals()[f'data_{m}'] = data[data.index.month== m]

    temp = globals()[f'data_{m}'].groupby('add').size()
    temp = pd.DataFrame(temp).reset_index()
    temp = pd.merge(temp, ADD, on = 'add', how = 'left')
    temp.columns = ['add','freq','lat','lon']
    temp.sort_values(by= 'freq', ascending = False)

    arr = np.empty((0, 2), int)
    k = temp['freq'].quantile(0.90)

    for i in range(len(temp)) :    
        loc = [temp.iat[i,2],temp.iat[i,3]]
        freq = int(temp.iat[i,1]/k)
        if freq > 0 :
            arr = np.append(arr, np.array([loc]*freq), axis = 0)  

    globals()[f'arr_{m}'] = arr

Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)

for m in range(1,13) :
    globals()[f'all_{m}'] = folium.FeatureGroup(name=str(m)) 
    Map.add_child(globals()[f'all_{m}'])
    plugins.HeatMap(globals()[f'arr_{m}']).add_to(globals()[f'all_{m}'])

folium.LayerControl(collapsed=False).add_to(Map)
Map.save(os.path.join('result', 'Month.html'))


#####################################################################

### 시간
for m in range(1,25) :
    globals()[f'data_{m}'] = data[data.index.hour== m]

    temp = globals()[f'data_{m}'].groupby('add').size()
    temp = pd.DataFrame(temp).reset_index()
    temp = pd.merge(temp, ADD, on = 'add', how = 'left')
    temp.columns = ['add','freq','lat','lon']
    temp.sort_values(by= 'freq', ascending = False)

    arr = np.empty((0, 2), int)
    k = temp['freq'].quantile(0.90)

    for i in range(len(temp)) :    
        loc = [temp.iat[i,2],temp.iat[i,3]]
        freq = int(temp.iat[i,1]/k)
        if freq > 0 :
            arr = np.append(arr, np.array([loc]*freq), axis = 0)  

    globals()[f'arr_{m}'] = arr

Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)

for m in range(1,25) :
    globals()[f'all_{m}'] = folium.FeatureGroup(name=str(m)) 
    Map.add_child(globals()[f'all_{m}'])
    plugins.HeatMap(globals()[f'arr_{m}']).add_to(globals()[f'all_{m}'])

folium.LayerControl(collapsed=False).add_to(Map)
Map.save(os.path.join('result', 'Hour.html'))


#####################################################################

### 요일
for m in range(0,7) :
    globals()[f'data_{m}'] = data[data.index.dayofweek== m]

    temp = globals()[f'data_{m}'].groupby('add').size()
    temp = pd.DataFrame(temp).reset_index()
    temp = pd.merge(temp, ADD, on = 'add', how = 'left')
    temp.columns = ['add','freq','lat','lon']
    temp.sort_values(by= 'freq', ascending = False)

    arr = np.empty((0, 2), int)
    k = temp['freq'].quantile(0.90)

    for i in range(len(temp)) :    
        loc = [temp.iat[i,2],temp.iat[i,3]]
        freq = int(temp.iat[i,1]/k)
        if freq > 0 :
            arr = np.append(arr, np.array([loc]*freq), axis = 0)  

    globals()[f'arr_{m}'] = arr

Map = folium.Map([data['lat'].mean(), data['lon'].mean()], zoom_start=11)

Map.add_child(add_cctv)
Map.add_child(add_bus)
Map.add_child(add_police)
Map.add_child(add_fire)
Map.add_child(add_co)
Map.add_child(add_public)
Map.add_child(add_private)

for m in range(0,7) :
    globals()[f'all_{m}'] = folium.FeatureGroup(name=str(m)) 
    Map.add_child(globals()[f'all_{m}'])
    plugins.HeatMap(globals()[f'arr_{m}']).add_to(globals()[f'all_{m}'])

folium.LayerControl(collapsed=False).add_to(Map)
Map.save(os.path.join('result', 'day.html'))

#####################################################################

D = pd.read_csv("DATA_1000")

D = D[['add', 'freq', 'lat', 'lon']]
D['cctv'] = D['bus']= D['police']= D['fire']= D['co']= D['private'] = D['public'] = 0

D.head()

effect = [cctv, bus, police, fire, co, private, public]
Diff = [.05, .01, .5, .5, .5, .5, .5]

for a in range(len(effect)) :
    e = effect[a]
    diff = Diff[a]
    for i in range(len(D)) :
        x_loc = [ D.iat[i,2], D.iat[i,3]]
        for j in range(len(e)) :
            y_loc = [e.iat[j,1], e.iat[j,2]]
            d = haversine(x_loc, y_loc)
            if d < diff :
                D.iat[i,a+4] += 1

D.to_csv("regression_1000")
