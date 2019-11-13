import pandas as pd
import folium # 설치 필요 : pip install folium
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


year_2014 = pd.read_csv("./2014년 적발수(위반장소명_위도_경도_적발수).csv", encoding="cp949")
year_2016 = pd.read_csv("./2016년 적발수(위반장소명_위도_경도_적발수).csv", encoding="cp949")
year_2017 = pd.read_csv("./2017년 적발수(위반장소명_위도_경도_적발수).csv", encoding="cp949")
year_2018 = pd.read_csv("./2018년 적발수(위반장소명_위도_경도_적발수).csv", encoding="cp949")

year_2014['적발수'].sum(), year_2016['적발수'].sum(), year_2017['적발수'].sum(), year_2018['적발수'].sum()


# 데이터 크기 확인
# year_2014['적발수'].plot()
# for index in year_2014.index:
#     if year_2014.loc[index, '적발수'] > 500:
#         year_2014.loc[index, '적발수'] = 500
# year_2014['적발수'].plot()

# year_2016['적발수'].plot()
# for index in year_2016.index:
#     if year_2016.loc[index, '적발수'] > 500:
#         year_2016.loc[index, '적발수'] = 500
# year_2016['적발수'].plot()

# year_2017['적발수'].plot()
# for index in year_2017.index:
#     if year_2017.loc[index, '적발수'] > 500:
#         year_2017.loc[index, '적발수'] = 500
# year_2017['적발수'].plot()

# year_2018['적발수'].plot()
# for index in year_2018.index:
#     if year_2018.loc[index, '적발수'] > 500:
#         year_2018.loc[index, '적발수'] = 500
# year_2018['적발수'].plot()

import folium

# 반월당 교차로 중심
lon = 35.865509
lat = 128.593416

daegu = folium.Map(location= [lon, lat], zoom_start=12)
# daegu1 = folium.Map(location= [lon, lat], zoom_start=12)
# daegu2 = folium.Map(location= [lon, lat], zoom_start=12)
# daegu3 = folium.Map(location= [lon, lat], zoom_start=12)
# daegu4 = folium.Map(location= [lon, lat], zoom_start=12)

for index in year_2014.index:
    lat = year_2014.loc[index, '위도']
    lon = year_2014.loc[index, '경도']

    # folium.CircleMarker( [lat,lon],
    #                    radius = year_2014.loc[index, '적발수']/20,
    #                    popup = year_2014.loc[index, '위반장소명'],
    #                    color='blue', fill= True).add_to(daegu1)
    
    folium.CircleMarker( [lat,lon],
                       radius = year_2014.loc[index, '적발수']/20,
                       popup = year_2014.loc[index, '위반장소명'],
                       color='blue', fill= True).add_to(daegu)   
    if index > 300:
        break
    
# daegu1

for index in year_2016.index:
    lat = year_2016.loc[index, '위도']
    lon = year_2016.loc[index, '경도']
    
    # folium.CircleMarker( [lat,lon],
    #                    radius = year_2016.loc[index, '적발수']/20,
    #                    popup = year_2016.loc[index, '위반장소명'],
    #                    color='red', fill= True).add_to(daegu2)
    
    folium.CircleMarker( [lat,lon],
                       radius = year_2016.loc[index, '적발수']/20,
                       popup = year_2016.loc[index, '위반장소명'],
                       color='red', fill= True).add_to(daegu)
    
    
    if index > 300:
        break
    
# daegu2

for index in year_2017.index:
    lat = year_2017.loc[index, '위도']
    lon = year_2017.loc[index, '경도']
    
    # folium.CircleMarker( [lat,lon],
    #                    radius = year_2017.loc[index, '적발수']/20,
    #                    popup = year_2017.loc[index, '위반장소명'],
    #                    color='green', fill= True).add_to(daegu3)
    
    folium.CircleMarker( [lat,lon],
                       radius = year_2017.loc[index, '적발수']/20,
                       popup = year_2017.loc[index, '위반장소명'],
                       color='green', fill= True).add_to(daegu)
    
    if index > 300:
        break
    
# daegu3

for index in year_2018.index:
    lat = year_2018.loc[index, '위도']
    lon = year_2018.loc[index, '경도']
    
    # folium.CircleMarker( [lat,lon],
    #                    radius = year_2018.loc[index, '적발수']/20,
    #                    popup = year_2018.loc[index, '위반장소명'],
    #                    color='brown', fill= True).add_to(daegu4)
    
    folium.CircleMarker( [lat,lon],
                       radius = year_2018.loc[index, '적발수']/20,
                       popup = year_2018.loc[index, '위반장소명'],
                       color='brown', fill= True).add_to(daegu)
    
    if index > 300:
        break
    
# daegu4

daegu





