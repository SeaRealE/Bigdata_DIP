#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import googlemaps

number = 0 
gmaps_key = '_GOOGLE MAP API KEY_'
gmaps = googlemaps.Client(key = gmaps_key)

addr = pd.read_csv('./_FILENAME_.csv', encoding="cp949")

# check reading
#addr.head(1)

addr["Latitude"] = ""
addr["Longitude"] = ""

# check adding columns
#addr.head(1)

for locate in addr["_LOCATION_"]:
    inf = gmaps.geocode(locate, language='ko')
    
    if not inf:
        continue
        
    else :
        lat = inf[0]['geometry']['location']['lat']
        addr["Latitude"][number] = lat

        lng = inf[0]['geometry']['location']['lng']  
        addr["Longitude"][number] = lng 
    
    number = number +1

addr.to_csv("output.csv", mode='w', index=False, encoding="cp949")
output = pd.read_csv('./output.csv', encoding="cp949")

# check result
#output



