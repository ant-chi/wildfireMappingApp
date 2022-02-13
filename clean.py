import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime#, timedelta


fires = gpd.read_file("../data/California_Fire_Perimeters_(all).geojson")

counties = gpd.read_file("../data/CA_Counties/CA_Counties_TIGER2016.shp").to_crs("EPSG:4326")
sfLowerBound = counties[counties["NAME"]=="San Francisco"]["geometry"].bounds["maxy"].values[0]
norCalCounties = counties[counties.bounds.apply(lambda x: x[3]>sfLowerBound, axis=1)]
countyGeo = {county: geometry for county, geometry in norCalCounties[["NAME", "geometry"]].values}

norCalFires = fires.bounds.apply(lambda x: x[1]>sfLowerBound, axis=1)

subFires = fires[(fires["YEAR_"].astype(int) >= 2013) & (norCalFires)]
subFires["YEAR_"] = subFires["YEAR_"].astype(int)
invalidDate = (subFires["ALARM_DATE"].apply(oobDate).isna()) | (subFires["CONT_DATE"].apply(oobDate).isna())

subFires = subFires[(~invalidDate) & (subFires["GIS_ACRES"] >= 300)]


subFires["centroid"] = subFires.centroid

countyLst = []
for centroid in subFires["centroid"].values:
    for county, geometry in countyGeo.items():
        appended = False
        if centroid.within(geometry):
            countyLst.append(county)
            appended = True
            break
        else:
            continue
    if not appended:  # Fires not in CA
        countyLst.append(None)

subFires["County"] = countyLst

subFires = subFires[~subFires["County"].isna()]

subFires["ALARM_DATE"] = subFires["ALARM_DATE"].apply(lambda x: datetime.fromisoformat(x[:10]))
subFires["CONT_DATE"] = subFires["CONT_DATE"].apply(lambda x: datetime.fromisoformat(x[:10]))

subFires["Month"] = subFires["ALARM_DATE"].apply(lambda x: x.month)
subFires["Size Class"] = subFires["GIS_ACRES"].apply(sizeCode)

subFires = subFires[["OBJECTID", "FIRE_NAME", "County", "YEAR_", "Month",
                     "ALARM_DATE", "CONT_DATE", "GIS_ACRES", "Size Class", "geometry"]]

subFires.columns = ["ID", "Fire", "County", "Year", "Month", "Start", "End", "Acres", "Size Class", "geometry"]

# subFires["bbox"] = subFires["geometry"].apply(lambda x: boundsBuffer(x.bounds))
