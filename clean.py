import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import date

def sizeCode(x):
    if 1000 <= x < 5000:
        return "F"
    elif 5000 <= x < 10000:
        return "G"
    elif 10000 <= x < 50000:
        return "H"
    elif 50000 <= x < 100000:
        return "I"
    elif 100000 <= x:
        return "J+"
    else:
        return "E"


def oobDate(x):
    """
    Returns None if Datetime object is out of bounds
    """
    try:
        x = pd.to_datetime(x)
        return x
    except:
        return None


fires = gpd.read_file("data/California_Fire_Perimeters_(all).geojson")

counties = gpd.read_file("data/CA_Counties/CA_Counties_TIGER2016.shp").to_crs("EPSG:4326")
sfLowerBound = counties[counties["NAME"]=="San Francisco"]["geometry"].bounds["maxy"].values[0]
norCalCounties = counties[counties.bounds.apply(lambda x: x[3]>sfLowerBound, axis=1)]
countyGeo = {county: geometry for county, geometry in norCalCounties[["NAME", "geometry"]].values}

norCalFires = fires.bounds.apply(lambda x: x[1]>sfLowerBound, axis=1)

subFires = fires[(fires["YEAR_"].astype(int) >= 2013) & (norCalFires)]
subFires["YEAR_"] = subFires["YEAR_"].astype(int)
invalidDate = (subFires["ALARM_DATE"].apply(oobDate).isna()) | (subFires["CONT_DATE"].apply(oobDate).isna())

subFires["GIS_ACRES"] = subFires["GIS_ACRES"].astype(int)
subFires = subFires[(~invalidDate) & (subFires["GIS_ACRES"] >= 500)]


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

subFires["ALARM_DATE"] = subFires["ALARM_DATE"].apply(lambda x: x[:10])
subFires["CONT_DATE"] = subFires["CONT_DATE"].apply(lambda x: x[:10])

subFires["Contained Month"] = subFires["CONT_DATE"].apply(lambda x: date.fromisoformat(x).month)

subFires["Size Class"] = subFires["GIS_ACRES"].apply(sizeCode)

subFires = subFires[["OBJECTID", "FIRE_NAME", "County", "YEAR_", "Contained Month",
                     "ALARM_DATE", "CONT_DATE", "GIS_ACRES", "Size Class", "geometry"]]

subFires.columns = ["ID", "Fire", "County", "Year", "Contained Month", "Start", "End", "Acres", "Size Class", "geometry"]


subFires.reset_index(drop=True).to_file("data/norCalFires.geojson", driver="GeoJSON")
