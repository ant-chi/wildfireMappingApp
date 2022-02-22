import streamlit as st
import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from datetime import datetime
from shapely.geometry import Polygon
import os
import altair as alt
import rasterio as rio
from functools import reduce
from operator import iconcat


def boundsBuffer(x, buffer=0):
    """
    Creates a bounding box with a default 7.5% buffer

    Args:
        x: GeoPandas geometry object

    Returns:
        Shapely.Polygon that represents buffered bounding box over input geometry
    """
    minx, miny, maxx, maxy = x
    buffer_x, buffer_y = np.abs(buffer*(maxx-minx)), np.abs(buffer*(maxy-miny))
    minx -= buffer_x
    maxx += buffer_x
    miny -= buffer_y
    maxy += buffer_y

    coords = ((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny))
    return Polygon(coords)


def convertDate(date):
    """
    Converts EE.Date or unix date to Y-M-D formst
    """
    if isinstance(date, ee.Date):
        date = date.getInfo()["value"]

    return datetime.utcfromtimestamp(date/1000).strftime("%Y-%m-%d")# %H:%M:%S')


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def loadData():
    fires = gpd.read_file("data/norCalFires.geojson")
    fires["Start"] = pd.DatetimeIndex(fires["Start"])
    fires["End"] = pd.DatetimeIndex(fires["End"])

    # fires["geometry"] = fires["geometry"].apply(lambda x: boundsBuffer(x.bounds))
    return fires


def updateIdState(fireID):
    st.session_state['idLst'].append(fireID)
    st.session_state['currentIndex'] += 1

    return st.session_state['idLst'], st.session_state['currentIndex']

def updateEE(preFireImage, postFireImage, combined, geometry):
    st.session_state["eeAssets"] = [preFireImage, postFireImage, combined, geometry]

def formatSelectBoxOptions(data):
    return {k: "{} ({})".format(v1, v2) for k, v1, v2 in data[["ID", "Fire", "Year"]].sort_values(by="Fire").values}



def subsetFires(data, startYear, endYear, sizeClass=None, counties=None):
    if startYear == endYear:
        st.error("### Select valid time interval")
    else:
        subset = data[(data["Year"] >= startYear) & (data["Year"] < endYear)]

    if len(sizeClass) == 0:
        st.warning("#### Size Class is not selected (Filter will not be applied)")
    else:
        subset = subset[subset["Size Class"].isin(sizeClass)]

    if len(counties) == 0:
        st.warning("#### County is not selected (Filter will not be applied)")
    else:
        subset = subset[subset["County"].isin(counties)]

    return subset.sort_values(by="Fire").reset_index(drop=True)

def prepData(fireGPD):
    fire_EE = geemap.gdf_to_ee(fireGPD).first()
    # year = fireGPD["Start"].year ###
    startDate, endDate = ee.Date(fire_EE.get("Start")), ee.Date(fire_EE.get("End"))
    fireGeometry = ee.Geometry(fire_EE.geometry())

    preFireImage = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
                    ).filterDate(startDate.advance(-60, "day"), startDate
                    ).filterBounds(fireGeometry
                    ).sort("CLOUD_COVER", True
                    ).limit(2
                    ).mosaic(
                    ).clip(fireGeometry)

    postFireImage = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
                     ).filterDate(endDate, endDate.advance(60, "day")
                     ).filterBounds(fireGeometry
                     ).sort("CLOUD_COVER", True
                     ).limit(2
                     ).mosaic(
                     ).clip(fireGeometry)

    # Calculate NBR, dNBR, and burn severity
    preFireNBR = preFireImage.normalizedDifference(['SR_B5', 'SR_B7'])
    postFireNBR = postFireImage.normalizedDifference(['SR_B5', 'SR_B7'])
    dNBR = (preFireNBR.subtract(postFireNBR)
                     ).multiply(1000
                     ).rename("dNBR")

    burnSeverity = dNBR.expression(" (b('dNBR') > 425) ? 5 "    # purple: high severity
                                   ":(b('dNBR') > 225) ? 4 "    # orange: moderate severity
                                   ":(b('dNBR') > 100) ? 3 "    # yellow: low severity
                                   ":(b('dNBR') > -60) ? 2 "    # green: unburned/unchanged
                                   ":(b('dNBR') <= -60) ? 1 "   # brown: vegetation growth
                                   ":0"                         # pseudo mask
                      ).rename("burnSeverity")

    # Get SRTM elevation, NLCD land coverpostFireImageNDVI, and GRIDMET weather
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

    nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2016_REL'
               ).filter(ee.Filter.eq('system:index', '2016')).first()

    lc = nlcd.select("landcover"
            ).expression(" (b('landcover') > 90) ? 1 "    # blue: other (wetland)
                         ":(b('landcover') > 80) ? 6 "    # brown: agriculture
                         ":(b('landcover') > 70) ? 5 "    # lightGreen: grassland/herbaceous
                         ":(b('landcover') > 50) ? 4 "    # yellow: shrub
                         ":(b('landcover') > 40) ? 3 "    # green: forest
                         ":(b('landcover') > 30) ? 1 "    # blue: other (barren land)
                         ":(b('landcover') > 20) ? 2 "    # red: developed/urban
                         ":(b('landcover') > 10) ? 1 "    # blue: other (water/perennial ice+snow)
                         ":0"                             # handle for potential exceptions
            ).rename("landCover")


    lcViz = ee.ImageCollection('USGS/NLCD_RELEASES/2016_REL'
             ).filter(ee.Filter.eq('system:index', '2011')
             ).first(
             ).select("landcover"
             ).expression(" (b('landcover') > 90) ? 1 "    # blue: other (wetland)
                          ":(b('landcover') > 80) ? 6 "    # brown: agriculture
                          ":(b('landcover') > 70) ? 5 "    # lightGreen: grassland/herbaceous
                          ":(b('landcover') > 50) ? 4 "    # yellow: shrub
                          ":(b('landcover') > 40) ? 3 "    # green: forest
                          ":(b('landcover') > 30) ? 1 "    # blue: other (barren land)
                          ":(b('landcover') > 20) ? 2 "    # red: developed/urban
                          ":(b('landcover') > 10) ? 1 "    # blue: other (water/perennial ice+snow)
                          ":0"                             # handle for potential exceptions)
             ).clip(fireGeometry
             ).rename("landCoverViz")


    ndvi = postFireImage.normalizedDifference(["SR_B5", "SR_B4"]
                       ).rename("NDVI"
                       ).multiply(1000)

    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET"
               ).filterBounds(fireGeometry
               ).filterDate(endDate.advance(-3, "day"), endDate
               ).mean()

    # Merge all image bands together
    combined = postFireImage.select('SR_B.'          # post-fire L8 bands 1-7
                           ).addBands(burnSeverity   # classified burn severity
                           ).addBands(dNBR           # dNBR
                           ).addBands(ndvi           # post-fire NDVI
                           ).addBands(dem            # SRTM elevation
                           ).addBands(gridmet        # all GRIDMET bands
                           ).addBands(nlcd.select("percent_tree_cover")
                           ).addBands(lc             # simplified landCover for model
                           ).addBands(lcViz)         # simplified landCover for viz

    return [preFireImage, postFireImage, combined, fireGeometry]


def loadRaster(imgScale, fireID, image, geometry):
    startTime = time.time()
    filename = "{}.tif".format(fireID)
    numTries = len(imgScale)
    success = False
    for i in range(numTries):
        try:
            geemap.ee_export_image(ee_object=image,
                                   filename=filename,
                                   scale=imgScale[i],
                                   region=geometry)
            success = True
            resolution = imgScale[i]
            break
        except Exception:
            continue

    if success:
        st.success("##### Downloaded raster at {}m scale in {} seconds".format(resolution, np.round((time.time()-startTime), 2)))
    else:
        st.error("#### Fire exceeds total request size")


def rasterToCsv(path):
    colNames = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
                'burnSeverity','dNBR','NDVI','elevation','pr','rmax','rmin',
                'sph','srad','th','tmmn','tmmx','vs','erc','eto','bi','fm100',
                'fm1000','etr','vpd','percent_tree_cover','landCover','landCoverViz']

    fireID = os.path.splitext(path)[0] ##
    intCols = colNames[:11] + colNames[-3:]
    floatCols = colNames[11:-3]
    colNames = {index+1:value for index, value in enumerate(colNames)}

    # path = os.path.join(dir, "{}.tif".format(fireID))

    # Open raster and store band data with dict
    img, data = rio.open(path), {}
    for index, val in colNames.items():
        data[val] = reduce(iconcat, img.read(index), [])

    # Convert to df, remove NA's if present (unlikely), apply pseudo mask, and cast to int for memory reduction
    df = pd.DataFrame(data).dropna()
    df = df[df["burnSeverity"]>0].reset_index(drop=True).round(2)
    df[intCols] = df[intCols].astype(int)
    df.to_csv("rasters/{}.csv".format(fireID), index=False)




# ############

def add_legend(map, legend_dict=None, opacity=1.0):
    """Adds a customized basemap to the map. Reference: https://bit.ly/3oV6vnH

    Args:
        title (str, optional): Title of the legend. Defaults to 'Legend'. Defaults to "Legend".
        colors ([type], optional): A list of legend colors. Defaults to None.
        labels ([type], optional): A list of legend labels. Defaults to None.
        legend_dict ([type], optional): A dictionary containing legend items as keys and color as values. If provided, legend_keys and legend_colors will be ignored. Defaults to None.
        builtin_legend ([type], optional): Name of the builtin legend to add to the map. Defaults to None.
        opacity (float, optional): The opacity of the legend. Defaults to 1.0.

    """
    from branca.element import MacroElement, Template

    legend_template = os.path.join("legendTemplate.txt")

    if legend_dict is not None:
        if not isinstance(legend_dict, dict):
            raise ValueError("The legend dict must be a dictionary.")
        else:
            labels = list(legend_dict.keys())
            colors = list(legend_dict.values())
            colors = ["#" + color for color in colors]

    content = []
    noneCount = 0
    with open(legend_template) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index <= 36:
                content.append(line)
            elif index < 39:
                content.append(line)
            elif index == 39:
                for i, color in enumerate(colors):
                    if color == "#None":
                        noneCount += 1
                        if noneCount == 2:
                            content.append("<div class='legend-title'><br>{}</div>".format(labels[i]))
                        else:
                            content.append("<div class='legend-title'>{}</div>".format(labels[i]))
                    else:
                        item = f"    <li><span style='background:{color};opacity:{opacity};'></span>{labels[i]}</li>\n"
                        content.append(item)
            elif index > 41:
                content.append(line)

    template = "".join(content)
    macro = MacroElement()
    macro._template = Template(template)

    return map.get_root().add_child(macro)


def altChart(data):
    bsMap = {1: "Vegetation Growth", 2: "Unburned", 3: "Low", 4: "Moderate", 5: "High"}
    lcMap = {1: "Other", 2: "Developed", 3: "Forest", 4: "Shrub", 5: "Grassland", 6: "Agriculture"}

    bsPivot = data.pivot_table(index="burnSeverity",
                               values="SR_B1",
                               aggfunc=len
                 ).reset_index(
                 ).sort_values(by="burnSeverity"
                 ).rename(columns={"burnSeverity": "Burn Severity",
                                   "SR_B1": "Percentage"})
    bsPivot["Percentage"] /= bsPivot["Percentage"].sum()
    bsPivot["Percentage"] = (100*bsPivot["Percentage"]).round(2)
    bsPivot["Burn Severity"] = bsPivot["Burn Severity"].apply(lambda x: bsMap[x])


    lcPivot = data.pivot_table(index="landCoverViz",
                               values="SR_B1",
                               aggfunc=len
                 ).reset_index(
                 ).sort_values(by="landCoverViz"
                 ).rename(columns={"landCoverViz": "Land Cover",
                                   "SR_B1": "Percentage"})
    lcPivot["Percentage"] /= lcPivot["Percentage"].sum()
    lcPivot["Percentage"] = (100*lcPivot["Percentage"]).round(2)
    lcPivot["Land Cover"] = lcPivot["Land Cover"].apply(lambda x: lcMap[x])


    bsChart = alt.Chart(bsPivot
                ).mark_bar(
                ).encode(x="Percentage:Q",
                         y=alt.X("Burn Severity:O", sort=list(bsMap.values())),
                         color=alt.Color("Burn Severity",
                                         legend=None,
                                         scale=alt.Scale(domain=list(bsMap.values()),
                                                         range=["rgb(112,108,30)", "rgb(78,157,92)",
                                                                "rgb(255,247,11)", "rgb(255,100,27)",
                                                                "rgb(164,31,214)"])),
                         tooltip=["Burn Severity", "Percentage"])

    lcChart = alt.Chart(lcPivot
                ).mark_bar(
                ).encode(x="Percentage:Q",
                         y=alt.X("Land Cover:O", sort=list(lcMap.values())),
                         color=alt.Color("Land Cover",
                                         legend=None,
                                         scale=alt.Scale(domain=list(lcMap.values()),
                                                         range=["rgb(162,214,242)", "rgb(255,127,104)",
                                                                "rgb(37,137,20)", "rgb(255,241,0)",
                                                                "rgb(123,216,96)", "rgb(185,155,86)"])),
                         tooltip=["Land Cover", "Percentage"])

    return bsChart, lcChart
