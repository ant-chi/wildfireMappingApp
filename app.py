import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import geemap
import geemap.foliumap as fmap
import os
# from ipyleaflet import LegendControl
# import altair as alt
import folium
# import rasterio as rio
# from rasterio.plot import show

from funcs import *


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def loadData():
#     fires = gpd.read_file("data/norCalFires.geojson")
#     fires["Start"] = pd.DatetimeIndex(fires["Start"])
#     fires["End"] = pd.DatetimeIndex(fires["End"])
#
#     fires["geometry"] = fires["geometry"].apply(lambda x: boundsBuffer(x.bounds))
#     return fires

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def loadAltBaseLayer():
#     counties = gpd.read_file("data/CA_Counties/CA_Counties_TIGER2016.shp"
#                  ).to_crs("EPSG:4326")
#
#     sfLowerBound = counties[counties["NAME"]=="San Francisco"]["geometry"].bounds["maxy"].values[0]
#
#     norCal = counties.bounds.apply(lambda x: x[3]>sfLowerBound, axis=1)
#     norCalCounties = counties[norCal]
#
#     return alt.Chart(norCalCounties
#              ).mark_geoshape(fill="#E6E6E6", stroke="black"
#              ).encode(tooltip=[alt.Tooltip("NAME", title="County")]
#              )#.properties(width=500, height=500)


########## App starts here ##########
ee.Initialize()

st.set_page_config(layout="wide", page_title="INSERT TITLE", page_icon=":earth_americas:")
df = loadData()
# altBaseLayer = loadAltBaseLayer()

# non rescaled l8
l8_viz = {"bands": ["SR_B7", "SR_B5", "SR_B3"],
          "gamma": [1.1, 1.1, 1],
          "min": 1000, "max": 25000}

burn_viz = {"bands": ["burnSeverity"],
            "palette": ["706c1e", "4e9d5c", "fff70b", "ff641b", "a41fd6"],
            "min": 1, "max": 5}

nlcd_viz = {"bands": ["landCover"],
            "palette": ["A2D6F2", "FF7F68", "258914", "FFF100", "7CD860", "B99B56"],
            "min": 1, "max": 6}


# if not os.path.exist

with st.container():
    st.write("## Filter Fires")
    col_1, emptyCol_1, col_2 = st.columns([5, 1, 5])

    with col_1:
        startYear, endYear = st.select_slider(label="Year",
                                              options=[i for i in range(2013, 2022)],
                                              value=[2015, 2021],
                                              on_change=None)
    # change options
    with col_2:
        sizeClasses = st.multiselect(label="Size Class",
                                     options=["<F", "F", "G",
                                              "H", "I", "J+"],
                                     default=["H", "I", "J+"],
                                     on_change=None)


    counties = st.multiselect(label="County",
                              options=sorted(df["County"].unique()),
                              default=["Humboldt", "Lassen", "Shasta", "Siskiyou"],
                              on_change=None)


dfSubset = subsetFires(df, startYear, endYear, sizeClasses, counties)
st.write("#### {} fires in query".format(dfSubset.shape[0]))
# queriedFires = st.expander("View data")
with st.expander("View data"):
    col_3, col_4 = st.columns(2)
    temp = dfSubset[["Fire", "County", "Start", "End", "Acres", "Size Class"]]
    temp["Start"] = temp["Start"].apply(lambda x: str(x)[:10])
    temp["End"] = temp["End"].apply(lambda x: str(x)[:10])

    # tempChart = alt.layer(altBaseLayer,
    #                       alt.Chart(dfSubset
    #                         ).mark_geoshape(stroke="red", fill="pink"
    #                         ).encode(tooltip=["Fire", "Start", "End", "Acres", "Size Class"]))
    st.write(temp)
    # col_3.write(temp)
    # col_4.write(altBaseLayer, use_container_width=True)

with st.form("Map Fire"):
    col_5, emptyCol_2, col_6 = st.columns([5, 1, 5])
    selectBoxOptions = formatSelectBoxOptions(dfSubset)

    fireID = col_5.selectbox(label="Select Fire to Map",
                          options=list(selectBoxOptions.keys()),
                          format_func=lambda x: selectBoxOptions[x])

    model = col_6.selectbox(label="Select Supervised Classifier",
                         options=["Random Forests", "Blah", "Blah Blah"],
                         on_change=None)

    mapFireSubmit = st.form_submit_button("Map Fire")

if mapFireSubmit:
    for i in os.listdir("tifs"):
        if os.path.splitext(i)[1] != ".md":
            os.remove(os.path.join("tifs", i))

    fire_EE = geemap.gdf_to_ee(dfSubset[dfSubset["ID"]==fireID]).first()
    startDate, endDate = ee.Date(fire_EE.get("Start")), ee.Date(fire_EE.get("End"))
    fireGeometry = ee.Geometry(fire_EE.geometry())

    # look into partial image issue
    startCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
                ).filterDate(startDate.advance(-60, "day"), startDate
                ).filterBounds(fireGeometry
                ).sort("CLOUD_COVER", True
                ).limit(2
                ).mosaic()

    endCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
              ).filterDate(endDate, endDate.advance(60, "day")
              ).filterBounds(fireGeometry
              ).sort("CLOUD_COVER", True
              ).limit(2
              ).mosaic()

    img1, img2 = startCol.clip(fireGeometry), endCol.clip(fireGeometry)

    combined = prepImage(img1, img2, fireGeometry, endDate)
    loadTif(5, [30, 50, 75, 100, 130], fireID, combined, fireGeometry)

    with st.container():
        m2 = fmap.Map(add_google_map=False)

        add_legend(map=m2,
                   legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
                                        ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))

        # m2.add_legend(title="Burn Severity",
        #              labels=["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["<h4>Land Cover</h4>"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
        #              colors=burn_viz["palette"]+[None]+nlcd_viz["palette"])

        # m2.add_legend(title="Land Cover",
        #              labels=["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
        #              colors=nlcd_viz["palette"])

        # burnLegend = LegendControl(dict(zip(["Vegetation Growth", "Unburned", "Low", "Moderate", "High"],
        #                                     ["#"+i for i in burn_viz["palette"]])),
        #                            name="Burn Severity",
        #                            position="bottomleft")
        #
        # nlcdLegend = LegendControl(dict(zip(["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
        #                                     ["#"+i for i in nlcd_viz["palette"]])),
        #                            name="NLCD",
        #                            position="bottomright")
        #
        # m1.add_control(nlcdLegend)
        # m1.add_control(burnLegend)

        m2.addLayer(img1, l8_viz, "Pre-Fire L8")
        m2.addLayer(img2, l8_viz, "Post-Fire L8")

        m2.addLayer(combined.clip(fireGeometry), burn_viz, "Burn Severity")
        m2.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")


        # img = rxr.open_rasterio("tifs/{}.tif".format(fireID))
        # img_plot = img.where(~img.isnull(), img.min())

        # m2.add_child(folium.raster_layers.ImageOverlay(scaled_img,
        #                                       opacity=.9))

        m2.add_local_tile(source="tifs/{}.tif".format(fireID),
                          band=8,
                          palette="Reds",
                          vmin=1,   # comment out to show entire raster with bbox
                          vmax=5,
                          nodata=0,
                          layer_name="Local Tif")


        lon, lat = fireGeometry.centroid().getInfo()["coordinates"]
        m2.setCenter(lon, lat, zoom=10)

        emptyCol_3, col_7, emptyCol_4 = st.columns([1,3.5,1])
        with col_7:
            m2.to_streamlit(height=700, width=600, scrolling=True)
