import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import geemap
import geemap.foliumap as fmap
import os
import time
# from ipyleaflet import LegendControl
# import altair as alt
import folium
# import rasterio as rio
# from rasterio.plot import show

from funcs import *


########## App starts here ##########
ee.Initialize()

st.set_page_config(layout="wide", page_title="INSERT TITLE", page_icon=":earth_americas:")

# load and cache data
df = loadData()

# add session states
if "idLst" not in st.session_state:
    st.session_state["idLst"] = [42033]
if "currentIndex" not in st.session_state:
    st.session_state["currentIndex"] = 0
if "eeAssets" not in st.session_state:
    st.session_state["eeAssets"] = None


if not os.path.exists("rasters"):
    os.mkdir("rasters")

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


#
# st.markdown(
#     """
# |  | Vegetation Growth | Unburned | Low | Moderate | High | Predicted Total | Precision |
# | --- | --- | --- | --- | --- | --- | --- | --- |
# | **Vegetation Growth** | blah | blah | blah | blah | blah | blah | blah |
# | **Unburned** | blah | blah | blah | blah | blah | blah | blah |
# | **Low** | blah | blah | blah | blah | blah | blah | blah |
# | **Moderate** | blah | blah | blah | blah | blah | blah | blah |
# | **High** | blah | blah | blah | blah | blah | blah | blah |
# | **Actual Total** | blah | blah | blah | blah | blah | blah | blah |
# | **Recall** | blah | blah | blah | blah | blah | blah | blah |
# """
# )
#
#
# st.markdown(
#     """
# | Class | Precision | Recall |
# | --- | --- | --- |
# | **Vegetation Growth** | blah | blah |
# | **Unburned** | blah | blah |
# | **Low** | blah | blah |
# | **Moderate** | blah | blah |
# | **High** | blah | blah |
# | **Actual Total** | blah | blah |
# | **Recall** | blah | blah |
# """
# )
#
# st.write(pd.DataFrame({"dog":[1,224.5, 1.6784, 0.98431]}).round(2))

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
    startTime = time.time()
    # Tracks if fireID has changed. If not, data will be accessed from previous session state
    idLst, currentIndex = updateIdState(fireID)
    tempMessage = st.empty()

    if idLst[currentIndex-1] != idLst[currentIndex] or len(idLst)==2:
        tempMessage.write("##### Querying data")
        for i in os.listdir("rasters"):
            os.remove(os.path.join("rasters", i))

        preFireL8, postFireL8, combined, fireGeometry = prepData(dfSubset[dfSubset["ID"]==fireID])
        # wrap with function
        # fire_EE = geemap.gdf_to_ee(dfSubset[dfSubset["ID"]==fireID]).first()
        # startDate, endDate = ee.Date(fire_EE.get("Start")), ee.Date(fire_EE.get("End"))
        # fireGeometry = ee.Geometry(fire_EE.geometry())
        #
        # # look into partial image issue
        # startCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
        #             ).filterDate(startDate.advance(-60, "day"), startDate
        #             ).filterBounds(fireGeometry
        #             ).sort("CLOUD_COVER", True
        #             ).limit(2
        #             ).mosaic()
        #
        # endCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
        #           ).filterDate(endDate, endDate.advance(60, "day")
        #           ).filterBounds(fireGeometry
        #           ).sort("CLOUD_COVER", True
        #           ).limit(2
        #           ).mosaic()
        #
        # img1, img2 = startCol.clip(fireGeometry), endCol.clip(fireGeometry)

        # combined = prepImage(preFireL8, postFireL8, fireGeometry, endDate)
        st.session_state["eeAssets"] = [preFireL8, postFireL8, combined, fireGeometry]

        loadRaster([30, 60, 90, 120, 150], fireID, combined, fireGeometry)
        rasterToCsv("rasters", fireID)
    else: # access session_state variables
        preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeAssets"]


    with st.container():
        tempMessage.empty()
        df = pd.read_csv("rasters/{}.csv".format(fireID))
        # st.write(df.head(), df.shape)

        m = fmap.Map(add_google_map=False)

        add_legend(map=m,
                   legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
                                        ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))

        m.addLayer(preFireL8, l8_viz, "Pre-Fire L8")
        m.addLayer(postFireL8, l8_viz, "Post-Fire L8")

        m.addLayer(combined.clip(fireGeometry), burn_viz, "Burn Severity")
        m.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")

        m.add_local_tile(source="rasters/{}.tif".format(fireID),
                          band=8,
                          palette="Reds",
                          vmin=1,   # comment out to show entire raster with bbox
                          vmax=5,
                          nodata=0,
                          layer_name="Local Tif")


        lon, lat = fireGeometry.centroid().getInfo()["coordinates"]
        m.setCenter(lon, lat, zoom=10)
        m.add_layer_control()

        emptyCol_3, col_7, emptyCol_4 = st.columns([1,3.5,1])
        with col_7:
            m.to_streamlit(height=700, width=600, scrolling=True)

    st.write("Runtime: {} seconds".format(np.round((time.time()-startTime), 2)))


#################
    # for i in os.listdir("rasters"):
    #     if os.path.splitext(i)[1] != ".md":
    #         os.remove(os.path.join("rasters", i))
    #
    # fire_EE = geemap.gdf_to_ee(dfSubset[dfSubset["ID"]==fireID]).first()
    # startDate, endDate = ee.Date(fire_EE.get("Start")), ee.Date(fire_EE.get("End"))
    # fireGeometry = ee.Geometry(fire_EE.geometry())
    #
    # # look into partial image issue
    # startCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
    #             ).filterDate(startDate.advance(-60, "day"), startDate
    #             ).filterBounds(fireGeometry
    #             ).sort("CLOUD_COVER", True
    #             ).limit(2
    #             ).mosaic()
    #
    # endCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"
    #           ).filterDate(endDate, endDate.advance(60, "day")
    #           ).filterBounds(fireGeometry
    #           ).sort("CLOUD_COVER", True
    #           ).limit(2
    #           ).mosaic()
    #
    # img1, img2 = startCol.clip(fireGeometry), endCol.clip(fireGeometry)
    #
    # combined = prepImage(img1, img2, fireGeometry, endDate)
    # loadTif(5, [30, 60, 90, 120, 150], fireID, combined, fireGeometry)
    #
    # tifToCsv("rasters", fireID)
    #
    #
    # with st.container():
    #     m2 = fmap.Map(add_google_map=False)
    #
    #     add_legend(map=m2,
    #                legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
    #                                     ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))

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

        # m2.addLayer(img1, l8_viz, "Pre-Fire L8")
        # m2.addLayer(img2, l8_viz, "Post-Fire L8")
        #
        # m2.addLayer(combined.clip(fireGeometry), burn_viz, "Burn Severity")
        # m2.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")


        # img = rxr.open_rasterio("tifs/{}.tif".format(fireID))
        # img_plot = img.where(~img.isnull(), img.min())

        # m2.add_child(folium.raster_layers.ImageOverlay(scaled_img,
        #                                       opacity=.9))

        # m2.add_local_tile(source="rasters/{}.tif".format(fireID),
        #                   band=8,
        #                   palette="Reds",
        #                   vmin=1,   # comment out to show entire raster with bbox
        #                   vmax=5,
        #                   nodata=0,
        #                   layer_name="Local Tif")




        # lon, lat = fireGeometry.centroid().getInfo()["coordinates"]
        # m2.setCenter(lon, lat, zoom=10)
        #
        # emptyCol_3, col_7, emptyCol_4 = st.columns([1,3.5,1])
        # with col_7:
        #     m2.to_streamlit(height=700, width=600, scrolling=True)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed in <img src="https://avatars3.githubusercontent.com/u/45109972?s=400&v=4" width="25" height="25"> by <a style='display: block; text-align: center;' href="https://github.com/cashcountinchi/capstoneApp" target="_blank">Anthony Chi</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
