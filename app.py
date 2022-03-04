import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import geemap
import geemap.foliumap as fmap
import os
import time
import folium
# from folium import plugins
# from PIL import Image
from funcs import *


st.set_page_config(layout="wide", page_title="INSERT TITLE", page_icon=":earth_americas:")

# initialize EE + cache data and models
geemap.ee_initialize()

df = loadData()
models = loadModels()


# initialize session states
if "idLst" not in st.session_state:
    st.session_state["idLst"] = [0]          # track changes in selected fire to avoid requerying data
if "currentIndex" not in st.session_state:
    st.session_state["currentIndex"] = 0     # tracks current fire ID's position in session state
if "eeObjects" not in st.session_state:
    st.session_state["eeObjects"] = None     # stores necessary EE objects if data is queried
if "rasterDims" not in st.session_state:
    st.session_state["rasterDims"] = None

# ee viz params
l8_432 = {"bands": ["SR_B4", "SR_B3", "SR_B2"],
          "gamma": [1.1, 1.1, 1],
          "min": 1000, "max": 25000}

l8_753 = {"bands": ["SR_B7", "SR_B5", "SR_B3"],
          "gamma": [1.1, 1.1, 1],
          "min": 1000, "max": 25000}

burn_viz = {"bands": ["burnSeverity"],
            "palette": ["706c1e", "4e9d5c", "fff70b", "ff641b", "a41fd6"],
            "min": 1, "max": 5}

nlcd_viz = {"bands": ["landCoverViz"],
            "palette": ["A2D6F2", "FF7F68", "258914", "FFF100", "7CD860", "B99B56"],
            "min": 1, "max": 6}

monthMap = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
            8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}


# changes sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# with st.sidebar:
#     st.write("# How does this app work?")
#     st.video("https://www.youtube.com/watch?v=5qap5aO4i9A")
#
#     st.write("# Project Details")
#     st.write("## [Visit our project repo!](https://github.com/a2lu/CAPSTONE_WILDFIRE)")
#     st.write("## [Visit our site for more details on this project!](https://cashcountinchi.github.io/b12_capstone/)")


with st.container():
    st.write("## Filter Fires")
    col_1, emptyCol_1, col_2 = st.columns([5, 1, 5])
    col_3, emptyCol_2, col_4 = st.columns([5, 1, 5])

    with col_1:
        startYear, endYear = st.select_slider(label="Year",
                                              options=[i for i in range(2013, 2022)],
                                              value=[2013, 2021],
                                              on_change=None)
    with col_2:
        endMonths = st.multiselect(label="Fire Containment Month",
                                   options=list(monthMap.keys()),
                                   format_func=lambda x: monthMap[x],
                                   default=[6, 7, 8, 9],
                                   help="Month that a fire is contained/extinguished.\
                                   \n Recommended months are June-October.")
    with col_3:
        counties = st.multiselect(label="County",
                                  options=sorted(df["County"].unique()),
                                  default=["Humboldt", "Lassen", "Mendocino",
                                           "Modoc", "Shasta", "Siskiyou"],
                                  on_change=None,
                                  help="Counties in Northern California")

    with col_4:
        sizeClasses = st.multiselect(label="Size Class",
                                     options=["E", "F", "G",
                                              "H", "I", "J+"],
                                     default=["H", "I", "J+"],
                                     on_change=None,
                                     help="National Wildfire Coordination Group (NWCG) wildfire size classes\
                                     \n E: 300-999 acres\
                                     \n F: 1000-4999 acres\
                                     \n G: 5000-9999 acres\
                                     \n H: 10000-49999 acres\
                                     \n I: 50000-99999 acres\
                                     \n J+: 100000+ acres")

dfSubset = subsetFires(df, startYear, endYear, endMonths, sizeClasses, counties)
st.write("#### {} fires in query".format(dfSubset.shape[0]))

with st.expander("View fire data"):
    temp = dfSubset[["Fire", "County", "Start", "End", "Acres", "Size Class"]]
    temp["Start"] = temp["Start"].apply(lambda x: str(x)[:10])
    temp["End"] = temp["End"].apply(lambda x: str(x)[:10])

    st.write(temp)


with st.form("Map Fire"):
    col_5, emptyCol_2, col_6 = st.columns([5, 1, 5])
    selectBoxOptions = formatFireSelectBox(dfSubset)

    fireID = col_5.selectbox(label="Select Fire to Map",
                          options=list(selectBoxOptions.keys()),
                          format_func=lambda x: selectBoxOptions[x])

    modelKey = col_6.selectbox(label="Select Supervised Classifier",
                               options=list(models.keys()),
                               on_change=None)

    mapFireSubmit = st.form_submit_button("Map Fire")


if mapFireSubmit:
    startTime = time.time()
    fireData = dfSubset[dfSubset["ID"]==fireID]
    fireBounds = list(fireData["geometry"].bounds.values[0])

    # with winterWarning:
    if list(fireData["Contained Month"])[0] in [11,12,1,2]:
            # winterWarning.clear()
        st.warning("##### Snow and seasonal changes in vegetation can produce \
        inaccurate/skewed results for winter fires.")

    model = models[modelKey]

    # Tracks if fireID has changed. If not, data will be accessed from previous session state
    idLst, currentIndex = updateIdState(fireID)
    tempMessage = st.empty()

    if idLst[currentIndex-1] != idLst[currentIndex] or len(idLst)==2:
        tempMessage.write("#### Querying data.....")

        # delete saved files
        for i in os.listdir():
            if os.path.splitext(i)[1] in [".tif", ".csv", ".xml", ".png", ".parquet"]:
                os.remove(i)

        st.session_state["eeObjects"] = prepImages(dfSubset[dfSubset["ID"]==fireID])
        preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeObjects"]

        # fileName = "{}.tif".format(fireID)


        # Download raster from EE and convert to parquet
        # loadRaster([30, 60, 90, 120, 150], fileName, combined, fireGeometry)
        # rasterToParquet(fileName)
        downloadRaster([30, 60, 90, 120, 150], combined, fireGeometry)
        rasterToParquet()

    else: # access session_state variables
        preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeObjects"]


    with st.container():
        tempMessage.write("#### Running model and rendering map.....")

        # df = pd.read_parquet("{}.parquet".format(fireID))
        df = pd.read_parquet("raster.parquet")
        modelData = prepData(df[['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                                 'NDVI','elevation', 'percent_tree_cover', 'landCover']])

        labels, predictions = df["burnSeverity"].values, model.predict(modelData)

        # +1 for oscar models
        if modelKey in ["log_boost", "SVM"]:
            predictions = predictions + 1

        # png of actual+predicted burn severity from raster data
        burnSeverityImage(labels, st.session_state["rasterDims"], "actualBS.png")
        burnSeverityImage(predictions, st.session_state["rasterDims"], "predictedBS.png")

        # initialize geemap.foliumMap and adds legend + image layers
        m = fmap.Map(add_google_map=False, plugin_LatLngPopup=False)
        add_legend(map=m,
                   legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
                                        ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))

        m.addLayer(preFireL8, l8_432, "Pre-Fire RGB")
        m.addLayer(postFireL8, l8_432, "Post-Fire RGB")

        m.addLayer(preFireL8, l8_753, "Pre-Fire (753)")
        m.addLayer(postFireL8, l8_753, "Post-Fire (753)")

        m.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")

        # adds burn severity png's as folium layers
        png_1 = folium.raster_layers.ImageOverlay(name='Burn Severity',
                                                  image="actualBS.png",
                                                  bounds=[fireBounds[:2][::-1],
                                                          fireBounds[2:][::-1]],
                                                  interactive=True)
        png_1.add_to(m)

        png_2 = folium.raster_layers.ImageOverlay(name='Predicted Burn Severity',
                                                  image="predictedBS.png",
                                                  bounds=[fireBounds[:2][::-1],
                                                          fireBounds[2:][::-1]],
                                                  interactive=True)
        png_2.add_to(m)

        lon, lat = fireGeometry.centroid().getInfo()["coordinates"]
        m.setCenter(lon, lat, zoom=10)
        m.add_layer_control()

        # chart_1, chart_2 = altChart(df)
        lcChart = altChart(df)
        cm, metrics = modelMetrics(labels, predictions)
        metrics = metrics.style.format(subset=["Precision (%)", "Recall (%)", "F1 (%)"],
                                       formatter="{:.2f}").set_properties(**{'text-align': 'center'})

        tempMessage.empty()

        emptyCol_3, col_7, emptyCol_4 = st.columns([1,3.75,1])
        with col_7:
            st.write("#### {} Accuracy: {}%".format(modelKey, np.round(100*np.mean(labels==predictions), 2)))
            m.to_streamlit(height=670, width=600, scrolling=True)

            st.write(cm.style.set_properties(**{'text-align': 'center'}).to_html(),
                     unsafe_allow_html=True)
            st.write(metrics.to_html(),
                     unsafe_allow_html=True)
            st.altair_chart(lcChart)




        # col_8, col_9 = st.columns(2)

        # col_8.write(cm.style.set_properties(**{'text-align': 'center'}).to_html(),
        #          unsafe_allow_html=True)
        # col_9.write(metrics.to_html(),
        #          unsafe_allow_html=True)
        #
        # tempMessage.empty()
        #
        # st.altair_chart(lcChart)

        # st.write(cm.style.set_properties(**{'text-align': 'center'}).to_html(),
        #          unsafe_allow_html=True)
        # st.write(metrics.to_html(),
        #          unsafe_allow_html=True)

        # st.altair_chart(chart_1)
        # st.altair_chart(chart_2)

    st.success("#### Total Runtime: {} seconds".format(np.round((time.time()-startTime), 2)))


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
<p>Developed in <img src="https://avatars3.githubusercontent.com/u/45109972?s=400&v=4" width="25" height="25"> by <a href="https://github.com/cashcountinchi/capstoneApp" target="_blank">Anthony Chi</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
