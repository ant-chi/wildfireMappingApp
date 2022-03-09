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
from datetime import date, timedelta
from funcs import *


st.set_page_config(layout="wide", page_title=" ", page_icon=":earth_americas:")

# initialize EE + cache data, models, folium drawMap
geemap.ee_initialize()

df = loadData()
models = loadModels()
drawMap = loadDrawMap()

if "eeObjects" not in st.session_state:
    st.session_state["eeObjects"] = None     # caches necessary EE objects if data is queried
if "rasterDims" not in st.session_state:
    st.session_state["rasterDims"] = None    # caches raster dimensions

# ee viz params
l8_432 = {"bands": ["SR_B4", "SR_B3", "SR_B2"],
          "gamma": [1.1, 1.1, 1],
          "min": 2000, "max": 22500}

l8_753 = {"bands": ["SR_B7", "SR_B5", "SR_B3"],
          "gamma": [1, 1.1, 1],
          "min": 1500, "max": 22500}

burn_viz = {"bands": ["burnSeverity"],
            "palette": ["706c1e", "4e9d5c", "fff70b", "ff641b", "a41fd6"],
            "min": 1, "max": 5}

nlcd_viz = {"bands": ["landCoverViz"],
            "palette": ["A2D6F2", "FF7F68", "258914", "FFF100", "7CD860", "B99B56"],
            "min": 1, "max": 6}

monthMap = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
            8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}


# increase sidebar width
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
    unsafe_allow_html=True)


with st.sidebar:
    manual = st.checkbox(label="Manual Fire Mapping",
                         value=False)

    st.write("# How does this app work?")
    st.video("https://www.youtube.com/watch?v=5qap5aO4i9A")

    st.write("# :star2: Project Links :star2:")
    st.write("## [Visit our site for more details on this project!](https://cashcountinchi.github.io/b12_capstone/)",
             "\n",
             "## [Visit our Github!](https://github.com/cashcountinchi/capstoneApp)&nbsp;&nbsp;&nbsp;\
             <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' alt='Github logo' align='middle' style='width:25;height:25px;'>",
             unsafe_allow_html=True)

    st.write("# :mailbox: Contact Information :mailbox:")
    sidebarContactInfo()

if manual:
    st.write("### Manual Fire Mapping")
    col_1, col_2 = st.columns([1, 0.6])

    if "idLst" in st.session_state:
        del st.session_state["idLst"]
    if "currentIndex" in st.session_state:
        del st.session_state["currentIndex"]


    if "widgetState" not in st.session_state:
        st.session_state["widgetState"] = [np.array([0,0,0])]
    if "currentState" not in st.session_state:
        st.session_state["currentState"] = 0


    # folium map for drawing custom polygons
    with col_1:
        drawMap.to_streamlit(height=500, width=500)

    with col_2:
        with st.form(" "):
            modelKey = st.selectbox(label="Select Supervised Classifier",
                                       options=list(models.keys()),
                                       on_change=None)

            geoFile = st.file_uploader(label="Upload a geometry file",
                                       type=["geojson", "kml", "zip"],
                                       accept_multiple_files=False)

            start = st.date_input(label="Fire Start Date",
                                  value=date.fromisoformat("2021-08-10"),
                                  min_value=date.fromisoformat("2013-03-19"),
                                  max_value=date.today() + timedelta(weeks=-2),
                                  help="Select the starting date of a fire. (Must be a valid date for Landsat 8 images) \
                                  \n For more information on Landsat 8: \
                                  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#description")

            end = st.date_input(label="Fire End Date",
                                value=date.fromisoformat("2021-09-10"),
                                  min_value=date.fromisoformat("2013-03-19"),
                                  max_value=date.today() + timedelta(weeks=-2),
                                  help="Select the end date of a fire. (Must be a valid date for Landsat 8 images) \
                                  \n For more information on Landsat 8: \
                                  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#description")

            mapFireSubmit = st.form_submit_button("Map Fire")

else:
    if "widgetState" in st.session_state:
        del st.session_state["widgetState"]
    if "currentState" in st.session_state:
        del st.session_state["currentState"]

    if "idLst" not in st.session_state:
        st.session_state["idLst"] = [0]          # track changes in selected fire to avoid requerying data
    if "currentIndex" not in st.session_state:
        st.session_state["currentIndex"] = 0     # tracks current fire ID's position in session state


    with st.container():
        st.write("### Filter Fires")
        col_3, emptyCol_1, col_4 = st.columns([5, 1, 5])
        col_5, emptyCol_2, col_6 = st.columns([5, 1, 5])

        with col_3:
            startYear, endYear = st.select_slider(label="Year",
                                                  options=[i for i in range(2013, 2022)],
                                                  value=[2013, 2021],
                                                  on_change=None)
        with col_4:
            endMonths = st.multiselect(label="Fire Containment Month",
                                       options=list(monthMap.keys()),
                                       format_func=lambda x: monthMap[x],
                                       default=[8, 9, 10, 11],
                                       help="Month that a fire is contained/extinguished.\
                                       \n Recommended months are June-October.")
        with col_5:
            counties = st.multiselect(label="County",
                                      options=sorted(df["County"].unique()),
                                      default=["Humboldt", "Lassen", "Mendocino",
                                               "Napa", "Shasta", "Sonoma"],
                                      on_change=None,
                                      help="Counties in Northern California")

        with col_6:
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
        col_7, emptyCol_3, col_8 = st.columns([5, 1, 5])
        selectBoxOptions = formatFireSelectBox(dfSubset)

        fireID = col_7.selectbox(label="Select Fire to Map",
                                 options=list(selectBoxOptions.keys()),
                                 format_func=lambda x: selectBoxOptions[x])

        modelKey = col_8.selectbox(label="Select Supervised Classifier",
                                   options=list(models.keys()),
                                   on_change=None)

        mapFireSubmit = st.form_submit_button("Map Fire")

##############################
# Shared code
##############################

if mapFireSubmit:
    startTime = time.time()
    model = models[modelKey]
    tempMessage = st.empty()

    if not manual:
        fireData = dfSubset[dfSubset["ID"]==fireID]
        fireBounds = list(fireData["geometry"].bounds.values[0])
        if list(fireData["Contained Month"])[0] in [11,12,1,2]:
            st.warning("##### Snow and seasonal changes in vegetation can produce \
            inaccurate/skewed results for winter fires.")

        # Tracks if fireID has changed. If not, data will be accessed from previous session state
        idLst, currentIndex = updateIdState(fireID)

        if idLst[currentIndex-1] != idLst[currentIndex]:
            tempMessage.write("#### Querying data.....")
            for i in os.listdir():
                if os.path.splitext(i)[1] in [".tif", ".csv", ".xml", ".png", ".parquet"]:
                    os.remove(i)

            preFireL8, postFireL8, combined, fireGeometry = prepImages(geometry=fireData["geometry"],
                                                                       startDate=fireData["Start"].values[0],
                                                                       endDate=fireData["End"].values[0])
            downloadRaster([30, 60, 90, 120, 150, 180], combined, fireGeometry)
        else:
            preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeObjects"]

    if manual:
        # checks validity of input widgets
        if geoFile is None:
            st.error("### Must upload a geometry file")
            st.stop()
        elif end-start < timedelta(days=0):
            st.error("### Select a valid, non-overlapping date interval")
            st.stop()
        elif not (date.fromisoformat("2013-03-19") < start < date.today() + timedelta(weeks=-2)):
            st.error("Selected Fire Start Date is invalid. Refer to Landsat 8 image availability here: \
            https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#description")
            st.stop()
        elif not (date.fromisoformat("2013-03-19") < end < date.today() + timedelta(weeks=-2)):
            st.error("Selected Fire End Date is invalid. Refer to Landsat 8 image availability here: \
            https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#description")
            st.stop()

        else:
            if end.month in [11,12,1,2]:
                st.warning("##### Snow and seasonal changes in vegetation can produce \
                            inaccurate/skewed results for winter fires.")

            widgetStates, currentState = updateWidgetState([geoFile, start, end])

            # converts uploaded file to GeoDataFrame and fits bounding box over geometry
            gdf = uploaded_file_to_gdf(geoFile)
            gdf["geometry"] = gdf["geometry"].apply(lambda x: bbox(x.bounds))
            fireBounds = list(gdf["geometry"].bounds.values[0])

            # Requeries data if file upload or date widgets change between runs
            if sum(widgetStates[currentState-1] == widgetStates[currentState]) != 3:
                tempMessage.write("#### Querying data.....")
                for i in os.listdir():
                    if os.path.splitext(i)[1] in [".tif", ".csv", ".xml", ".png", ".parquet"]:
                        os.remove(i)

                preFireL8, postFireL8, combined, fireGeometry = prepImages(geometry=gdf,
                                                                           startDate=start,
                                                                           endDate=end)

                downloadRaster([30, 60, 90, 120, 150, 180], combined, fireGeometry)
            else:
                preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeObjects"]


    with st.container():
        tempMessage.write("#### Running model and rendering map.....")

        df = pd.read_parquet("raster.parquet")
        modelData = prepData(df[['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                                 'NDVI','elevation', 'percent_tree_cover', 'landCover']])

        labels, predictions = df["burnSeverity"].values, model.predict(modelData)

        # scale oscar models by +1
        if modelKey in ["log_boost", "SVM", "Boosted Trees"]:
            predictions = predictions + 1

        # png of actual+predicted burn severity from raster data
        burnSeverityImage(labels, st.session_state["rasterDims"], "thresholdBS.png")
        burnSeverityImage(predictions, st.session_state["rasterDims"], "predictedBS.png")

        # initialize geemap.foliumMap adds legend + image layers
        m = fmap.Map(add_google_map=False, plugin_LatLngPopup=False)
        add_legend(map=m,
                   legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
                                        ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))

        m.addLayer(preFireL8, l8_432, "Pre-Fire (RGB)")
        m.addLayer(postFireL8, l8_432, "Post-Fire (RGB)")

        m.addLayer(preFireL8, l8_753, "Pre-Fire (False-Color)")
        m.addLayer(postFireL8, l8_753, "Post-Fire (False-Color)")

        m.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")

        # adds burn severity png's as folium layers
        png_1 = folium.raster_layers.ImageOverlay(name="Thresholded Burn Severity",
                                                  image="thresholdBS.png",
                                                  bounds=[fireBounds[:2][::-1],
                                                          fireBounds[2:][::-1]],
                                                  interactive=True)
        png_1.add_to(m)

        png_2 = folium.raster_layers.ImageOverlay(name="Predicted Burn Severity",
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

        emptyCol_4, col_9, emptyCol_5 = st.columns([1,3.75,1])
        with col_9:
            # st.write("#### {} Accuracy: {}%".format(modelKey, np.round(100*np.mean(labels==predictions), 2)))
            m.to_streamlit(height=670, width=600, scrolling=True)

            with st.expander("View model metrics"):
                st.write(cm.style.set_properties(**{'text-align': 'center'}).to_html(),
                         unsafe_allow_html=True)
                st.write(metrics.to_html(),
                         unsafe_allow_html=True)
                st.altair_chart(lcChart)


    st.success("#### Total Runtime: {} seconds".format(np.round((time.time()-startTime), 2)))

# App footer
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
