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

from funcs import *


########## App starts here ##########

st.set_page_config(layout="wide", page_title="INSERT TITLE", page_icon=":earth_americas:")

# initialize EE + load and cache data
geemap.ee_initialize()

df = loadData()
models = loadModels()

# initialize session states
if "idLst" not in st.session_state:
    st.session_state["idLst"] = [42033]      # stores fireID and intialized with ID for Abney (default)
if "currentIndex" not in st.session_state:
    st.session_state["currentIndex"] = 0     # initializes index of current fireID
if "eeObjects" not in st.session_state:
    st.session_state["eeObjects"] = None      # stores necessary EE objects when data is queried

# Viz params
l8_viz = {"bands": ["SR_B7", "SR_B5", "SR_B3"],
          "gamma": [1.1, 1.1, 1],
          "min": 1000, "max": 25000}

burn_viz = {"bands": ["burnSeverity"],
            "palette": ["706c1e", "4e9d5c", "fff70b", "ff641b", "a41fd6"],
            "min": 1, "max": 5}

nlcd_viz = {"bands": ["landCoverViz"],
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
    # update options
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
    # col_3, col_4 = st.columns(2)
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

    modelKey = col_6.selectbox(label="Select Supervised Classifier",
                               options=list(models.keys()),
                               on_change=None)

    mapFireSubmit = st.form_submit_button("Map Fire")


if mapFireSubmit:
    startTime = time.time()
    model = models[modelKey]

    # Tracks if fireID has changed. If not, data will be accessed from previous session state
    idLst, currentIndex = updateIdState(fireID)
    tempMessage = st.empty()

    if idLst[currentIndex-1] != idLst[currentIndex] or len(idLst)==2:
        tempMessage.write("#### Querying data...")

        for i in os.listdir():
            if os.path.splitext(i)[1] in [".tif", ".csv", ".xml"]:
                os.remove(i)

        # st.write(os.listdir())
        # preFireL8, postFireL8, combined, fireGeometry = prepData(dfSubset[dfSubset["ID"]==fireID])
        # st.session_state["eeObjects"] = [preFireL8, postFireL8, combined, fireGeometry]

        st.session_state["eeObjects"] = prepImages(dfSubset[dfSubset["ID"]==fireID])
        preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeObjects"]

        fileName = "{}.tif".format(fireID)

        # with st.container():
        #     tempMessage.empty()
        #
        #     m = fmap.Map(add_google_map=False)   # initialize folium Map
        #     add_legend(map=m,
        #                legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
        #                                     ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))
        #
        #     m.addLayer(preFireL8, l8_viz, "Pre-Fire L8")
        #     m.addLayer(postFireL8, l8_viz, "Post-Fire L8")
        #
        #     m.addLayer(combined.clip(fireGeometry), burn_viz, "Burn Severity")
        #     m.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")
        #
        #     lon, lat = fireGeometry.centroid().getInfo()["coordinates"]
        #     m.setCenter(lon, lat, zoom=10)
        #     m.add_layer_control()
        #
        #     emptyCol_3, col_7, emptyCol_4 = st.columns([1,3.5,1])
        #     with col_7:
        #         m.to_streamlit(height=700, width=600, scrolling=True)
        # st.write("### 1")
        loadRaster([30, 60, 90, 120, 150], fileName, combined, fireGeometry)
        # st.write(sorted(os.listdir()))
        # files = []
        # for r, d, f in os.walk(os.getcwd()):
        #     for file in f:
        #         # if '.tif' in file:
        #         files.append(os.path.join(r, file))
        #
        # for f in files:
        #     st.write(f)

        # shutil.move(fileName, os.path.join("rasters", fileName))
        # st.write(os.listdir("rasters"), os.listdir())
        # st.write(os.listdir())
        # st.write("## export image")
        # geemap.ee_export_image(ee_object=combined,
        #                        filename=fileName,
        #                        scale=30,
        #                        region=fireGeometry)
        # st.write("### 2")
        rasterToCsv(fileName)
        # st.write(sorted(os.listdir()))
        # st.write(os.listdir("rasters"), os.listdir())

        # files = []
        # for r, d, f in os.walk(os.getcwd()):
        #     for file in f:
        #         # if '.tif' in file:
        #         files.append(os.path.join(r, file))
        #
        # for f in files:
        #     st.write(f)


    else: # access session_state variables
        preFireL8, postFireL8, combined, fireGeometry = st.session_state["eeObjects"]


    with st.container():
        tempMessage.empty()
        df = pd.read_csv("{}.csv".format(fireID))

        labels = df["burnSeverity"]
        modelData = prepData(df[['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                                 'NDVI','elevation', 'percent_tree_cover', 'landCover']])

        df["Prediction"] = model.predict(modelData)
        confusionMatrix, df_2 = modelMetrics(df)


        m = fmap.Map(add_google_map=False)   # initialize geemap.foliumMap
        add_legend(map=m,
                   legend_dict=dict(zip(["Burn Severity"]+["Vegetation Growth", "Unburned", "Low", "Moderate", "High"]+["Land Cover"]+["Other", "Developed", "Forest", "Shrub", "Grassland", "Agriculture"],
                                        ["None"]+burn_viz["palette"]+["None"]+nlcd_viz["palette"])))

        m.addLayer(preFireL8, l8_viz, "Pre-Fire L8")
        m.addLayer(postFireL8, l8_viz, "Post-Fire L8")

        m.addLayer(combined.clip(fireGeometry), burn_viz, "Burn Severity")
        m.addLayer(combined.clip(fireGeometry), nlcd_viz, "Land Cover")

        m.add_local_tile(source="{}.tif".format(fireID),
                          band=8,
                          palette="Reds",
                          vmin=1,   # comment out to show entire raster
                          vmax=5,
                          nodata=0,
                          layer_name="Local Raster")


        lon, lat = fireGeometry.centroid().getInfo()["coordinates"]
        m.setCenter(lon, lat, zoom=10)
        m.add_layer_control()
        chart_1, chart_2 = altChart(df)

        emptyCol_3, col_7, emptyCol_4 = st.columns([1,3.5,1])
        with col_7:
            m.to_streamlit(height=700, width=600, scrolling=True)


        st.write("#### Accuracy: {}%".format(np.round(100*np.mean(df["burnSeverity"]==df["Prediction"]), 2)))

        st.write(confusionMatrix.to_markdown())
        st.write(df_2.to_markdown())

        st.altair_chart(chart_1)
        st.altair_chart(chart_2)


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
        # | Class | Precision | Recall | Accuracy |
        # | --- | --- | --- | --- |
        # | **Vegetation Growth** | blah | blah | blah |
        # | **Unburned** | blah | blah | blah |
        # | **Low** | blah | blah | blah |
        # | **Moderate** | blah | blah | blah |
        # | **High** | blah | blah | blah |
        # """
        # )

    st.write("Total Runtime: {} seconds".format(np.round((time.time()-startTime), 2)))


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
