import streamlit as st
import ee
import geemap
import numpy as np
import time
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
import os


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



def formatSelectBoxOptions(data):
    return {k: "{} ({})".format(v1, v2) for k, v1, v2 in data[["ID", "Fire", "Year"]].sort_values(by="Fire").values}



def subsetFires(data, startYear, endYear, sizeClass=None, counties=None):
    if startYear == endYear:
        st.error("### Select valid time interval")
    else:
        subset = data[(data["Year"] >= startYear) & (data["Year"] < endYear)]

    if len(sizeClass) == 0:
        st.warning("### Size Class is not selected (Filter will not be applied)")
    else:
        subset = subset[subset["Size Class"].isin(sizeClass)]

    if len(counties) == 0:
        st.warning("### County is not selected (Filter will not be applied)")
    else:
        subset = subset[subset["County"].isin(counties)]

    return subset.sort_values(by="Fire").reset_index(drop=True)



def prepImage(preFireImage, postFireImage, geometry, endDate):

    # Calculate NBR, dNBR, and burn severity
    preFireNBR = preFireImage.normalizedDifference(['SR_B5', 'SR_B7'])
    postFireNBR = postFireImage.normalizedDifference(['SR_B5', 'SR_B7'])
    dNBR = (preFireNBR.subtract(postFireNBR)).multiply(1000).rename("dNBR")

    # postFireDate = convertDate(postFireImage.date())
    burnSeverity = dNBR.expression(" (b('dNBR') > 425) ? 4 "    # purple: high severity
                                   ":(b('dNBR') > 225) ? 3 "    # orange: moderate severity
                                   ":(b('dNBR') > 100) ? 2 "    # yellow: low severity
                                   ":(b('dNBR') > -60) ? 1 "    # green: unburned/unchanged
                                   ":0"                         # brown: vegetation growth
                      ).rename("burnSeverity")

    # Get SRTM elevation, NLCD land coverpostFireImageNDVI, and GRIDMET weather
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2016_REL'
            ).filter(ee.Filter.eq('system:index', '2016')).first()

    lc = nlcd.select("landcover")
    nlcd = nlcd.select([i for i in range(1,13)])
    nlcd = nlcd.addBands(lc.expression(" (b('landcover') > 90) ? 0 "    # blue: other (wetland)
                                         ":(b('landcover') > 80) ? 5 "    # brown: agriculture
                                         ":(b('landcover') > 70) ? 4 "    # lightGreen: grassland/herbaceous
                                         ":(b('landcover') > 50) ? 3 "    # yellow: shrub
                                         ":(b('landcover') > 40) ? 2 "    # green: forest
                                         ":(b('landcover') > 30) ? 0 "    # blue: other (barren land)
                                         ":(b('landcover') > 20) ? 1 "    # red: developed/urban
                                         ":0"                             # blue: other (water/perennial ice+snow)
                            ).rename("landcover"))


    ndvi = postFireImage.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET"
               ).filterBounds(geometry
               ).filterDate(endDate.advance(-3, "day"), endDate
               ).mean()

    # Merge all image bands together
    combined = postFireImage.select('SR_B.'          # post-fire L8 bands 1-7
                           ).addBands(burnSeverity   # classified burn severity
                           ).addBands(dNBR           # dNBR
                           ).addBands(ndvi           # post-fire NDVI
                           ).addBands(dem            # SRTM elevation
                           ).addBands(gridmet        # all GRIDMET bands
                           ).addBands(nlcd           # all NLCD bands
                           )#.set("FIRE_NAME", fireName)
    return combined


def loadTif(numTries, imgScale, fireID, image, geometry):
    startTime = time.time()
    for i in range(numTries):
        try:
            geemap.ee_export_image(image, "tifs/{}.tif".format(fireID), scale=imgScale[i], region=geometry)
            st.success("#### Downloaded tif at {}m scale".format(imgScale[i]))
            break
        except Exception:
            if i == numTries-1:
                st.warning("### Fire exceeds total request size")
            else:
                st.write("#### Retrying at {}m scale".format(imgScale[i+1]))

    st.write("Runtime: {} minutes".format(np.round((time.time()-startTime)/60, 3)))





def add_legend(
    map,
    title="Legend",
    colors=None,
    labels=None,
    legend_dict=None,
    builtin_legend=None,
    opacity=1.0,
    **kwargs,
):
    """Adds a customized basemap to the map. Reference: https://bit.ly/3oV6vnH

    Args:
        title (str, optional): Title of the legend. Defaults to 'Legend'. Defaults to "Legend".
        colors ([type], optional): A list of legend colors. Defaults to None.
        labels ([type], optional): A list of legend labels. Defaults to None.
        legend_dict ([type], optional): A dictionary containing legend items as keys and color as values. If provided, legend_keys and legend_colors will be ignored. Defaults to None.
        builtin_legend ([type], optional): Name of the builtin legend to add to the map. Defaults to None.
        opacity (float, optional): The opacity of the legend. Defaults to 1.0.

    """

    # import pkg_resources
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
