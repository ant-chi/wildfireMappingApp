## Supervised Classification Approach to Wildfire Mapping in Northern California

This **[Streamlit app](https://ant-chi-wildfiremappingapp-app-r4o1l0.streamlitapp.com/)** is developed as part of the senior data science capstone at UCSD, DSC180B. It demonstrates the use of supervised classification models for mapping wildfire burn severities. 

For details on how this app works, please watch this **[video](https://www.youtube.com/watch?v=NXZ4kPdbnyo)**.

Special thanks to Dr. Qiusheng Wu for developing the amazing **[geemap](https://geemap.org/)** package.

### About Wildfire Burn Severity
Burn severity describes the effect of wildfire on aboveground and belowground biomass. For example, common ecological measures of burn severity include canopy cover, tree crown volume, and soil hydrophobicity. Burn severity maps are widely used by federal agencies and forest managers to map fire damage, prioritize forest recovery efforts, land cover maps, and identify future environmental dangers.

Traditional methods of producing wildfire burn severity maps are dependent on sample data collected by teams of surveyors and ecologists. However this is very time-consuming, expensive, and often not useful for fires that occur in regions with harsh terrain and weather. 
This approach is still occassionally used, but has been largely phased out with the introduction of remotely sensed data from Earth observing satellites. Fires can be mapped at a much faster and larger scale and at a fraction of the cost of field surveys, while still maintaining high accuracy.

Normalized Burn Ratio (NBR) is a spectral index that is currently used by government agencies to map wildfires. It is sensitive to live vegetation and moisture content in soil and is used to identify burned areas. Taking the difference in NBR before and after a fire provides a measure of change and this is known as differenced NBR (dNBR). From here analysts determine dNBR thresholds that correspond to different burn severities to produce a final burn severity map. 

### Problems With dNBR Thresholding

The first problem with dNBR thresholding is that it can produce very inconsistent results because the thresholds are set subjectively by analysts. Because these thresholds are not validated with field data or ecologically quantified, different agencies often produce conflicting burn severity maps. 

The second is that because NBR is a spectral index that only uses two bands, near infrared (NIR) and shortwave infrared (SWIR), data from other spectral bands are not used at all even though they might contain valuable information. Relevant environmental data, such as air humidity, land cover, or terrain, is not considered even though they are key factors to the behavior and severity of wildfires.

The third problem is that due to the significant human involvement it takes to generate a burn severity map with dNBR thresholding, there are many wildfires that go unmapped every year. This limits the amount of research that could be done on wildfires and makes it harder to manage forest recovery efforts and identify potential environmental dangers (eg: increased risk of flash flooding due to charred soil)


### Benefits of ML Based Approach

Using an ML based approach to wildfire mapping addresses the problems inherent with dNBR thresholding. It largely removes human subjectivity from the process and would produce consistent results. Feedback from ecologists and wildfire experts can be used to engineer features and models that are accurate, understandable, and scientifically supported.
ML models can be trained on additional spectral and environmental data, which could provide valuable insights because they perform well at modeling complex relationships and benefit from large, high-dimensional datasets. Burn severity maps can also be produced much faster and at a larger scale.
Finally, this approach would greatly reduce the cost of producing burn severity maps.

