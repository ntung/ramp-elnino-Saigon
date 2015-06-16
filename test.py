
# coding: utf-8

# # <a href="http://www.datascience-paris-saclay.fr">Paris Saclay Center for Data Science</a>
# #<a href=http://www.datascience-paris-saclay.fr/en/site/newsView/12>RAMP</a> on El Nino prediction
# 
# <i> Balázs Kégl (CNRS), Claire Monteleoni (GWU), Mahesh Mohan (GWU), Timothy DelSole (COLA), Kathleen Pegion (COLA), Julie Leloup (UPMC), Alex Gramfort (LTCI) </i>

# ## Introduction
# 
# A climate index is real-valued time-series which has been designated of interest in the climate literature. For example, the El Niño–Southern Oscillation (ENSO) index has widespread uses for predictions of regional and seasonal conditions, as it tends to have strong (positive or negative) correlation with a variety of weather conditions and <a href=http://www.ipcc-wg2.gov/SREX/images/uploads/SREX-SPMbrochure_FINAL.pdf>extreme events</a> throughout the globe. The ENSO index is just one of the many climate indices studied. However there is currently significant room for improvement in predicting even this extremely well studied index with such high global impact. For example, most statistical and climatological models erred significantly in their predictions of the 2015 El Niño event; their predictions were off by several months. Better tools to predict such indices are critical for seasonal and regional climate prediction, and would thus address grand challenges in the study of climate change (<a href=http://wcrp-climate.org/grand-challenges>World Climate Research Programme: Grand Challenges, 2013)</a>.
# 
# ### El Niño
# 
# <a href="https://www.ncdc.noaa.gov/teleconnections/enso/indicators/sst.php">El Niño</a> (La Niña) is a phenomenon in the equatorial Pacific Ocean characterized by a five consecutive 3-month running mean of sea surface temperature (SST) anomalies in the <a href=http://www1.ncdc.noaa.gov/pub/data/cmb/teleconnections/nino-regions.gif>Niño 3.4 region</a> that is above (below) the threshold of $+0.5^\circ$C ($-0.5\circ$C). This standard of measure is known as the Oceanic Niño Index (ONI).
# 
# <img src=http://www1.ncdc.noaa.gov/pub/data/cmb/teleconnections/nino-regions.gif>
# 
# Mor information can be found <a href=https://www.ncdc.noaa.gov/teleconnections/enso/indicators/sst.php>here</a> on why it is an important region, and what is the history of the index.
# 
# Here are the <a href = http://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/>current ENSO predictions</a>, updated monthly.
# 
# 
# ### The CCSM4 simulator
# 
# Our data is coming from the <a href=http://www.cesm.ucar.edu/models/ccsm4.0/>CCSM4.0</a> model (simulator). This allows us to access a full regular temperature map for a 500+ year period which makes the evaluation of the predictor more robust than if we used real measurements. 
# 
# ### The data
# 
# The data is a time series of "images" $z_t$, consisting of temperature measurements (for a technical reason it is not SST that we will work with, rather air temperature) on a regular grid on the Earth, indexed by lon(gitude) and lat(itude) coordinates. The average temperatures are recorded every month for 501 years, giving 6012 time points. The goal is to predict the temperature in the El Nino region, <span style="color:red">6 months ahead</span>.
# 
# ### The prediction task
# 
# Similarly to the variable stars RAMP, the pipeline will consists of a feature extractor and a predictor. Since the task is regression, the predictor will be a regressor, and the score (to minimize) will be the <a href=http://en.wikipedia.org/wiki/Root-mean-square_deviation>root mean square error</a>. The feature extractor will have access to the whole data. It will construct a "classical" feature matrix where each row corresponds to a time point. You should collect all information into these features that you find relevant to the regressor. The feature extractor can take <span style="color:red">anything from the past</span>, that is, it will implement a function $x_t = f(z_1, \ldots, z_t)$. Since you will have access to the full data, in theory you can cheat (even inadvertantly) by using information from the future. Please do your best to avoid this since it would make the results irrelevant.
# 
# ### Domain-knowledge suggestions
# 
# You are of course free to explore any regression technique to improve the prediction. Since the input dimension is relatively large (2000+ dimensions per time point even after subsampling) sparse regression techniques (eg. LASSO) may be the best way to go, but this is just an a priori suggestion. The following list provides you other hints to start with, based on domain knowledge. 
# <ul>
# <li>There is a strong seasonal cycle that must be taken into account.
# <li>There is little scientific/observational evidence that regions outside the Pacific play a role in NINO3.4 variability, so it is probably best to focus on Pacific SST for predictions.  
# <li>The relation between tropical and extra-tropical Pacific SST is very unclear, so please explore!
# <li>The NINO3.4 index has an oscillatory character (cold followed by warm followed by cold), but this pattern does not repeat exactly.  It would be useful to be able to predict periods when the oscillation is “strong” and when it “breaks down.”  
# <li>A common shortcoming of empirical predictions is that they under-predict the <i>amplitude</i> of warm and cold events.  Can this be improved?
# <li>There is evidence that the predictability is low when forecasts start in, or cross over, March and April (the so-called “spring barrier”). Improving predictions through the spring barrier would be important.
# <ul>

# # Exploratory data analysis

# Packages to install:
# 
# conda install basemap<BR>
# conda install -c https://conda.binstar.org/anaconda xray<BR>
# conda install netcdf4 h5py<BR>
# pip install pyresample<BR>

# In[68]:

#get_ipython().magic(u'matplotlib inline')
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xray # should be installed with pip
import pyresample  # should be installed with pip
from sklearn.cross_validation import cross_val_score


# Let's start by reading the data into an xray Dataset object. You can find all information on how to access and manipulate <code>Dataset</code> and <code>DataArray</code> objects at the <a href=http://xray.readthedocs.org/en/stable/>xray site</a>.

# In[2]:

temperatures_xray = xray.open_dataset(
    '../resampled_tas_Amon_CCSM4_piControl_r1i1p1_080001-130012.nc', decode_times=False)
#temperatures_xray = xray.open_dataset(
#    'COLA_data/tas_Amon_CCSM4_piControl_r2i1p1_095301-110812.nc', decode_times=False)
#temperatures_xray = xray.open_dataset(
#    'COLA_data/tas_Amon_CCSM4_piControl_r3i1p1_000101-012012.nc', decode_times=False)

# there is no way to convert a date starting with the year 800 into pd array so we 
# shift the starting date to 1700
temperatures_xray['time'] = pd.date_range('1/1/1700', 
                                          periods=temperatures_xray['time'].shape[0],
                                          freq='M') - np.timedelta64(15, 'D')


# Printing it, you can see that it contains all the data, indices, and other metadata.

# In[3]:

temperatures_xray


# The main data is in the 'tas' ("temperature at surface") DataArray.

# In[4]:

temperatures_xray['tas']


# You can index it in the same way as a <code>pandas</code> or <code>numpy</code> array. The result is always a <coda>DataArray</code>

# In[5]:

t = 123
lat = 13
lon = 29
temperatures_xray['tas'][t]
temperatures_xray['tas'][t, lat]
temperatures_xray['tas'][t, lat, lon]
temperatures_xray['tas'][:, lat, lon]
temperatures_xray['tas'][t, :, lon]
temperatures_xray['tas'][:, :, lon]


# You can convert any of these objects into a <code>numpy</code> array.

# In[6]:

temperatures_xray['tas'].values
temperatures_xray['tas'][t].values
temperatures_xray['tas'][t, lat].values
temperatures_xray['tas'][t, lat, lon].values


# You can also use slices, and slice bounds don't even have to be in the index arrays. The following function computes the target at time $t$. The input is an xray DataArray (3D panel) that contains the temperatures. We select the El Nino 3.4 region, and take the mean temperatures, specifying that we are taking the mean over the spatial (lat and lon) coordinates. The output is a vector with the same length as the original time series.

# In[7]:

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120

def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(dim=('lat','lon'))

def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region at all time points."""
    return get_area_mean(tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)


# The following function plots the temperatures at a given $t$ (time_index). 

# In[8]:

el_nino_lats = [en_lat_bottom, en_lat_top, en_lat_top, en_lat_bottom]
el_nino_lons = [en_lon_right, en_lon_right, en_lon_left, en_lon_left]

from matplotlib.patches import Polygon

def plot_map(temperatures_xray, time_index):
    def draw_screen_poly(lats, lons, m):
        x, y = m(lons, lats)
        xy = zip(x,y)
        poly = Polygon(xy, edgecolor='black', fill=False)
        plt.gca().add_patch(poly)

    lons, lats = np.meshgrid(temperatures_xray['lon'], temperatures_xray['lat'])

    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9,0.9])
    map = Basemap(llcrnrlon=0, llcrnrlat=-89, urcrnrlon=360, urcrnrlat=89, projection='mill')
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    #map.drawcountries(linewidth=0.25)
    #map.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    #map.drawmapboundary(fill_color='aqua')
    im = map.pcolormesh(lons, lats, temperatures_xray[time_index] - 273.15,
                        shading='flat', cmap=plt.cm.jet, latlon=True)
    cb = map.colorbar(im,"bottom", size="5%", pad="2%")
    draw_screen_poly(el_nino_lats, el_nino_lons, map)

    time_str = str(pd.to_datetime(str(temperatures_xray['time'].values[time_index])))[:7]
    ax.set_title("Temperature map " + time_str)
    #plt.savefig("test_plot.pdf")
    plt.show()


# Let's plot the temperature at a given time point. Feel free to change the time, play with the season, discover visually the variability of the temperature map.

# In[9]:

t = 12
plot_map(temperatures_xray['tas'], t)
