# Robust Ecosystem Water and Energy Fluxes
This repository supports creating a robust evapotranspiration and latent energy flux estimation using a variety of eddy covariance towers. Different naming conventions, instrumentation configurations, and missing data create challenges to close the energy balance and fully leverage globally distributed eddy covariance tower observations. The methods employed here produce a range in water and energy flux estimates to support the evaluation of models with on-ground observations.

This section of code was used to produce a range of estimates of latent heat to compare with ECOSTRESS PT-JPL Estimates.  See Fisher, J.B., Lee, B., <b>Purdy, A.J.,</b> et al. ECOSTRESS: NASA's Next Generation Mission to Measure Evapotranspiration From the International Space Station. https://doi.org/10.1029/2019WR026058
for details on eddy covariance flux estimates. 

## Get set up

Tested under: Python 3.7.4 :: Anaconda custom (64-bit)
Last updated: 2020-06-08

### Installation using Anaconda Environment
Download [REWEF.yml](https://github.com/sciencebyAJ/Robust-Ecosystem-Water-and-Energy-Fluxes/blob/master/REWEF.yml)
```
$ conda env create -f REWEF.yml
$ conda activate myenv
```

### Running the code
* Upload your data to the /data/insitu folder as a csv file
* Edit the data/tower_var.csv file to reflect file naming convension for site
* Edit the insitu_subset list within run_rewef.py
* Run the run_rewef.py python script
```
$ python run_rewef.py
```
* Closed energy balance estimates will be saved to data/results as csv files

### Planned future updates

This repository will be updated to include scripts to select the observations surrounding tower locations.
