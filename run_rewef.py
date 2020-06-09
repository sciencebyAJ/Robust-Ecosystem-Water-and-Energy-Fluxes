"""
This module contains a robust array of energy balance closure techniques for eddy covariance observations.
The focused use of this code is to support broad ingestion of Latent Energy observations for use in the evaluation of
satellite driven evapotranspiration estimates from ECOSTRESS.


Before running this script make sure to:
    save tower data to the data/insitu folder as a csv file
    update the tower_var.csv file with tower naming conventions

Usage:  save this script and run
    $python run_rewef.py

The energy balance closed data will be saved to data/results as a csv file

See Fisher et al., 2020 for more details
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026058

Tested under: Python 3.7.4 :: Anaconda custom (64-bit)
Last updated: 2020-06-08

This code was developed with contributions from A.J. Purdy, Brian Lee, Matt Dohlen & Joshua Fisher for the ECOSTRESS validation analysis
If you have any questions, suggestions, or comments on this example, please email adamjpurdy@gmail.com

"""
__author__ = 'A.J. Purdy, PhD'

import src.rewef_functions as ebc

import pandas as pd
import os
import os.path
import numpy as np
import traceback as tb

PROJDIR = os.getcwd()
os.chdir('./data')
DATADIR = PROJDIR+'/data'
OUTDIR = PROJDIR+'/data/results'
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

def tower_var():
    tower_frame = pd.read_csv(os.path.join(DATADIR, 'tower_var.csv'), skiprows=[1], index_col =0, encoding='latin-1')
    tower_dict = tower_frame.to_dict(orient='index')
    return tower_dict,tower_frame

tower_dict, tower_df = tower_var()


def read_eco():
    '''
#     This function reads in ECOSTRESS data and renames column names to be matched with insitu flux tower data.
#     INPUT DATA:
#     -ECOSTRESS data over flux towers. This script has not been posted yet. We plan on sharing this part of the script in Version 2 of this repository.
#     OUTPUT DATA:
#     -cleaned ECOSTRESS dataframe
#     '''
    eco_df = pd.read_csv(os.path.join(DATADIR, 'processed_ecostress.csv'))
    eco_df['eco_datetime_solar'] = pd.to_datetime(eco_df['eco_datetime_solar'])
    eco_df.rename(columns={'datetime_solar': 'eco_datetime_solar',
                           'eco_LE': '5x5_eco_median',
                           'LE_eco_mean': '5x5_eco_mean',
                           'LE_eco_std': '5x5_eco_std',
                           'LE_nonans': '5x5_values',
                           'LE_nonans_inset': '3x3_values',
                           'LE_eco_mean_inset': '3x3_eco_mean',
                           'LE_eco_std_inset': '3x3_eco_std',
                           'LE_eco_iq1_inset': '3x3_eco_iq1',
                           'LE_eco_iq3_inset': '3x3_eco_iq3',
                           'LE_eco_iq1': '5x5_eco_iq1',
                           'LE_eco_iq3': '5x5_eco_iq3'}, inplace=True)

    # standardize the naming functions from sites
    eco_df['name'] = eco_df['name'].str.replace('-', '_').str.lower()
    eco_df = eco_df.set_index(pd.DatetimeIndex(eco_df['eco_datetime_solar']))

    # remove null values
    eco_df = eco_df[eco_df['5x5_eco_median'].notnull()]

    return eco_df


def read_insitu(site_id):
    '''
    This function reads in insitu flux tower data, formats column headers using lookup table, formats data type per column,
    filters out bad/flagged, and returns the cleaned dataframe
    INPUT DATA:
    -pre-processed insitu data (processed manually)
    -tower variable dictionary/look-up table
    OUTPUT DATA:
    -processed insitu data
    '''
    os.chdir(DATADIR)
    insitu_df = pd.read_csv(os.path.join(DATADIR, 'insitu/' + site_id + '.csv'), skiprows=[1])
    # Rename the columns using tower dictionary
    insitu_df.rename(columns={tower_dict[site_id]['TIME_START']: 'insitu_datetime_solar',
                              tower_dict[site_id]['Ground Heat Flux']: 'insitu_GHF',
                              tower_dict[site_id]['Ground Heat Flux Quality Flag']: 'insitu_GHF_qc',
                              tower_dict[site_id]['Net Radiation']: 'insitu_Rn',
                              tower_dict[site_id]['Net Radiation Quality Flag']: 'insitu_Rn_qc',
                              tower_dict[site_id]['Sensible Heat Flux']: 'insitu_SHF',
                              tower_dict[site_id]['Sensible Heat Flux Quality Flag']: 'insitu_SHF_qc',
                              tower_dict[site_id]['Latent Heat Flux']: 'insitu_LE',
                              tower_dict[site_id]['Latent Heat Flux Quality Flag']: 'insitu_LHF_qc',
                              tower_dict[site_id]['Incoming Shortwave Radiation']: 'insitu_RDS',
                              tower_dict[site_id]['Air Temperature above Canopy']: 'insitu_Ta',
                              tower_dict[site_id]['Precipitation']: 'precip'}, inplace=True)
    # formats the datetime before converting to datetime format
    try:
        insitu_df = insitu_df[np.isfinite(insitu_df['insitu_datetime_solar'])]
    except:
        pass
    # turns datetime from mystery formats (some sites) to integer for easier translation
    try:
        insitu_df['insitu_datetime_solar'] = insitu_df['insitu_datetime_solar'].astype(int)
    except:
        pass
    # formats the datetime integer to string
    insitu_df['insitu_datetime_solar'] = insitu_df['insitu_datetime_solar'].astype(str)
    insitu_df['insitu_datetime_solar'] = pd.to_datetime(insitu_df['insitu_datetime_solar'])
    # copy insitu datetime
    insitu_df['time'] = [str(d.time()) for d in insitu_df['insitu_datetime_solar']]
    insitu_df = insitu_df.set_index(pd.DatetimeIndex(insitu_df['insitu_datetime_solar']))
    # only select values we are interested in, starting in June 2018
    insitu_df = insitu_df.loc['2018-06-01':]
    # replace non values
    insitu_df.replace(-9999.0, np.NaN, inplace=True)
    # save the raw insitu LE value for future calculations
    insitu_df['insitu_LE_raw'] = insitu_df['insitu_LE']
    insitu_df['insitu_LE_raw'] = insitu_df['insitu_LE_raw'].astype(float)

    try:
        insitu_df = insitu_df[insitu_df['insitu_LE'].notnull()]
    except:
        pass
    try:
        insitu_df = insitu_df[insitu_df['insitu_SHF'].notnull()]
    except:
        pass

    return insitu_df


def run_site(site_id, eco_df):

    insitu_df = read_insitu(site_id)
    print('processing data for: '+site_id)
    try:
        insitu_df = ebc.close(insitu_df)
        print('\t-performed ebc')
    except:
        tb.print_exc()
        pass
    try:
        insitu_df = ebc.close_fluxnet(insitu_df)
        print('\t-performed fluxnet 2015 ebc')
    except:
        tb.print_exc()
        pass
    return insitu_df


eco_df = read_eco()

insitu_subset = ['au_asm','au_cum']

print('running rewef ebc closure for '+str(len(insitu_subset))+' sites')
for site_id in tower_df.index:
    if site_id in insitu_subset:
        try:
            site_df = eco_df.loc[eco_df['name'] == site_id]
            site_df = site_df.sort_index(axis=0)
            site_rewef_df = run_site(site_id, site_df)
            site_rewef_df.to_csv(os.path.join(OUTDIR, site_id+'_rewef_ebc_closed.csv'))
        except:
            tb.print_exc()
print('run_rewef.py script complete')
