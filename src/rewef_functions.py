"""
This module contains a robust array of energy balance closure techniques for eddy covariance observations.
The focused use of this code is to support broad ingestion of Latent Energy observations for use in the evaluation of
satellite driven evapotranspiration estimates from ECOSTRESS.

This code was developed with contributions from A.J. Purdy, Brian Lee, Matt Dohlen & Joshua Fisher for the ECOSTRES validation analysis

See Fisher et al., 2020 for more details
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026058

"""

__author__ = 'A.J. Purdy, Brian Lee, and Matt Dohlen'

import numpy as np
import pandas as pd
from pandas.tseries import offsets


def force_close(insitu_df):
    """
    Energy Balance Forced Closure according to Twine et al., 2000 with Bowen Ratio Preservation.
        Twine et al., 2000. Correcting eddy-covariance flux underestimates over a grassland. AFM 103: 279-300
        See link to article here: https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1010&context=nasapub
    Ground heat flux is not required and removed from consideration (@ less than 20% of record) to maximize record.
    INPUT DATA: insitu_df # pandas dataframe with columns insitu_Rn, insitu_GHF, insitu_LE, & insitu_SHF
    OUTPUT DATA: closure_ratio # tower closure ratio or closure percentage
    Parameters:
    	insitu_df
    Returns:
    	closure_ratio
    """
    if np.count_nonzero(~np.isnan(insitu_df['insitu_GHF'])) == 0:
        x = np.array(insitu_df['insitu_Rn'])
    elif (np.isnan(insitu_df['insitu_GHF']).sum() / len(insitu_df)) > 0.2:
        x = np.array(insitu_df['insitu_Rn'])
    else:
        x = np.array(insitu_df['insitu_Rn']) - np.array(insitu_df['insitu_GHF'])
    y = np.array(insitu_df['insitu_LE']) + np.array(insitu_df['insitu_SHF'])
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask][np.newaxis]
    y = y[mask]
    closure_ratio = np.linalg.lstsq(x.T, y, rcond=None)

    return np.round(closure_ratio[0][0], 2)


def close(insitu_df, ebc_l=0.5, ebc_u=1.5):
    """
    Applying the closure ratio to close energy balance across varying time intervals including: half-hourly (instantaneous), daily, and annually
    INPUT DATA:
                insitu_df: a dataframe including columns for:
                    Net Radiation (W/m2) labeled 'insitu_Rn'; Ground Heat Flux  (W/m2) labeled 'insitu_GHF';
    	            Latent Energy (W/m2) labeled 'insitu_LE'; Sensible Heat (W/m2) labeled 'insitu_SHF'
    	        ebc_l:  lower threshold limit set to 0.5 as default
    	        ebc_u:  upper threshold limit set to 1.5 as default
    OUTPUT DATA:
                insitu_df: the same input dataframe with added columns of multiple energy balance closure methods (30 min, daily, annual)
    Parameters:
        insitu_df;
        ebc_l;
        ebc_u;
    Returns:
        insitu_df;
    """

    # Instantaneous tower closure
    insitu_df['insitu_fc'] = insitu_df.apply(force_close, axis=1)
    # Filtering for instantaneous closure beyond 50% and 150%
    EBC_lower_thresh = ebc_l
    EBC_upper_thresh = ebc_u
    insitu_df.loc[(insitu_df['insitu_fc'] < EBC_lower_thresh) | (insitu_df['insitu_fc'] > EBC_upper_thresh),'insitu_fc'] = np.nan
    insitu_df.loc[np.isnan(insitu_df['insitu_fc']), 'insitu_LE'] = np.nan

    # The colosure ratio is computed from the forced closure
    insitu_df['insitu_cr'] = insitu_df['insitu_fc']

    # LE and SHF are divided by cr to compute the closed energy balance while preserving the Bowen Ratio
    insitu_df['insitu_LE_fc_inst'] = insitu_df.insitu_LE / insitu_df['insitu_cr']
    insitu_df.loc[insitu_df['insitu_LE_fc_inst'] < 0, 'insitu_LE_fc_inst'] = np.nan
    insitu_df['insitu_SHF_fc_inst'] = insitu_df.insitu_SHF / insitu_df['insitu_cr']
    insitu_df.loc[insitu_df['insitu_SHF_fc_inst'] < 0, 'insitu_SHF_fc_inst'] = np.nan

    # Daily tower closure
    # LE and SHF are now closed by the daily tower closure following same steps as above
    daily_df = pd.DataFrame()
    # Compute daily closure ratio using each date
    daily_df['insitu_fc_day'] = insitu_df.groupby(insitu_df.index.date).apply(force_close)
    daily_df.index = pd.to_datetime(daily_df.index)
    # Creating a column with daily closure ratio at each half hour time-step
    df_30 = daily_df.resample('30min').asfreq()
    df_30 = df_30.fillna(method='ffill')
    insitu_df['insitu_fc_day'] = df_30['insitu_fc_day']
    # Filtering for daily closure beyond 50% and 150%
    insitu_df.loc[(insitu_df['insitu_fc_day'] < EBC_lower_thresh) | (insitu_df['insitu_fc_day'] > EBC_upper_thresh),
                  'insitu_fc_day'] = np.nan
    insitu_df['insitu_LE_fc_day'] = insitu_df['insitu_LE'] / insitu_df['insitu_fc_day']
    insitu_df.loc[insitu_df['insitu_LE_fc_day'] < 0, 'insitu_LE_fc_day'] = np.nan
    insitu_df['insitu_SHF_fc_day'] = insitu_df['insitu_SHF'] / insitu_df['insitu_fc_day']
    insitu_df.loc[insitu_df['insitu_SHF_fc_day'] < 0, 'insitu_SHF_fc_day'] = np.nan

    # Annual tower closure
    # LE and SHF are now closed by the annual (or multi-annual) tower closure following same steps as above
    # Compute annual closure ratio
    closure_ratio = force_close(insitu_df)
    # Apply annual closure ratio at each half hour time step
    insitu_df['insitu_LE_fc_ann'] = insitu_df['insitu_LE'] / closure_ratio
    insitu_df['insitu_SHF_fc_ann'] = insitu_df['insitu_SHF'] / closure_ratio

    # Compare to instantaneous Rn. If greater than Rn then set to nan.
    insitu_df.loc[insitu_df['insitu_LE_fc_inst'] > insitu_df['insitu_Rn'], 'insitu_LE_fc_inst'] = np.nan
    insitu_df.loc[insitu_df['insitu_LE_fc_day'] > insitu_df['insitu_Rn'], 'insitu_LE_fc_day'] = np.nan
    insitu_df.loc[insitu_df['insitu_LE_fc_ann'] > insitu_df['insitu_Rn'], 'insitu_LE_fc_ann'] = np.nan


    return insitu_df


def close_fluxnet(insitu_df, offset=15):
    """
    This function applies the fluxnet methodology of closing EBC_CF Method 1 using a moving window of +/- 15 days
    Conservatively we set the default to include the +/- 15 day windows.
    Correction factors (closure ratios) outside of 1.5 x the 25th percentile and 75th percentiles are filtered
    This parameter can be changed to emulate EBC_CF Method 2 or EBCF Method 3 with minor changes to the time window
    See: https://fluxnet.fluxdata.org/data/fluxnet2015-dataset/data-processing/

    INPUT:
       	insitu_df | is a dataframe including columns
    	    # Rn (W/m2) labeled 'insitu_Rn'; G  (W/m2) labeled 'insitu_GHF';
    	    # LE (W/m2) labeled 'insitu_LE'; H (W/m2) labeled 'insitu_SHF'
    	offset | is the moving window range in units of days

    OUTPUT:
        insitu_df | added columns for EBC_CF energy balance at 50th percentile, 25th percentile, and 75th percentile
    Parameters:
    	insitu_df
    	offset
    Returns:
    	insitu_df

    """
    try:
        flag_day_lim = [] 
        insitu_cr_25 = []
        insitu_cr_50 = []
        insitu_cr_75 = []

        delta = offsets.Day(offset)
        
        for t0 in insitu_df.index:
            ss_1 = insitu_df[t0-delta:t0+delta]
            hour = ss_1.index.hour
            selector = (22 < hour) | (hour < 3) | ((10 <= hour) & (hour < 15))
            ss_2 = ss_1[selector]
            if (ss_2['insitu_GHF'].isna().sum() / len(ss_2.index)) > 0.2:
                cr_ss2 = ss_2.insitu_Rn / (ss_2.insitu_SHF + ss_2.insitu_LE)
            else:
                cr_ss2 = (ss_2.insitu_Rn - ss_2.insitu_GHF)/(ss_2.insitu_SHF + ss_2.insitu_LE)
            ds_sort = sorted(cr_ss2)
            q1, q3 = np.percentile(ds_sort, [25, 75])
            iqr = q3-q1
            # computing thresholds for closure quality filtering
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            # filtering data
            cr_ss2[cr_ss2 > upper_bound] = np.nan
            cr_ss2[cr_ss2 < lower_bound] = np.nan
            cr_ss2 = cr_ss2[~np.isnan(cr_ss2)]
            ds_sort2 = sorted(cr_ss2)
            # computing clorure ratios
            cr_q1, cr_q3 = np.percentile(ds_sort2, [25, 75])
            cr_med = np.percentile(ds_sort2, [50])[0]
            insitu_cr_25.append(cr_q1)
            insitu_cr_50.append(cr_med)
            insitu_cr_75.append(cr_q3)
            inst_flag = len(cr_ss2) < 100  # less than 5 days of data points
            flag_day_lim.append(inst_flag)

        insitu_df['insitu_cr25'] = np.array(insitu_cr_25)
        insitu_df['insitu_cr50'] = np.array(insitu_cr_50)
        insitu_df['insitu_cr75'] = np.array(insitu_cr_75)

        EBC_lower_thresh = 0.5  # <-- These thresholds can be implemented on top of the FLUXNET2015 processing
        EBC_upper_thresh = 1.5  # <-- These thresholds can be implemented on top of the FLUXNET2015 processing
        insitu_df.loc[(insitu_df['insitu_cr50'] > EBC_upper_thresh or
             insitu_df['insitu_cr50'] < EBC_lower_thresh),'insitu_cr50'] = np.nan

        insitu_df.loc[np.isnan(insitu_df.insitu_cr50), 'insitu_cr25'] = np.nan
        insitu_df.loc[np.isnan(insitu_df.insitu_cr50), 'insitu_cr75'] = np.nan

        # Correcting LE and SHF observations with closure ratios
        # filtering data for unrealistic fluxes less than 0
        insitu_df['insitu_LE_flux50'] = insitu_df.insitu_LE_raw * insitu_df.insitu_cr50
        insitu_df.loc[insitu_df['insitu_LE_flux50'] < 0, 'insitu_LE_flux50'] = np.nan
        insitu_df['insitu_LE_flux25'] = insitu_df.insitu_LE_raw * insitu_df.insitu_cr25
        insitu_df.loc[insitu_df['insitu_LE_flux50'] < 0, 'insitu_LE_flux25'] = np.nan
        insitu_df['insitu_LE_flux75'] = insitu_df.insitu_LE_raw * insitu_df.insitu_cr75
        insitu_df.loc[insitu_df['insitu_LE_flux50'] < 0, 'insitu_LE_flux75'] = np.nan

        insitu_df['insitu_SHF_flux50'] = insitu_df.insitu_SHF * insitu_df.insitu_cr50
        insitu_df.loc[insitu_df['insitu_SHF_flux50'] < 0, 'insitu_SHF_flux_fc'] = np.nan
        insitu_df['insitu_H_flux25'] = insitu_df.insitu_SHF * insitu_df.insitu_cr25
        insitu_df.loc[insitu_df['insitu_SHF_flux50'] < 0, 'insitu_H_flux25'] = np.nan
        insitu_df['insitu_H_flux75'] = insitu_df.insitu_SHF * insitu_df.insitu_cr75
        insitu_df.loc[insitu_df['insitu_SHF_flux50'] < 0, 'insitu_H_flux75'] = np.nan

        # Creating a flag when there are less than 5 days of data
        insitu_df['ebc_1_flag'] = flag_day_lim
        
    except:
        # If the tower data cannot be closed due to missing data, we apply a range of artificial closure rates
        # These are only applied in the cases when net radiation or sensible heat flux observations are not available
        insitu_df['insitu_LE_1.1'] = insitu_df['insitu_LE_raw'].apply(lambda x: x * 1.1)
        insitu_df['insitu_LE_1.3'] = insitu_df['insitu_LE_raw'].apply(lambda x: x * 1.3)
        insitu_df['insitu_LE_1.5'] = insitu_df['insitu_LE_raw'].apply(lambda x: x * 1.5)

    return insitu_df
