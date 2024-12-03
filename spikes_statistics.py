import pandas as pd
from matplotlib import pyplot as plt
import glob
import numpy as np
import optics_rig as orig
from scipy.interpolate import interp1d
from scipy import signal as signal
from scipy.stats import gamma, burr12

from iteration_utilities import flatten

import os
import glob
from scipy import optimize
import copy
import dill

import colorcet as cc

from scipy.integrate import trapezoid as trapz


  
def get_spikes(rbr_smpl, METHOD='Nathan'):
    ## Estimate spikes by removing median +/- minimum value

    half_noise = np.median(rbr_smpl) - np.min(rbr_smpl)


    if METHOD=='Nathan':
        
        rbr_spikes = rbr_smpl - np.median(rbr_smpl) 
        rbr_spikes[np.where(rbr_spikes < half_noise)[0]] = np.nan
        
    else:
        
        rbr_spikes = rbr_smpl - np.median(rbr_smpl) - half_noise
        rbr_spikes[np.where(rbr_spikes < 0)[0]] = np.nan
            
    return rbr_spikes



def get_spike_prctiles(rbr_):

    rbr_spikes = get_spikes(rbr_)
    

    # extract percentiles
    prct_pos = np.asarray([5, 16, 50, 84, 95, 100])
    prct_vals = np.percentile(rbr_spikes, prct_pos)
    
    return prct_vals



def create_bin_index(rbr, dt):
    
    rbr['bin'] = np.floor(rbr.index.values/(1000*dt))
    
    return rbr



def with_pandas_groupby(func, x, b):
    
    grouped = pd.Series(x).groupby(b)
    
    return grouped #.agg(func)




def function_hist(data, bins):
    
    weightsa = np.ones_like(data)#/float(len(a))
    hist = np.histogram(np.array(data), bins, weights = weightsa)
    
    return hist



def get_binned_BB(rbr, time_bin, rbr_binned, bins, METHOD="Nathan"):
    # extract data for this bin
    rbr_smpl = rbr.loc[rbr['bin']==time_bin]

    ## Extract baseline

    rbr_binned.loc[time_bin, 'baseline'] = np.nanpercentile(rbr_smpl['BB700'].values, 50)
    # rbr_baseline

    ## Extract spikes and log-transform them
    rbr_spikes = get_spikes(rbr_smpl['BB700'].values, METHOD=METHOD);
    rbr_spikes_log = np.log10(rbr_spikes)

    ## Extract log-spike percentiles
    prct_pos = np.asarray([1, 16, 50, 84, 100])
    rbr_binned.iloc[time_bin, 2:7] = np.nanpercentile(rbr_spikes_log, prct_pos)

    ## Compute how many spikes
    # compute how many spikes after removing nans
    rbr_binned.loc[time_bin, 'N_spikes'] = len(rbr_spikes_log[~np.isnan(rbr_spikes_log)])

    # Extract histogram
    tmp = function_hist(rbr_spikes_log, np.insert(bins, 0, -7))
    rbr_binned.iloc[time_bin, 8:] = tmp[0]
    
    return rbr_binned