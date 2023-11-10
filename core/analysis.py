# Imports
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt  # plotting library
from matplotlib import colors
import numpy as np  # work with numeric arrays without labeled axes
import xarray as xr  # work with arrays with labeled axes
import xrscipy.signal as dsp  # xarray signal filtering etc.
import scipy as sps
from cdb_extras import xarray_support as cdbxr  # access to COMPASS Database (CDB)
import pickle # to save data
from pathlib import Path # to easily work with different files
from progressbar import ProgressBar

# Import DataClass for condensed pedestal data

# Define a dataclass to hold the pedestal fit results
@dataclass
class PedestalParams:
    shot_number: int
    
    ne_height: np.ndarray
    ne_height_err: np.ndarray
    ne_grad: np.ndarray
    ne_grad_err: np.ndarray
    ne_width: np.ndarray
    ne_width_err: np.ndarray
    ne_time: np.ndarray
    ne_ELM_phase: np.ndarray
    ne_ELM_time: np.ndarray
    ne_ST_phase: np.ndarray
    ne_ST_time: np.ndarray
        
    Te_height: np.ndarray
    Te_height_err: np.ndarray
    Te_grad: np.ndarray
    Te_grad_err: np.ndarray
    Te_width: np.ndarray
    Te_width_err: np.ndarray
    Te_time: np.ndarray
    Te_ELM_phase: np.ndarray
    Te_ELM_time: np.ndarray
    Te_ST_phase: np.ndarray
    Te_ST_time: np.ndarray
        
    pe_height: np.ndarray
    pe_height_err: np.ndarray
    pe_grad: np.ndarray
    pe_grad_err: np.ndarray
    pe_width: np.ndarray
    pe_width_err: np.ndarray
    pe_time: np.ndarray
    pe_ELM_phase: np.ndarray
    pe_ELM_time: np.ndarray
    pe_ST_phase: np.ndarray
    pe_ST_time: np.ndarray


def combine_pedestalparams(pp_list):
    def extract_subset(name):
        return np.concatenate(tuple([getattr(pp, name).data for pp in pp_list]))
        
    toplist = ['pe', 'ne', 'Te']
    subset_list = ['height', 'height_err', 'grad', 'grad_err', 'width', 'width_err', 'time', 'ELM_phase', 'ST_phase', 'ST_time']
    
    d = {}
    for topname in toplist:
        subd = {}
        for subname in subset_list:
            name = f"{topname}_{subname}"
            concat_arr = extract_subset(name)
            subd[subname] = concat_arr
        subd[topname] = subd
    
    subd['shot_nr'] = np.concatenate(tuple([np.full(pp.shot_number, len(pp.ne_height.data)) for pp in pp_list]))
    return subd
    

def scatter_hist(x, y, ax, ax_histx, ax_histy, c='C0', cmap='jet'):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, c=c, alpha=0.4, cmap=cmap)

    # now determine nice limits by hand:
    x_max = np.nanmax(x)
    y_max = np.nanmax(y)
    x_min = np.nanmin(x)
    y_min = np.nanmin(y)

    x_bins = np.linspace(x_min, x_max, 60)
    y_bins = np.linspace(y_min, y_max, 60)
    
    if isinstance(c, np.ndarray):
        ax_histx.hist(x, bins=x_bins, color='C2')
        ax_histy.hist(y, bins=y_bins, orientation='horizontal', color='C2')
    else:
        ax_histx.hist(x, bins=x_bins, color=c)
        ax_histy.hist(y, bins=y_bins, orientation='horizontal', color=c)

def load_ST_crash_time(nshot):
    
    shot = cdbxr.Shot(nshot)  # dict-like accessor to all signals measured in a given shot
    
    # Load positions of sawtooth crashes
    sawtooth_data_folder = Path('./sawtooth_data')
    sawtooth_data_folder.mkdir(exist_ok=True)
    filename = f"{nshot}_st.bin"
    filepath = sawtooth_data_folder / Path(filename)
    
    if not filepath.exists():
        ST_detector(nshot, save_path = './sawtooth_data')
        
    with open(filepath, 'rb') as fp:
        ST_data = pickle.load(fp)
        ST_times = ST_data.times
        
    return ST_times