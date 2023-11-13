# Imports
import statsmodels
from dataclasses import dataclass
import sys
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
plt.rcParams.update({'font.size': 14})

from core.sawtooth_extraction import ST_detector, ST_time_and_phase


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
    
    
    
def load_pedestal_data(load_path=str):
    pedestal_data_folder = Path(load_path)
    pedestal_data_list = []

    for item in pedestal_data_folder.iterdir():
        if item.is_file and item.suffix == ".bin":
            with open(item, 'rb') as fp:
                pedestal_data_list.append(pickle.load(fp))
    return pedestal_data_list

# Define some functions for generating plots with a specific style.
def _get_plot():
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel('time [ms]')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    return fig, ax

def _add_legend(ax):
    ax.legend(loc="upper right", fontsize="12")
    

# Define functions for calculating the ELM lengths and ELM times
def get_elm_length_and_time(shot_nr: int, times: np.ndarray):
    shot = cdbxr.Shot(shot_nr)
    t_ELM_start = shot['t_ELM_start']
    idx = np.searchsorted(t_ELM_start.data, times)
    no_search_result_map = idx == len(t_ELM_start.data)
    idx[no_search_result_map] = 1
    elm_times = times - t_ELM_start.data[idx-1]
    elm_periods = t_ELM_start.data[idx] - t_ELM_start.data[idx-1]
    elm_periods[no_search_result_map] = np.nan
    
    return elm_periods, elm_times


# Define function for getting the ST phase, ELM durations, ST amplitudes, and ST times of each ELM 
def get_ELM_ST_phase_and_duration(shot_nr: int, load_path: str):
    shot = cdbxr.Shot(shot_nr)
    t_ELM_start = shot['t_ELM_start']
    ELM_duration = np.diff(t_ELM_start)
    
    ELM_ST_phase, ELM_ST_time, ST_amplitudes = ST_time_and_phase(shot_nr, t_ELM_start[1:], load_path)
    return ELM_ST_phase, ELM_duration, ST_amplitudes, ELM_ST_time 


def scatter_pedestal_params(load_path: str,x: str = 'ELM_phase', s: str = 'pe', p: str = 'grad'):
    """
    Creates a scatterplot of one pedestal parameter (grad, height, width) of a Thomson Scatter variable (pe, Te, ne) as 
    a function of either ELM or ST phase or time (ELM_phase, ELM_time, ST_phase, ST_time).
    """
    pedestal_data_list=load_pedestal_data(load_path=load_path)
    max_data = 0
    fig, ax = _get_plot()
    
    all_x = np.empty(0)
    for i, d in enumerate(pedestal_data_list):
        time = getattr(d, s+'_time')
        data = getattr(d, s+'_'+p)
        data_err = getattr(d, s+'_'+p+'_err')
        
        mask = time < 1200

        if x == 'ELM_phase':
            x_values = getattr(d, s+'_'+x)
            x_values = x_values[0]
        if x == 'ELM_time':
            _, x_values = get_elm_length_and_time(d.shot_number, time)
            valid_elm_lengths = np.logical_and(0 < x_values, x_values < 20)
            mask = np.logical_and(valid_elm_lengths, mask)
        else:
            x_values = getattr(d, s+'_'+x)
            
        # Plot
        ax.errorbar(x_values[mask], data[mask], data_err[mask], 
                     c='red', ls='None', marker='d', label=f"{d.shot_number}", markersize=6, alpha=0.5)
        
        if np.nanmax(data[mask]) > max_data:
            max_data = np.nanmax(data[mask])
        all_x = np.concatenate((all_x, x_values[mask]))
            
    ax.set_ylabel(s+'_'+p)
    ax.set_xlabel(x)
    ax.set_ylim((0, max_data))
    if x == 'ELM_time':
        ax.set_xlim((0, np.percentile(all_x, 95)))
    ax.set_title("TS pedestal parameters")
    




def scatter_pedestal_params_4plots(load_path: str, x: str = 'ELM_phase', s: str = 'pe', p: str = 'grad', ax=None):
    """
    Creates a scatterplot of one pedestal parameter (grad, height, width) of a Thomson Scatter variable (pe, Te, ne) as 
    a function of either ELM or ST phase or time (ELM_phase, ELM_time, ST_phase, ST_time).
    """
    # Assume load_pedestal_data and get_elm_length_and_time functions are also in this library
    # from .your_data_loading_module import load_pedestal_data, get_elm_length_and_time

    pedestal_data_list = load_pedestal_data(load_path=load_path)
    max_data = 0
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size

    all_x = np.empty(0)
    for i, d in enumerate(pedestal_data_list):
        time = getattr(d, s+'_time')
        data = getattr(d, s+'_'+p)
        data_err = getattr(d, s+'_'+p+'_err')

        mask = time < 1200

        if x == 'ELM_phase':
            x_values = getattr(d, s+'_'+x)
            x_values = x_values[0]
        if x == 'ELM_time':
            _, x_values = get_elm_length_and_time(d.shot_number, time)
            valid_elm_lengths = np.logical_and(0 < x_values, x_values < 20)
            mask = np.logical_and(valid_elm_lengths, mask)
        else:
            x_values = getattr(d, s+'_'+x)

        # Plot
        ax.errorbar(x_values[mask], data[mask], data_err[mask],
                     c='red', ls='None', marker='d', label=f"{d.shot_number}", markersize=6, alpha=0.5)

        if np.nanmax(data[mask]) > max_data:
            max_data = np.nanmax(data[mask])
        all_x = np.concatenate((all_x, x_values[mask]))

    ax.set_ylabel(s+'_'+p)
    ax.set_xlabel(x)
    ax.set_ylim((0, max_data))
    if x == 'ELM_time':
        ax.set_xlim((0, np.percentile(all_x, 95)))