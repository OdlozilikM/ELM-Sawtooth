# Imports
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
from typing import Optional, Union
from dataclasses import dataclass
from typing import Optional

from core.sawtooth_extraction import ST_detector


# Define a dataclass to hold the pedestal fit results
@dataclass
class PedestalParams:
    """
    Dataclass for holding pedestal parameters. 
    To access its parameters:
    ```
    Example:
            ne_height = PedestalParams.ne_height
    ```
    """
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

        

def get_thomson_data(shot_nr, psi_n_lim:int):
    """
    Function for obtaining Thomson Scattering data for a given shot number. psi_n values are sorted so they can be used as a
    coordinate later.
    """
    shot = cdbxr.Shot(shot_nr)  # dict-like accessor to all signals measured in a given shot
    try:
        ne = shot(name="ne", variant="stray_corrected")
        ne_err = shot(name="ne_err", variant="stray_corrected")
        Te = shot(name="Te", variant="stray_corrected")
        Te_err = shot(name="Te_err", variant="stray_corrected")
        pe = Te * ne * 1.602 / 1E19
        pe_err = np.sqrt((ne_err/ne)**2 + (Te_err/Te)**2) * pe
        pe['units'] = 'Pa'
        print("Using stray corrected data")
    except Exception as exc:
        print(exc)
        ne = shot[f"ne/THOMSON:{shot_nr}"]
        ne_err = shot[f"ne_err/THOMSON:{shot_nr}"]
        Te = shot[f"Te/THOMSON:{shot_nr}"]
        Te_err = shot[f"Te_err/THOMSON:{shot_nr}"]
        pe = shot[f"pe/THOMSON:{shot_nr}"]
        pe_err = shot[f"pe_err/THOMSON:{shot_nr}"]
        print("NOT using stray corrected data")

    psi_n = shot['TS_psi_n']

    for i, psi_n_vals in enumerate(psi_n.data):
        psi_n.data[i] = psi_n.data[i, np.argsort(psi_n_vals)]
    
    # Define interpolated Z for fits
    psi_n_fit_coords = np.linspace(psi_n_lim, 1.15, 100)

    # H-mode times
    t_H_mode_start = shot['t_H_mode_start']
    t_H_mode_end = shot['t_H_mode_end']
    t_H_mode_start = np.atleast_1d(t_H_mode_start.values)
    t_H_mode_end = np.atleast_1d(t_H_mode_end.values)
    
    return ne, ne_err, Te, Te_err, pe, pe_err, psi_n, psi_n_fit_coords, t_H_mode_start, t_H_mode_end


# Define fitting functions
def mtanh(x, b_slope):
    return   ((1+b_slope*x)*np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)) + 1

def F_ped(r, b_height, b_pos, b_width, b_slope):
    return ((b_height)/2) * (mtanh((b_pos-r)/(2*b_width), b_slope))

def fit(data: np.ndarray, error: np.ndarray, psi_n: np.ndarray, psi_n_lim: int, shot_nr: Optional[int] = None, plot: bool = False):
    """
    Function that takes the data and error arrays to return the fiting parameters. Can also plot the fit if plot=True.
    """
    nanmask = np.isnan(data.data)
    data.data[nanmask] = 0
    error.data[nanmask] = np.inf
    error.data[-30:-27] = np.inf
    data = data.assign_coords(Z=('Z', psi_n.values))
    data = data.rename(Z='psi_n')
    error = error.assign_coords(Z=('Z', psi_n.values))
    error = error.rename(Z='psi_n')

    data = data.sel(psi_n = slice(psi_n_lim, None))
    error = error.sel(psi_n = slice(psi_n_lim, None))
    
    b_height_estimate = np.median(data.data[:10])
    amplitude_scale_factor = 10**int(np.log10(b_height_estimate))
    # Redefining F_ped to rescale the data to the same order of magnitude which improves fit robustness
    def _F_ped(r, b_height, b_pos, b_width, b_slope):
        b_height *= amplitude_scale_factor
        b_width /= 1E3
        return F_ped(r, b_height, b_pos, b_width, b_slope)

    # Parameters estimate
    b_height_estimate /= amplitude_scale_factor
    b_SOL_estimate = 0
    b_pos_estimate = 1.05
    b_width_estimate = 0.02 * 1E3
    b_slope_estimate = 0.1
    p0 = [b_height_estimate, b_pos_estimate, b_width_estimate, b_slope_estimate]
    popt, pcov = sps.optimize.curve_fit(_F_ped, data.coords['psi_n'], data.data, p0=p0, sigma=error.data, method='lm')
    stdevs = np.sqrt(np.diag(pcov))
    # Rescale the data to their correct values
    popt[0] *= amplitude_scale_factor
    popt[2] /= 1E3
    stdevs[0] *= amplitude_scale_factor
    stdevs[2] /= 1E3
    p0[0] *= amplitude_scale_factor
    p0[2] /= 1E3
    
    # Plot if specified in the options
    if plot:
        fig, ax = _get_plot()
        max_error = np.clip(error.data, a_min=0, a_max = np.max(data.data))
        ax.errorbar(data.coords['psi_n'] , data.data, max_error, label=f"TS {data.name}",
                    ls='None', marker='d', markersize=6, c='black')
        z = np.linspace(data.coords['psi_n'][0], data.coords['psi_n'][-1], 100)
        ax.plot(z, F_ped(z, *popt), label='Fit', c='red')
        #ax.plot(z, F_ped(z, *p0), label='Initial guess')
        ax.set_ylabel(f"{data.name} [{data.units}]")
        ax.set_title(f"Shot #{shot_nr} - {data.coords['time'].values:.0f}ms - TS pedestal & fit")
        _add_legend(ax)
    
    return popt, stdevs


def ST_time_and_phase(nshot, t, relative_to_nearest=False):
    shot = cdbxr.Shot(nshot)  # dict-like accessor to all signals measured in a given shot
    
    # Load positions of sawtooth crashes
    sawtooth_data_folder = Path('./sawtooth_data')
    sawtooth_data_folder.mkdir(exist_ok=True)
    filename = f"{nshot}_st.bin"
    filepath = sawtooth_data_folder / Path(filename)
    
    if not filepath.exists():
        ST_detector(nshot, save_path='./sawtooth_data')
        
    with open(filepath, 'rb') as fp:
        ST_data = pickle.load(fp)
        ST_times = ST_data.times
        ST_amplitudes = ST_data.amplitudes

    # Allocate empty arrays to hold sawtooth phase and sawtooth amplitudes
    ST_phases = np.full(fill_value=np.nan, shape=t.shape)
    ret_t_delays = np.full(fill_value=np.nan, shape=t.shape)
    ret_ST_amplitudes = np.full(fill_value=np.nan, shape=t.shape)
    
    if len(ST_times) == 0:
        print(f"No STs in shot {nshot}. Cannot compute ST phases.")
        return ST_phases, ST_amplitudes
    
    # Only calculate phases of times lying within first and last ST timestamp
    mask = np.logical_and(ST_times[0] < t, ST_times[-1] > t)
    t_masked = t[mask]
    
    # Get ST timestamps preceding and following each time
    ST_ind_following_t = np.searchsorted(ST_times, t_masked)
    ST_time_following_t = ST_times[ST_ind_following_t]
    ST_time_preceding_t = ST_times[ST_ind_following_t-1]
    
    # Get ST amplitudes preceding each time
    ST_amplitude_preceding_t = ST_amplitudes[ST_ind_following_t-1]
    
    # Calculate ST phase of each time
    ST_duration = ST_time_following_t - ST_time_preceding_t
    t_delay = t_masked - ST_time_preceding_t
    if relative_to_nearest:
        t_early = t_masked - ST_time_following_t
        selection_mask = t_delay > np.abs(t_early)
        t_delay[selection_mask] = t_early[selection_mask]
        
    phases = t_delay / ST_duration
    
    # Mask out ST phases that lasted more than 20ms
    max_duration_mask = ST_duration > 20
    phases[max_duration_mask] = np.nan
    ST_amplitude_preceding_t[max_duration_mask] = np.nan
    t_delay[max_duration_mask] = np.nan
    
    ST_phases[mask] = phases
    ret_ST_amplitudes[mask] = ST_amplitude_preceding_t
    ret_t_delays[mask] = t_delay
    
    # Sanity check
    assert len(ST_phases) == len(t)
    assert len(ret_t_delays) == len(t)
    assert len(ret_ST_amplitudes) == len(t)
    if not relative_to_nearest:
        assert np.all(ST_phases[~np.isnan(ST_phases)] >= 0)
        assert np.all(ret_t_delays[~np.isnan(ret_t_delays)] >= 0)
    assert np.all(ret_ST_amplitudes[~np.isnan(ret_ST_amplitudes)] >= 0)
    
    return ST_phases, ret_t_delays, ret_ST_amplitudes 


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




def gradient_estimate(fit_params: np.ndarray, stdevs: np.ndarray):
    """
    Calculates an estimate of the gradient from a collection of fit parameters, and estimates a worst-case error.
    """
    window_width = 0.002
    psi_pedestal_pos = np.linspace(fit_params[1]-window_width/2, fit_params[1]+window_width/2, 2, endpoint=True)
    
    worst_params = np.copy(fit_params)
    worst_params[0] -= stdevs[0]
    worst_params[2] += stdevs[2]
    
    fit_curve_pedestal = F_ped(psi_pedestal_pos, *fit_params)
    fit_curve_pedestal_worst = F_ped(psi_pedestal_pos, *worst_params)
    
    gradient_estimate = (fit_curve_pedestal[0] - fit_curve_pedestal[-1])/window_width
    gradient_worst = (fit_curve_pedestal_worst[0] - fit_curve_pedestal_worst[-1])/window_width
    gradient_err = gradient_estimate - gradient_worst
    
    return gradient_estimate, gradient_err



def pedestal_params_extraction(shot_nr: int, psi_n_lim:int, save_path:Optional[Union[str, Path]]=None):
    """
    Takes one shot number, and fits the pedestal of all thomson scattering profiles recorded within H-modes. Also calculates 
    the position of the TS measurement in relation to the sawteeth and the ELMs, both in units of time and phase.
    
    From the fits, it calculates the pedestal height, width, and gradient for pressure, density and temperature, as well 
    as their fit errors in standard deviations.
    
    To automatically save the results, pass a folder path as `save_path`.
    """
    save_path = Path(save_path) if save_path is not None else save_path
    # Obtain thomson scattering data and H-mode intervals
    ne, ne_err, Te, Te_err, pe, pe_err, psi_n, psi_n_fit_coords, t_H_mode_start, t_H_mode_end = get_thomson_data(shot_nr, psi_n_lim)
    time_data = ne.coords['time']
    print(f"Number of H-mode intervals : {len(t_H_mode_start)}")
    if len(t_H_mode_start) == 0:
        print(f"No H-mode for shot {shot_nr}. Skipping.")
        return

    # Allocate arrays to hold fit results
    ne_height = np.zeros(len(time_data))
    ne_height_err = np.zeros(len(time_data))
    ne_grad = np.zeros(len(time_data))
    ne_grad_err = np.zeros(len(time_data))
    ne_width = np.zeros(len(time_data))
    ne_width_err = np.zeros(len(time_data))
    ne_time = np.zeros(len(time_data))

    Te_height = np.zeros(len(time_data))
    Te_height_err = np.zeros(len(time_data))
    Te_grad = np.zeros(len(time_data))
    Te_grad_err = np.zeros(len(time_data))
    Te_width = np.zeros(len(time_data))
    Te_width_err = np.zeros(len(time_data))
    Te_time = np.zeros(len(time_data))

    pe_height = np.zeros(len(time_data))
    pe_height_err = np.zeros(len(time_data))
    pe_grad = np.zeros(len(time_data))
    pe_grad_err = np.zeros(len(time_data))
    pe_width = np.zeros(len(time_data))
    pe_width_err = np.zeros(len(time_data))
    pe_time = np.zeros(len(time_data))
    
    # Counters to keep track of total number of succesful fits
    no_fit_counter = 0
    n_pe_fits = 0
    n_Te_fits = 0
    n_ne_fits = 0
    bar = ProgressBar(max_value=len(time_data)).start()  
    # Iterate over all thomson scattering timestamps
    for i, time_point in enumerate(time_data):
        # Skip thomson scattering data if it is not within an H-mode
        if not np.any([time_point > s and time_point < e for s,e in zip(t_H_mode_start,t_H_mode_end)]):
            bar.update(bar.value + 1)
            continue
        # Select the thomson scattering data for the current timestamp
        ne_one = ne.sel(time=time_point, method='nearest')
        ne_err_one = ne_err.sel(time=time_point, method='nearest')
        Te_one = Te.sel(time=time_point, method='nearest')
        Te_err_one = Te_err.sel(time=time_point, method='nearest')
        pe_one = pe.sel(time=time_point, method='nearest')
        pe_err_one = pe_err.sel(time=time_point, method='nearest')
        psi_n_one = psi_n.sel(time=time_point, method='nearest')
        
        # Iterate over density, temperature, and pressure
        for d,e in zip([ne_one, Te_one, pe_one], [ne_err_one, Te_err_one, pe_err_one]):
            try:
                # Try to fit. If an error occurs, go to the next thomson scattering data
                params, stdevs = fit(data=d, error=e, psi_n=psi_n_one, psi_n_lim=psi_n_lim, plot=False)
                if stdevs[0] > params[0]:
                    print(f"Huge error on b_height at {time_point:.0f}ms. err: {stdevs[0]:.1e}, val:{params[0]:.1e}. Dropping it.")
                    continue
                if stdevs[2] > params[2]:
                    print(f"Huge error on b_width at {time_point:.0f}ms. {stdevs[0]:.1e}, val:{params[0]:.1e}. Dropping it.")
                    continue
            except Exception as exc:
                print(f"Could not fit at {time_point:.0f}ms. Reason: {exc}")
                no_fit_counter += 1
                continue
            
            # Save the fit results into the correct fit result arrays and increase succesful fit counters
            if d is ne_one:
                ne_height[n_ne_fits] = params[0]
                ne_height_err[n_ne_fits] = stdevs[0]
                ne_grad[n_ne_fits], ne_grad_err[n_ne_fits] = gradient_estimate(params, stdevs)
                ne_width[n_ne_fits] = params[2]
                ne_width_err[n_ne_fits] = stdevs[2]
                ne_time[n_ne_fits] = time_point
                n_ne_fits += 1
                
            elif d is Te_one:
                Te_height[n_Te_fits] = params[0]
                Te_height_err[n_Te_fits] = stdevs[0]
                Te_grad[n_Te_fits], Te_grad_err[n_Te_fits] = gradient_estimate(params, stdevs)
                Te_width[n_Te_fits] = params[2]
                Te_width_err[n_Te_fits] = stdevs[2]
                Te_time[n_Te_fits] = time_point
                n_Te_fits += 1
                
            elif d is pe_one:
                pe_height[n_pe_fits] = params[0]
                pe_height_err[n_pe_fits] = stdevs[0]
                pe_grad[n_pe_fits], pe_grad_err[n_pe_fits] = gradient_estimate(params, stdevs)
                pe_width[n_pe_fits] = params[2]
                pe_width_err[n_pe_fits] = stdevs[2]
                pe_time[n_pe_fits] = time_point
                n_pe_fits += 1
            else:
                raise Exception
        bar.update(bar.value + 1)   

              
    print(f"Could not fit {no_fit_counter} out of {len(time_data)} measurements.")
    
    ne_st_phase, ne_st_time, _ = ST_time_and_phase(shot_nr, ne_time[:n_ne_fits], relative_to_nearest=True)
    ne_elm_phase, ne_elm_time, _ = ELM_phase(shot_nr, ne_time[:n_ne_fits])
    pe_st_phase, pe_st_time, _ = ST_time_and_phase(shot_nr, pe_time[:n_pe_fits], relative_to_nearest=True)
    pe_elm_phase, pe_elm_time, _ = ELM_phase(shot_nr, pe_time[:n_pe_fits])
    Te_st_phase, Te_st_time, _ = ST_time_and_phase(shot_nr, Te_time[:n_Te_fits], relative_to_nearest=True)
    Te_elm_phase, Te_elm_time, _ = ELM_phase(shot_nr, Te_time[:n_Te_fits])

    pedestal_data = PedestalParams(shot_number=shot_nr,
                                    ne_height = ne_height[:n_ne_fits],
                                    ne_height_err = ne_height_err[:n_ne_fits],
                                    ne_grad = ne_grad[:n_ne_fits],
                                    ne_grad_err = ne_grad_err[:n_ne_fits],
                                    ne_width = ne_width[:n_ne_fits],
                                    ne_width_err = ne_width_err[:n_ne_fits],
                                    ne_time = ne_time[:n_ne_fits],
                                    ne_ELM_phase = ne_elm_phase,
                                    ne_ELM_time = ne_elm_time,
                                    ne_ST_phase = ne_st_phase,
                                    ne_ST_time = ne_st_time,

                                    Te_height = Te_height[:n_Te_fits],
                                    Te_height_err = Te_height_err[:n_Te_fits],
                                    Te_grad = Te_grad[:n_Te_fits],
                                    Te_grad_err = Te_grad_err[:n_Te_fits],
                                    Te_width = Te_width[:n_Te_fits],
                                    Te_width_err = Te_width_err[:n_Te_fits],
                                    Te_time = Te_time[:n_Te_fits],
                                    Te_ELM_phase = Te_elm_phase,
                                    Te_ELM_time = Te_elm_time,
                                    Te_ST_phase = Te_st_phase,
                                    Te_ST_time = Te_st_time,

                                    pe_height = pe_height[:n_pe_fits],
                                    pe_height_err = pe_height_err[:n_pe_fits],
                                    pe_grad = pe_grad[:n_pe_fits],
                                    pe_grad_err = pe_grad_err[:n_pe_fits],
                                    pe_width = pe_width[:n_pe_fits],
                                    pe_width_err = pe_width_err[:n_pe_fits],
                                    pe_time = pe_time[:n_pe_fits],
                                    pe_ELM_phase = pe_elm_phase,
                                    pe_ELM_time = pe_elm_time,
                                    pe_ST_phase = pe_st_phase,
                                    pe_ST_time = pe_st_time)

    
    # Pickle results if a save path has been given
    if save_path is not None:
        save_path.mkdir(exist_ok=True)
        filename = f"{shot_nr}_ped.bin"
        filepath = save_path / Path(filename)
        with open(filepath, 'wb') as fp:
            pickle.dump(pedestal_data, fp)
        
    return pedestal_data




