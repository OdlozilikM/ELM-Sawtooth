def ELM_phase(nshot: int, t: np.ndarray, relative_to_nearest: bool =False):
    """
    Calculates the ELM phase and ELM delay of the timestamps in t, as well as the ELM period. 
    If `relative_to_nearest` is True, the phase and delay is calculated with respect to the nearest ELM, not the previous.
    ELMs that had periods greater than 20ms are not counted.
    """
    shot = cdbxr.Shot(nshot) 
    
    # Get ELM start timestamps
    t_ELM_start = shot['t_ELM_start'].values
    
    # Allocate array to hold phase values
    ELM_phases = np.full(fill_value=np.nan, shape=t.shape)
    ELM_delays = np.full(fill_value=np.nan, shape=t.shape)
    ELM_period = np.full(fill_value=np.nan, shape=t.shape)
    
    if len(t_ELM_start) == 0:
        print(f"No ELMs in shot {nshot}. Cannot compute ELM phases.")
        return ELM_phases
    
    # Only calculate phases of times lying within first and last ELM timestamp
    elm_range_mask = np.logical_and(t_ELM_start[0] < t, t_ELM_start[-1] > t)
    t_masked = t[elm_range_mask]
        
    # Get ELMs preceding and following each time
    ELM_ind_following_t = np.searchsorted(t_ELM_start, t_masked)
    ELM_time_following_t = t_ELM_start[ELM_ind_following_t]
    ELM_time_preceding_t = t_ELM_start[ELM_ind_following_t-1]
    
    # Calculate duration of ELM
    elm_period = ELM_time_following_t - ELM_time_preceding_t
    
    # Calculate phase of t within each ELM
    t_delay = t_masked - ELM_time_preceding_t
    t_early = t_masked - ELM_time_following_t
    selection_mask = t_delay > np.abs(t_early)
    if relative_to_nearest:
        t_delay[selection_mask] = t_early[selection_mask] 
        
    phases = t_delay / elm_period
    
    # Mask out ELMs that lasted more than 20ms
    max_duration_mask = elm_period > 20
    phases[max_duration_mask] = np.nan
    t_delay[max_duration_mask] = np.nan
    ELM_phases[elm_range_mask] = phases
    ELM_delays[elm_range_mask] = t_delay
    ELM_period[elm_range_mask] = elm_period
    
    # Sanity check
    assert len(ELM_phases) == len(t)
    return ELM_phases, ELM_delays, ELM_period

