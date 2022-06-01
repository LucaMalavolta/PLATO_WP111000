# From https://github.com/hippke/wotan
# wotan/huber_spline.py
from __future__ import print_function, division
from numba import jit
import numpy as np
import scipy.interpolate

FTOL = 1e-6


def get_gaps_indexes(time, break_tolerance):
    """Array indexes where ``time`` has gaps longer than ``break_tolerance``"""
    gaps = np.diff(time)
    gaps_indexes = np.where(gaps > break_tolerance)
    gaps_indexes = np.add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = np.concatenate(gaps_indexes).ravel()  # Flatten
    gaps_indexes = np.append(np.array([0]), gaps_indexes)  # Start
    gaps_indexes = np.append(gaps_indexes, np.array([len(time)+1]))  # End point
    return gaps_indexes


@jit(fastmath=True, nopython=True, cache=True)
def running_segment(time, flux, mask, window_length, edge_cutoff, cval):
    """Iterator for a single time-series segment using time-series window sliders"""

    size = len(time)
    mean_all = np.full(size, np.nan)
    half_window = window_length / 2
    # 0 < Edge cutoff < half_window:
    if edge_cutoff > half_window:
        edge_cutoff = half_window

    # Pre-calculate border checks before entering the loop (reason: large speed gain)
    low_index = np.min(time) + edge_cutoff
    hi_index = np.max(time) - edge_cutoff
    idx_start = 0
    idx_end = 0

    # Make negative mask to filter out masked values from flux array
    # Use negative (unphysical) flux values because numba doesn't handle NaNs
    mask[mask==0] = -1
    masked_flux = flux * mask

    for i in range(size):
        if time[i] >= low_index and time[i] <= hi_index:
            # Nice style would be:
            #   idx_start = np.argmax(time > time[i] - window_length/2)
            #   idx_end = np.argmax(time > time[i] + window_length/2)
            # But that's too slow (factor 10). Instead, we write:
            while time[idx_start] < time[i] - half_window:
                idx_start += 1
            while time[idx_end] < time[i] + half_window and idx_end < size-1:
                idx_end += 1
            # Get the location estimate for the segment in question
            # iterative method for: biweight, andrewsinewave, welsch
            # drop negative values (these were masked)
            f = masked_flux[idx_start:idx_end]
            f = f[f>-0.000000000001]
            if len(f)==0:
                mean_all[i] = np.nan
            else:
                mean_all[i] = location_iter(f, cval)

    return mean_all


@jit(fastmath=True, nopython=True, cache=True)
def location_iter(data, cval):
    """Robust location estimators"""

    # Numba can't handle strings, so we're passing the location estimator as an int:
    # 1 : biweight
    # 2 : andrewsinewave
    # 3 : welsch
    # (the others are not relevant for location_iter)

    # Initial estimate for the central location
    delta_center = np.inf
    median_data = np.median(data)
    mad = np.median(np.abs(data - median_data))
    center = center_old = median_data

    # Neglecting this case was a bug in scikit-learn
    if mad == 0:
        return center

    # one expensive division here, instead of two per loop later
    cmad = 1 / (cval * mad)

    # Newton-Raphson iteration, where each result is taken as the initial value of the
    # next iteration. Stops when the difference of a round is below ``FTOL`` threshold
    while np.abs(delta_center) > FTOL:
        distance = data - center
        dmad = distance * cmad

        # Inlier weights
        # biweight
        weight = (1 - dmad ** 2) ** 2

        # Outliers with weight zero
        # biweight or welsch
        weight[(np.abs(dmad) >= 1)] = 0

        center += np.sum(distance * weight) / np.sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
    return center
