'''
 NOTE: the code below is ALL taken from wotan  https://github.com/hippke/wotan
 this is just a stripped down version with just the routines required to run
 the selected algorithm, but the same results can be obtained wunning wotan
 directly (an example is provided)
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Reading files (support for mask is included, although not used)
lightcurves=pd.read_csv('../plato_brightbinned158allE.csv')
time=np.array(lightcurves['time'], dtype=np.float64)
flux=np.array(lightcurves['flux_w_transit'], dtype=np.float64)
mask = None

#Alternative implementation by calling wotan directly
#from wotan import flatten
#flatten_lc, trend_lc = flatten(time, flux, method='hspline', window_length=0.3, return_trend=True) #spline Huber

from wotan_biweight import running_segment, get_gaps_indexes

# Maximum gap in time should be half a window size.
# Any larger is nonsense,  because then the array has a full window of data
window_length=1.0 # As Canocchi et al. 2022
cval = 5 # Default in wotan
edge_cutoff = 0 # Default in wotan

break_tolerance = window_length / 2 # Default in wotan





# Numba is very fast, but doesn't play nicely with NaN values
# Therefore, we make new time-flux arrays with only the floating point values
# All calculations are done within these arrays
# Afterwards, the trend is transplanted into the original arrays (with the NaNs)
if mask is None:
    mask = np.ones(len(time))
else:
    mask = np.array(~mask, dtype=float64)  # Invert to stay consistent with TLS
mask_nans = np.isnan(time * flux)

time_compressed = np.ma.compressed(np.ma.masked_array(time, mask_nans))
flux_compressed = np.ma.compressed(np.ma.masked_array(flux, mask_nans))
mask_compressed = np.ma.compressed(np.ma.masked_array(mask, mask_nans))



# Get the indexes of the gaps
gaps_indexes = get_gaps_indexes(time_compressed, break_tolerance=break_tolerance)
filter_model_masked = np.array([])
filter_segment = np.array([])

# Iterate over all segments
for i in range(len(gaps_indexes) - 1):
    time_view = time_compressed[gaps_indexes[i] : gaps_indexes[i + 1]]
    flux_view = flux_compressed[gaps_indexes[i] : gaps_indexes[i + 1]]
    mask_view = mask_compressed[gaps_indexes[i] : gaps_indexes[i + 1]]


    filter_segment = running_segment(
        time_view,
        flux_view,
        mask_view,
        window_length,
        edge_cutoff,
        cval,
    )

    filter_model_masked = np.append(filter_model_masked, filter_segment)


# Insert results of non-NaNs into original data stream
filter_model = np.full(len(time), np.nan)
mask_nans = np.where(~mask_nans)[0]
for idx in range(len(mask_nans)):
    filter_model[mask_nans[idx]] = filter_model_masked[idx]
filter_model[filter_model == 0] = np.nan  # avoid division by zero

filtered = flux / filter_model


plt.scatter(time, flux, s=2, label = 'Input data')
plt.scatter(time, filtered, s=2, label='Filtered data')
plt.plot(time, filter_model, c='C3', label='Filter model')
plt.xlabel('Time [d]')
plt.ylabel('Flux (arbitrary units)')
plt.legend()
plt.show()

P = 32.82977127027492
Tc = 200.23980469999998

plt.figure()
phase = ((time - Tc + P/2) % P ) / P - 0.5
plt.scatter(phase, filtered, s=2)
plt.xlabel('Orbital phase')
plt.ylabel('Flux (arbitrary units)')
plt.show()
