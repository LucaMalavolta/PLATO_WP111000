#Adapted frpom https://github.com/mbattley/YSD

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from ysd_lowess  import filter_ysd_lowess, get_gaps_indexes


lightcurves=pd.read_csv('../plato_brightbinned158allE.csv')
time=np.array(lightcurves['time'], dtype=np.float64)
flux=np.array(lightcurves['flux_w_transit'], dtype=np.float64)
mask = None
# NOTE: mask support not implemented
# NOTE: flux is already normalized, I cannot guarantee that everything
# will work with raw fluxes

#NOTE: masked arrays are not supported here
if mask is None:
    mask = np.ones(len(time))
else:
    mask = np.array(~mask, dtype=float64)


break_tolerance = 0.3 # days
# Get the indexes of the gaps
gaps_indexes = get_gaps_indexes(time, break_tolerance=break_tolerance)
filter_model = np.array([])
filter_mask = np.array([])




# Iterate over all segments
for i in range(len(gaps_indexes) - 1):
    time_view = time[gaps_indexes[i] : gaps_indexes[i + 1]]
    flux_view = flux[gaps_indexes[i] : gaps_indexes[i + 1]]
    mask_view = mask[gaps_indexes[i] : gaps_indexes[i + 1]]
    median_val = np.median(flux_view)

    filter_segment, mask_segment = filter_ysd_lowess(time_view, flux_view / median_val, mask_view)
    # In addition to the filter, a mask is given back to flag points where
    # interpolation did not work
    filter_model= np.append(filter_model, filter_segment * median_val)
    filter_mask= np.append(filter_mask, mask_segment)


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
