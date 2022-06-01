
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.signal import find_peaks
import statsmodels.api as sm
from scipy import interpolate

lightcurves=pd.read_csv('plato_brightbinned058_lowess.csv')
time=np.array(lightcurves['time'], dtype=np.float64)
flux=np.array(lightcurves['flux_w_transitEarth'], dtype=np.float64)
filter_model = np.array(lightcurves['model_YSD'], dtype=np.float64)
mask = None
# NOTE: mask support not implemented

filtered = flux / np.median(flux) /filter_model

plt.scatter(time, flux/ np.median(flux), s=2, label = 'Input data')
plt.scatter(time, filtered, s=2, label='Filtered data')
plt.plot(time, filter_model, c='C3', label='Filter model')
plt.xlabel('Time [d]')
plt.ylabel('Flux (arbitrary units)')
plt.legend()
plt.show()



P =  15.831225651698773,
Tc = 200.2398047

plt.figure()
phase = ((time - Tc + P/2) % P ) / P - 0.5
plt.scatter(phase, filtered, s=2)
plt.xlabel('Orbital phase')
plt.ylabel('Flux (arbitrary units)')
plt.show()
