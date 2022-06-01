import numpy as np
from scipy.signal import find_peaks
import statsmodels.api as sm
from scipy import interpolate


def get_gaps_indexes(time, break_tolerance):
    """Array indexes where ``time`` has gaps longer than ``break_tolerance``"""
    gaps = np.diff(time)
    gaps_indexes = np.where(gaps > break_tolerance)
    gaps_indexes = np.add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = np.concatenate(gaps_indexes).ravel()  # Flatten
    gaps_indexes = np.append(np.array([0]), gaps_indexes)  # Start
    gaps_indexes = np.append(gaps_indexes, np.array([len(time)+1]))  # End point
    return gaps_indexes



def filter_ysd_lowess(time, flux, mask, frac=0.02, prominence=0.001, width=20):

    try:
        peaks, peak_info = find_peaks(flux, prominence = prominence, width = width)
        #print('Peaks:',peaks)
        #if len(peaks) == 0:
        #    print('No peaks found')
        troughs, trough_info = find_peaks(-normalized_flux, prominence = -prominence, width = width)
        #if len(troughs) == 0:
        #    print('No troughs found')
        flux_peaks = flux[peaks]
        flux_troughs = flux[troughs]
        amplitude_peaks = ((flux_peaks[0]-1) + (1-flux_troughs[0]))/2
        #print("Absolute amplitude of main variability = {}".format(amplitude_peaks))
        near_peak_or_trough = [False]*len(flux)

        for i in peaks:
            for j in range(len(time)):
                if abs(time[j] - time[i]) < 0.1:
                    near_peak_or_trough[j] = True

        for i in troughs:
            for j in range(len(time)):
                if abs(time[j] - time[i]) < 0.1:
                    near_peak_or_trough[j] = True

        near_peak_or_trough = np.array(near_peak_or_trough)

        t_cut = time[~near_peak_or_trough]
        flux_cut = flux[~near_peak_or_trough]
        #print('Flux cut done')
    except:
        t_cut = time
        flux_cut = flux
        #print('Flux cut failed')

    #Lowess detrending
    #print('Lowess detrending...')
    full_lowess_flux = np.array([])
    lowess = sm.nonparametric.lowess(flux_cut, t_cut, frac=frac) #model

    #Applying the model to all the data points (including those selected as peaks/troughs)
    poly_function=interpolate.interp1d(t_cut,lowess[:,1],kind='cubic')
    try:
        filter_model=poly_function(time) #new array with the same number of points of flux, representing the model
    except:
        #Discarding some values before and above the interpolation range, otherwise interpolation cannot be done
        time_sel = (time>t_cut[0]) & (time<t_cut[-1])
        filter_model = np.ones(len(time))
        filter_model[time_sel]=poly_function(time[time_sel])

        mask[~time_sel] = 0

    return filter_model, mask
