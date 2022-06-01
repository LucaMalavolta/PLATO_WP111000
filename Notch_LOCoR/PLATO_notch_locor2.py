#Notch and LOCoR for PLATO LCs: let's first consider individual quarters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import NandL_core

## Same LC and planet as in figure 12 by Canocchi et al. 2022

lightcurves=pd.read_csv('../plato_brightbinned158allE.csv')
time=np.array(lightcurves['time'], dtype=np.float64)
flux=np.array(lightcurves['flux_w_transit'], dtype=np.float64)
mask = None
# NOTE: mask support not implemented


Prot = 3.12

# Package reuired by N&L
# https://github.com/evertrol/mpyfit


# setting up the data in the format required by N&L

dl = len(time)
data         = np.recarray((dl,),dtype=[('t',float),('fraw',float),('fcor',float),('s',float),('qual',int),('divisions',float)])
data.t = time
data.fcor=flux # / np.nanmedian(outdata.fcor)

data.fraw[:]    = 0
data.s[:]      = 0
data.qual[:]    = 0

# These parameters shouldn't be changed
arclength=False
use_raw=False
deltabic = -1.0
transmask=[-1,-1]
alias_num=0.01
raw = False
resolvabletrans = False
show_progress = True
# demode==1: NOTCH
# demode==2: LOCoR


#Running Notch:
if(Prot>=13):
    wsize = 2.0
    fittimes, depth, filtered, filter_model, badflag = NandL_core.sliding_window(data,windowsize=wsize,use_arclength=arclength,use_raw=raw,deltabic=deltabic,resolvable_trans=resolvabletrans,cleanmask=transmask,show_progress=show_progress) ##Notch Filter
if(Prot>2 and Prot<13): #K2 lcs have a Prot between 0.1-13 days; PLATO lcs have Prot between 1-107 days!!!
    wsize = 1.0
    fittimes, depth, filtered, filter_model, badflag = NandL_core.sliding_window(data,windowsize=wsize,use_arclength=arclength,use_raw=raw,deltabic=deltabic,resolvable_trans=resolvabletrans,cleanmask=transmask,show_progress=show_progress) ##Notch Filter
if(Prot<=2):
    wsize = 0.5
    fittimes, depth, filtered, filter_model, badflag = NandL_core.sliding_window(data,windowsize=wsize,use_arclength=arclength,use_raw=raw,deltabic=deltabic,resolvable_trans=resolvabletrans,cleanmask=transmask,show_progress=show_progress) ##Notch Filter

#Running LOCoR:
if(Prot<=2):
    alias_num=1.0
    wsize = Prot
    fittimes, depth, filtered, filter_model, badflag = NandL_core.rcomb(data,wsize,cleanmask=transmask,aliasnum=alias_num) ##LOCoR


notch_depth = depth[0].copy()
deltabic    = depth[1].copy()
bicstat = deltabic-np.median(deltabic)
bicstat = 1- bicstat/np.max(bicstat)


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
phase = ((time - Tc + P/2) % P ) / P + 0.5
plt.scatter(phase, filtered, s=2)
plt.xlabel('Orbital phase')
plt.ylabel('Flux (arbitrary units)')
plt.show()
