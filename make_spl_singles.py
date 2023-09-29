'''
Author: Daniella C. Bardalez Gagliuffi
Make singles spectral template library based on SpeX prism library in SPLAT. 
Then interpolates all spectra to a common wavelength range. 
Only needs to be run once or every time the SPLAT library is updated. 
It will take ~20 minutes to complete. 
Next, run make_spl_binary_pickle.py to make all binary spectral template combinations.
Date: 09-17-2020

Modified by: Sandesh Ghimire
Date: 10-01-2023
'''

import splat
from splat import empirical
import pandas as pd
import numpy as np
import time
from scipy.interpolate import interp1d
import sys
sys.setrecursionlimit(1500)

start = time.time()

print('Gathering spectra...')
spc = splat.getSpectrum(spexspt=['M6','T9'], giant=False, VLM=True)
print(time.time() - start)

print('Done with spectra! Now getting some numbers...')
specdf = pd.DataFrame(index=np.arange(len(spc)),columns=['spectra'])
specdf['spectra'] = spc
specdf['optspt'] = specdf['spectra'].map(lambda x: x.opt_type)
specdf['nirspt'] = specdf['spectra'].map(lambda x: x.nir_type)
specdf['spt'] = specdf['spectra'].map(lambda x: splat.classifyByStandard(x)[0])
specdf['sptn'] = specdf['spt'].map(lambda x: splat.typeToNum(x))
specdf['name'] = specdf['spectra'].map(lambda x: x.name)

print('Trim down bad spectral classifications...')
print(specdf.shape)
specdf = specdf[specdf['sptn'] >= 16].reset_index().drop('index',1)
print(specdf.shape)
specdf.to_pickle('test_spl.pickle')

specdf['shortname'] = specdf['spectra'].map(lambda x: x.shortname.split('J')[1])
specdf['MJ'] = specdf['spt'].map(lambda x: empirical.typeToMag(x, '2MASS J', ref='filippazzo2015')[0]).astype('float')
specdf['MH'] = specdf['spt'].map(lambda x: empirical.typeToMag(x, '2MASS H', ref='filippazzo2015')[0]).astype('float')
specdf['MK'] = specdf['spt'].map(lambda x: empirical.typeToMag(x, '2MASS KS', ref='filippazzo2015')[0]).astype('float')
specdf['library'] = specdf['spectra'].map(lambda x: x.library)
specdf['young'] = specdf['library'].map(lambda x: 1 if 'young' in x else 0)
print(time.time() - start)

print('Interpolating spectra to common wavelength range...')
interpdf = pd.DataFrame(index=np.arange(len(specdf)),columns=['spectra'])
interpdf['spectra'] = specdf['spectra']
interpdf['wave'] = interpdf['spectra'].map(lambda x: x.wave)
interpdf['flux'] = interpdf['spectra'].map(lambda x: x.flux)
interpdf['noise'] = interpdf['spectra'].map(lambda x: x.noise)
interpdf['delta_wave'] = interpdf['wave'].map(lambda x: (x[-1]-x[0])/len(x))
interpdf['min_wave'] = interpdf['wave'].map(lambda x: x[0])
interpdf['max_wave'] = interpdf['wave'].map(lambda x: x[-1])
new_wave = np.arange(interpdf['min_wave'].max().value, interpdf['max_wave'].min().value, interpdf['delta_wave'].mean().value)
print(time.time() - start)

for i in range(len(interpdf)):
    f = interp1d(interpdf.loc[i,'wave'],interpdf.loc[i,'flux'])
    fn = interp1d(interpdf.loc[i,'wave'],interpdf.loc[i,'noise'])
    new_flux = f(new_wave)
    new_noise = fn(new_wave)
    new_sp = splat.Spectrum(wave=new_wave, flux=new_flux, noise=new_noise, name=specdf.loc[i,'name'])
    interpdf.loc[i,'new_spectra'] = new_sp
    
specdf['interpsp'] = interpdf['new_spectra']
print(time.time() - start)

print('Flux calibrating spectra...')
tmp = [specdf.loc[i,'interpsp'].fluxCalibrate('2MASS J', specdf.loc[i,'MJ'], absolute=True) for i in range(len(specdf))]
    
specdf['spJ'] = specdf['interpsp'].map(lambda x: splat.filterMag(x,'2MASS J')[0])
specdf['spH'] = specdf['interpsp'].map(lambda x: splat.filterMag(x,'2MASS H')[0])
specdf['spK'] = specdf['interpsp'].map(lambda x: splat.filterMag(x,'2MASS KS')[0])
specdf['interpwave'] = specdf['interpsp'].map(lambda x: x.wave.value)
specdf['interpflux'] = specdf['interpsp'].map(lambda x: x.flux.value)
specdf['interpnoise'] = specdf['interpsp'].map(lambda x: x.noise.value)

specdf = specdf.reset_index().drop('index',1).sort_values('sptn')

#Default protocol = 5 for python 3.8
specdf.to_pickle('template_singles.pickle')
print('Spectral library saved as template_singles.pickle')

specdf = specdf.drop(['spectra','interpsp'],1)
specdf.head()

specdf.to_json('template_singles.json',orient='columns')

print('Spectral library saved as template_singles.json')
print(time.time() - start)