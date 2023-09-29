'''
Author: Sandesh Ghimire

Make binary spectral template library based on SpeX prism library in SPLAT. 
It takes 1800 spectral objects from the SPLAT library, and makes combination of each object with the other, in total ~800,000 binary combinations. Then, each of the 
combination data is stored, along with its wavelength, flux, Spectral Type, and error information. Then, Chi square is calculated for a given template against the binary 
combination that we made. Then, 5 closest combinations are mapped against the original spectrum to check its closeness.

Date: 21-01-2023
'''
import time
import splat
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import itertools
from itertools import combinations
import sys
sys.setrecursionlimit(1500)

# path = '/Users/sandesh816/Research/spexbinaryfit/' 

start = time.time()
specdf = pd.read_json('fluxcal_singles.json')
specdf = specdf.sort_values('sptn').reset_index().drop(['index'],1)
specdf['sp'] = ''

print(time.time() - start)
print('Building spectra...')
for i in range(len(specdf)):
    flux = pd.Series(specdf.loc[i,'interpflux']).fillna(np.nan)
    noise = pd.Series(specdf.loc[i,'interpnoise']).fillna(np.nan)
    specdf.loc[i,'sp'] = splat.Spectrum(wave=specdf.loc[i,'interpwave'],
    flux=flux,noise=noise,name=specdf.loc[i,'DESIGNATION'],instrument='SPEX')
print(time.time() - start)

print('Combining primary and secondary spectra...')
#spl = specdf['spectra']

print(specdf['spt'])
valid_values =  ['L3.0', 'L4.0', 'L5.0']#  ['L0.0', 'L1.0', 'L2.0', 'L3.0', 'L4.0', 'L5.0', 'L6.0']
mask = specdf['spt'].isin(valid_values)
specdf = specdf[mask]
print("The new length is ", len(specdf['spt']))

spl = specdf['sp']#[:100]#[:5]
ncomb = len(list(combinations(spl,2)))
combdf = pd.DataFrame(index=np.arange(ncomb))
sp1sp2 = pd.Series(list(combinations(spl,2)))
# combinations = list(itertools.product(specdf['sp'][:2], specdf['sp'][:2]))
# combdf = pd.DataFrame(index=np.arange(len(combinations)))
# result = [x+y for x, y in combinations]
# sp1sp2 = pd.Series(result)
# print(sp1sp2)
combdf['sp1'] = sp1sp2.map(lambda x: x[0])
combdf['sp2'] = sp1sp2.map(lambda x: x[1])
print(time.time() - start)

specdf1 = specdf.copy()
specdf1 = specdf1.add_suffix('1')
specdf1 = specdf1.rename(columns={'spectra1':'sp1'})

specdf2 = specdf.copy()
specdf2 = specdf2.add_suffix('2')
specdf2 = specdf2.rename(columns={'spectra2':'sp2'})

print('Merging dataframes...')
tt1 = pd.merge(combdf,specdf1,on='sp1')
tt2 = pd.merge(tt1,specdf2,on='sp2')
bindf = tt2.copy()
print(time.time() - start)

'''
print('Exclude cases where primary is young and secondary is not - unless it is a T dwarf (because we have no young T dwarf templa\
tes, and they may look similar to field T dwarfs)')
okbin = bindf[(bindf['sptn1'] <= 28) & (bindf['young1'] == 1) & (bindf['young2'] == 1)].index
badbin = bindf[(bindf['sptn1'] <= 28) & (bindf['young1'] == 1) & (bindf['sptn2'] < 30) & (bindf['young2'] == 0)].index
print('Check that good binaries do not overlap with bad binaries...')
okbin.intersection(badbin)
print('All good! Drop those bad binaries...')
bindf = bindf.drop(badbin,0).reset_index().drop('index',1).sort_values('sptn1')
print(time.time() - start)
'''
print('Building binary templates...')
start = time.time()
bindf['bflux'] = bindf['sp1'].map(lambda x: x.flux) + bindf['sp2'].map(lambda x: x.flux)
print(time.time() - start)
bindf['bwave'] = bindf['sp1'].map(lambda x: x.wave)
print(time.time() - start)
bindf['bnoise'] = (bindf['sp1'].map(lambda x: x.noise**2) + bindf['sp2'].map(lambda x: x.noise**2))**0.5
print(time.time() - start)
bindf['binsp'] = [splat.Spectrum(wave=bindf.loc[i,'bwave'],flux=bindf.loc[i,'bflux'],noise=bindf.loc[i,'bnoise']) for i in range(len(bindf))]
print(time.time() - start)
print('Spectral typing the combined-light synthetic binaries...(This will take time)')
vfunc = np.vectorize(lambda x: splat.classifyByStandard(x, fit_ranges=[[0.9,1.35],[1.45,1.80],[1.90,2.45]])[0])
bindf['Spectral Type'] = vfunc(bindf['binsp'].values)
#bindf['Spectral Type'] = bindf['binsp'].map(lambda x: splat.classifyByStandard(x, fit_ranges=[[0.9,1.35],[1.45,1.80],[1.90,2.45]])[0])
print(time.time() - start)

bindf['deltaMJ'] = bindf['MJ2'] - bindf['MJ1']
bindf['deltaMH'] = bindf['MJ2'] - bindf['MJ1']
bindf['deltaMK'] = bindf['MJ2'] - bindf['MJ1']

bindf = bindf.drop(['sp1','sp2','binsp'],1)
bindf.to_json('template_binaries.json',double_precision=15,orient='records')
end = time.time()
print(end-start)
print("Done") 