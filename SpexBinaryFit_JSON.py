'''
Author: Sandesh Ghimire

This is the main program of the SPLAT project. It imports the data made using make_spl_singles.py and make_spl_binaries.py programs. Then, it measures the Chi Square
value for all the single and binary templates made. Then, top 5 closest spectrums are selected and they are graphically compared against the original spectrum object.
Based on the graphical comparison, using Hypothesis testing, we determine whether the given original spectrum is a Single Brown Dwarf or a Binary Brown Dwarf.
Make binary spectral template library based on SpeX prism library in SPLAT. 

It imports the data in JSON format.

Date: 06-02-2023
'''
import splat
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import matplotlib
import json
import pickle
from splat import empirical
matplotlib.rcParams['figure.figsize'] = 12,8
nbest = 5

# Getting the spectrum of the object we want to do comparison with:
sp = splat.getSpectrum(shortname='1341-3052')[0]

def make_spectrum(wave, flux, noise, name, spt):
    flux = pd.Series(flux).fillna(np.nan)
    noise = pd.Series(noise).fillna(np.nan)
    sp = splat.Spectrum(wave=wave,flux=flux,noise=noise,name=name,instrument='SPEX')
    sp.lit_type = spt
    return sp

def build_spectra_single(file):
    start = time.time()
    specdf = pd.read_json(file)
    specdf['spectra'] = np.nan
    rows = range(len(specdf))
    spectra = []
    for i, row in enumerate (rows):
        spectra.append(make_spectrum(wave=specdf.loc[i,'interpwave'],
            flux=specdf.loc[i,'interpflux'],noise=specdf.loc[i,'interpnoise'],
            name=specdf.loc[i,'DESIGNATION'],spt=specdf.loc[i,'spt']))

    specdf.loc[rows, 'spectra']= spectra
    print("Time to run build_spectra_single is ",time.time() - start)
    return specdf

def build_spectra_binary(file):
    start = time.time()
    specdf = pd.read_json(file)
    specdf['binsp'] = np.nan
    specdf['sp1'] = np.nan
    specdf['sp2'] = np.nan
    rows = range(len(specdf))
    binsp = []
    sp1 = []
    sp2 = []
    bwave = specdf['bwave'].values
    bflux = specdf['bflux'].values
    bnoise = specdf['bnoise'].values
    shortname1 = specdf['SHORTNAME1'].values
    shortname2 = specdf['SHORTNAME2'].values
    spectral_type = specdf['Spectral Type'].values
    interpwave1 = specdf['interpwave1'].values
    interpflux1 = specdf['interpflux1'].values
    interpnoise1 = specdf['interpnoise1'].values
    spt1 = specdf['spt1'].values
    interpwave2 = specdf['interpwave2'].values
    interpflux2 = specdf['interpflux2'].values
    interpnoise2 = specdf['interpnoise2'].values
    spt2 = specdf['spt2'].values
    
    for i, row in enumerate(rows):
        binsp.append(make_spectrum(wave=bwave[row], flux=bflux[row],
                            noise=bnoise[row],
                            name=shortname1[row] + ' & ' + shortname2[row],
                            spt=spectral_type[row]))
        sp1.append(make_spectrum(wave=interpwave1[row],
                              flux=interpflux1[row],
                              noise=interpnoise1[row],
                              name=shortname1[row],
                              spt=spt1[row]))
        sp2.append(make_spectrum(wave=interpwave2[row],
                              flux=interpflux2[row],
                              noise=interpnoise2[row],
                              name=shortname2[row],
                              spt=spt2[row]))

    specdf.loc[rows, 'binsp'] = binsp
    specdf.loc[rows, 'sp1'] = sp1
    specdf.loc[rows, 'sp2'] = sp2
    print("Time taken for build_spectra_binary is ", time.time()-start)
    return specdf

def spexsinglefit(sp, specdf):
    start = time.time()
    tmp = specdf['spectra'].map(lambda x: splat.compareSpectra(x,sp,mask_telluric=True, statistic = 'chisqr'))
    chi2 = tmp.map(lambda x: x[0]) 
    scl = tmp.map(lambda x: x[1]) 
    scomp = pd.DataFrame(index = np.arange(len(specdf)), columns = ['single','chi2','scl']) 
    scomp['single'] = specdf['spectra'] 
    scomp['name'] = scomp['single'].map(lambda x: x.name)
    scomp['spt'] = scomp['single'].map(lambda x: x.lit_type)
    scomp['chi2'] = chi2
    scomp['scl'] = scl
    scomp['dof'] = scomp['single'].map(lambda x: x.dof) 
    scomp['redchi2'] = scomp['chi2'] / (sp.dof + scomp['dof'] - 2) 
    scomp = scomp.sort_values('redchi2') 
    summ = scomp[['name','spt','redchi2']] 
    summ.head(nbest).to_csv('singlefits.csv') 
    sp.singlefit = scomp.loc[0,'single'] 
    sp.singlefit_chi2 = scomp.loc[0,'chi2'] 
    sp.singlefit_scl = scomp.loc[0,'scl']
    sp.singlefit_redchi2 = scomp.loc[0,'redchi2'] 
    print("Time to run spex_singlefit is ",time.time() - start)  
    return scomp.sort_values('redchi2',ascending=True) 

def plot_singles(sp,scomp):
    for i in np.arange(nbest):
        fig = plt.figure(figsize=(12,8))
        pp = scomp.loc[i,'single']
        sp.normalize()
        pp.normalize()
        scl = splat.compareSpectra(pp, sp, mask_telluric = True)[1]     
        print("single scl after flux calibration is", str(scl))
        print("without flux calibration, scl was", str(scomp.loc[i, 'scl']))
        plt.plot(sp.wave,sp.flux * scl, label=str(sp.name),color='k') # * scomp.loc[i,'scl']
        plt.plot(pp.wave,pp.flux, label=pp.name,color=sns.xkcd_rgb['scarlet'])
        plt.legend()
        #plt.ylim(-0.5*1e-11,4*1e-11)
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Flux (erg/$\mu$m/s/cm$^2$)')
        plt.savefig('singlefit_'+str(i)+'.jpg')
        plt.show()
        plt.close()

import numpy as np
import pandas as pd
import splat

def spexbinaryfit(sp, bindf):
    start = time.time()
    sp1 = np.array(bindf['sp1'].tolist())
    sp2 = np.array(bindf['sp2'].tolist())
    binsp = np.array(bindf['binsp'].tolist())
    chi2 = np.array([splat.compareSpectra(b, sp, mask_telluric=True)[0] for b in binsp])
    scl = np.array([splat.compareSpectra(b, sp, mask_telluric=True)[1] for b in binsp])
    dof = np.array([s1.dof + s2.dof - 2 for s1, s2 in zip(sp1, sp2)])
    redchi2 = chi2 / (sp.dof + dof - 2)
    bcomp = pd.DataFrame({'binsp': binsp, 'sp1': sp1, 'sp2': sp2, 'chi2': chi2, 'scl': scl, 'dof': dof, 'redchi2': redchi2})
    bcomp = bcomp.sort_values('redchi2')
    bcomp.head(nbest).to_csv('binaryfits.csv')
    print("The time to run spexbinaryfit is ", time.time()-start)
    return bcomp.sort_values('redchi2',ascending=True)

def plot_binaries(sp,bcomp):
    for i in np.arange(nbest):
        fig = plt.figure(figsize=(12,8))
        pp = bcomp.loc[i,'sp1']
        ss = bcomp.loc[i,'sp2']
        bb = bcomp.loc[i,'binsp']
        sp.normalize()
        pp.normalize()
        ss.normalize()
        bb.normalize()
        scl = splat.compareSpectra(bb, sp, mask_telluric= True)[1]
        chi2 = splat.compareSpectra(bb, sp, mask_telluric= True)[0]
        print("New scl for binary is ", scl)
        print("Previous scl for binary is ", bcomp.loc[i, 'scl'])
        print("chi2 for binary is ", chi2)
        plt.plot(sp.wave,sp.flux * scl, label=str(sp.name),color='k') #bcomp.loc[i,'scl']
        plt.plot(pp.wave,pp.flux *0.5, label='P '+ str(pp.name),color=sns.xkcd_rgb['scarlet'])
        plt.plot(ss.wave,ss.flux * 0.5, label='S '+ str(ss.name),color=sns.xkcd_rgb['mid blue'])
        plt.plot(bb.wave, bb.flux, label = 'Binary Template', color= 'g')
        plt.legend()
        #plt.ylim(-0.5*1e-11,4*1e-11)
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Flux (erg/$\mu$m/s/cm$^2$)')
        plt.savefig('binaryfit_'+str(i)+'.jpg')
        plt.show()
        plt.close()   
  
slib = build_spectra_single("fluxcal_singles.json")
scomp= spexsinglefit(sp, slib)
plot_singles(sp, scomp)

print("Done with singles")

blib = build_spectra_binary("template_binaries.json")
bcomp = spexbinaryfit(sp, blib)
plot_binaries(sp, bcomp)

def print_summary():

    return


if __name__ == '__main__':
    start = time.time()
    # Write a path and plotpath here to save files to specific location. Also add path in line 126 and line 172
    # path = '/Users/sandesh816/Research/spexbinaryfit/' 
    # plotpath = '/Users/sandesh816/Research/spexbinaryfit/'

    
 