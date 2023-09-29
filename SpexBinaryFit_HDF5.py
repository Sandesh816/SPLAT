'''
Author: Sandesh Ghimire

This is the main program of the SPLAT project. It imports the data made using make_spl_singles.py and make_spl_binaries.py programs. Then, it measures the Chi Square
value for all the single and binary templates made. Then, top 5 closest spectrums are selected and they are graphically compared against the original spectrum object.
Based on the graphical comparison, using Hypothesis testing, we determine whether the given original spectrum is a Single Brown Dwarf or a Binary Brown Dwarf.
Make binary spectral template library based on SpeX prism library in SPLAT. 

It imports data in HDF5 format.

Date: 03-02-2023
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
nbest = 4

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
    specdf = pd.read_hdf(file)
    spectra = [make_spectrum(wave=row['interpwave'],
        flux=row['interpflux'],noise=row['interpnoise'],
        name=row['DESIGNATION'],spt=row['spt']) for _, row in specdf.iterrows()]

    specdf['spectra'] = spectra
    print("Time to run build_spectra_single is ",time.time() - start)
    return specdf

def build_spectra_binary(file):
    start = time.time()
    specdf = pd.read_hdf(file)
    bwaves = np.stack(specdf['bwave'].values)
    bfluxes = np.stack(specdf['bflux'].values)
    bnoises = np.array(specdf['bnoise'].values)
    shortnames = np.array(specdf['SHORTNAME1'] + ' & ' + specdf['SHORTNAME2'].values)
    spectral_types = np.array(specdf['Spectral Type'].values)
    interpwaves1 = np.stack(specdf['interpwave1'].values)
    interpfluxes1 = np.stack(specdf['interpflux1'].values)
    interpnoises1 = np.array(specdf['interpnoise1'].values)
    shortnames1 = np.array(specdf['SHORTNAME1'].values)
    spt1 = np.array(specdf['spt1'].values)
    interpwaves2 = np.stack(specdf['interpwave2'].values)
    interpfluxes2 = np.stack(specdf['interpflux2'].values)
    interpnoises2 = np.array(specdf['interpnoise2'].values)
    shortnames2 = np.array(specdf['SHORTNAME2'].values)
    spt2 = np.array(specdf['spt2'].values)
    binsp = [make_spectrum(wave=wave, flux=flux, noise=noise, name=name, spt=spt)
             for wave, flux, noise, name, spt in zip(bwaves, bfluxes, bnoises, shortnames, spectral_types)]
    sp1 = [make_spectrum(wave=wave, flux=flux, noise=noise, name=name, spt=spt)
           for wave, flux, noise, name, spt in zip(interpwaves1, interpfluxes1, interpnoises1, shortnames1, spt1)]
    sp2 = [make_spectrum(wave=wave, flux=flux, noise=noise, name=name, spt=spt)
           for wave, flux, noise, name, spt in zip(interpwaves2, interpfluxes2, interpnoises2, shortnames2, spt2)]
    specdf['binsp'] = binsp
    specdf['sp1'] = sp1
    specdf['sp2'] = sp2
    print("Time taken for build_spectra_binary is ", time.time()-start)
    return specdf

def spexsinglefit(sp, specdf):
    start = time.time()
    tmp = specdf['spectra'].apply(lambda x: splat.compareSpectra(x,sp,mask_telluric=True, statistic = 'chisqr'))
    chi2 = tmp.apply(lambda x: x[0])
    scl = tmp.apply(lambda x: x[1])
    scomp = pd.DataFrame(index = np.arange(len(specdf)), columns = ['single','chi2','scl'])
    scomp['single'] = specdf['spectra']
    scomp['name'] = scomp['single'].apply(lambda x: x.name)
    scomp['spt'] = scomp['single'].apply(lambda x: x.lit_type)
    scomp['chi2'] = chi2
    scomp['scl'] = scl
    scomp['dof'] = scomp['single'].apply(lambda x: x.dof)
    scomp['redchi2'] = scomp['chi2'] / (sp.dof + scomp['dof'] - 2)
    scomp = scomp.sort_values('redchi2')
    summ = scomp[['name','spt','redchi2']]
    summ.head(nbest).to_csv('singlefits.csv', index=False)
    sp.singlefit = scomp.loc[0,'single']
    sp.singlefit_chi2 = scomp.loc[0,'chi2']
    sp.singlefit_scl = scomp.loc[0,'scl']
    sp.singlefit_redchi2 = scomp.loc[0,'redchi2']
    print("Time to run spexsinglefit is ",time.time() - start)
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

def spexbinaryfit(sp, bindf):
    start = time.time()  
    tmp = bindf['binsp'].map(lambda x: splat.compareSpectra(x,sp,mask_telluric=True))
    chi2 = tmp.map(lambda x: x[0])
    scl = tmp.map(lambda x: x[1])
    bcomp = pd.DataFrame(index = np.arange(len(bindf)), columns = ['binsp','chi2','scl','sp1','sp2'])
    bcomp['binsp'] = bindf['binsp']
    bcomp['sp1'] = bindf['sp1']
    bcomp['sp2'] = bindf['sp2']
    bcomp['chi2'] = chi2
    bcomp['scl'] = scl
    bcomp['dof'] = bcomp['sp1'].map(lambda x: x.dof) + bcomp['sp2'].map(lambda x: x.dof) - 2 
    bcomp['redchi2'] = bcomp['chi2'] / (sp.dof + bcomp['dof'] - 2) 
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
  
slib = build_spectra_single("fluxcal_singles.h5")
scomp= spexsinglefit(sp, slib)
plot_singles(sp, scomp)

print("Done with singles")
print("Running binaries...")

blib = build_spectra_binary("template_binaries.h5")
bcomp = spexbinaryfit(sp, blib)
plot_binaries(sp, bcomp)

def print_summary():

    return


if __name__ == '__main__':
    start = time.time()
    # Write a path and plotpath here to save files to a specific location. Also add path in lines 105 and 150
    # path = '/Users/sandesh816/Research/spexbinaryfit/' 
    # plotpath = '/Users/sandesh816/Research/spexbinaryfit/'

    
 