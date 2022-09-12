#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:52:21 2022

@author: adrian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean as cmo
from matplotlib.patches import Circle
import astropy.io.fits as fits
from astropy.modeling.models import Gaussian2D
import os
import sys
import warnings

if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))
from SSIM_utils import __compare_images


##### Addition of synthetic stars
def __synthetic_stars(gal, hdu, n=None, peak=None, fwhm=None, e=None, x=None, y=None, theta=None, \
                      peak_range=[0.25,1.25], fwhm_range=[2,6], e_range=[0,0.5], theta_range=[0,360], \
                      gal_size=None, simil=False, split=False, plot=False, circles=False, savepath=None, cmap=None, SSIM_cmap=None):        
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r
    
    ########## Data preparation
    if n is None:
        for i in [peak, fwhm, e, x, y, theta]:
            try:
                n = len(i)
                break
            except:
                n = 1
                
    if peak is not None and np.shape(peak) == (): peak = n*[peak]
    if fwhm is not None and np.shape(fwhm) == (): fwhm = n*[fwhm]
    if e is not None and np.shape(e) == (): e = n*[e]
    if x is not None and np.shape(x) == (): x = n*[x]
    if y is not None and np.shape(y) == (): y = n*[y]
    if theta is not None and np.shape(theta) == (): theta = n*[theta]
    
    ##########
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']
    img = np.array(hdu.data, dtype=float);  head = hdu.header
    
    if peak is None: peak = np.random.uniform(low=peak_range[0]*np.max(img), high=peak_range[1]*np.max(img), size=n)
    if fwhm is None: fwhm = np.random.uniform(low=fwhm_range[0], high=fwhm_range[1], size=n)
    if e is None: e = np.random.uniform(low=e_range[0], high=e_range[1], size=n)
    if gal_size and not x and not y:
        mask_image = np.ones(shape=np.shape(img))
        mask_image[int((0.5-gal_size/2)*np.shape(img)[1]):int((0.5+gal_size/2)*np.shape(img)[1]),
                   int((0.5-gal_size/2)*np.shape(img)[0]):int((0.5+gal_size/2)*np.shape(img)[0])] = 0
        
        x, y = np.where(mask_image == 1)
        ind = np.random.randint(low=0, high=len(x), size=n)
        x = x[ind];  y = y[ind]
        
    if x is None: x = np.random.randint(low=0, high=np.shape(img)[1], size=n)
    if y is None: y = np.random.randint(low=0, high=np.shape(img)[0], size=n)
    if theta is None: theta = np.random.uniform(low=theta_range[0], high=theta_range[1], size=n)

    if e is not(None): fwhmx = fwhm;  fwhmy = fwhmx*(1-np.array(e))
    else: fwhmx = fwhm;  fwhmy = fwhm
    sigmax = fwhmx/(2*np.sqrt(2*np.log(2)))
    sigmay = fwhmy/(2*np.sqrt(2*np.log(2)))

    synth_img = img.copy()
    y_img, x_img = np.mgrid[0:np.shape(img)[1], 0:np.shape(img)[0]]

    for j in range(len(x)):
        synth_img += Gaussian2D(peak[j], y[j], x[j], sigmax[j], sigmay[j], theta=theta[j]*np.pi/180)(y_img, x_img)
    
    if len(x) == 1: synth_label = 'Synthetic object\n(Peak=%.2f, X=%i, Y=%i, \nFWHMX=%.2f, FWHMY=%.2f, θ=%.2fº)' % (peak[0], x[0], y[0], abs(fwhmx[0]), abs(fwhmy[0]), theta[0])
    else: synth_label = '%i Synthetic stars' % (n)
    
    if simil: _, synth_ssim = __compare_images(img, synth_img, plot=plot, split=split, \
                                               suptitle=gal+' - '+band, title1='Original', title2=synth_label, cmap=cmap, SSIM_cmap=SSIM_cmap)
    if plot and not simil: 
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
        fig.suptitle(gal+' - '+band)
        im0 = ax[0].imshow(img, norm=LogNorm(), cmap=cmap);        ax[0].set_title('Original')
        ax[1].imshow(synth_img, norm=LogNorm(vmin=im0.norm.vmin, vmax=im0.norm.vmax), cmap=cmap);  ax[1].set_title(synth_label)
        
        ax_cb = fig.add_axes(rect=[.079,-.025,0.87,.02], frameon=False)
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=0)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()
        if circles:
            for j in range(len(x)): ax[1].add_patch(Circle((x[j], y[j]), radius=3*np.mean([fwhmx[j], fwhmy[j]]), fill=None, color='r', lw=3))
                
   
    synth_fits = fits.PrimaryHDU(data=synth_img, header=head)

    if plot and (savepath is not None): 
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_synth_'+str(n)+'.png', bbox_inches='tight')
    
    if simil: return synth_fits, synth_ssim
    return synth_fits