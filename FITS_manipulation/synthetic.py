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


'''_____»_____»_____»_____»_____» Synthetic sources «_____«_____«_____«_____«_____'''
def __synthetic_sources(gal, hdu, n=None, peak=None, fwhm=None, e=None, x=None, y=None, theta=None, \
                        peak_range=[0.25,1.25], fwhm_range=[2,6], e_range=[0,0.5], theta_range=[0,360], \
                        gal_size=None, simil=False, split=False, plot=False, circles=False, cmap=None, SSIM_cmap=None, savepath=None):  
    """
    # Description
    -------------
    Generates synthetics sources in a given image with random or given parameters.

    # Parameters
    ------------
    · gal         : str                                   / Name of the galaxy
    · hdu         : Astropy FITS                          / FITS file containing the galaxy data
    · n           : int, optional                         / Number of synthetic objects
    · peak        : float or float list, optional         / Fixed peak value/s
    · fwhm        : float or float list, optional         / Fixed Full Width at Half-Maximum value/s in pixels
    · e           : float or float list, optional         / Fixed ellipticity value/s
    · x           : int or int list, optional             / Source/s X position/s
    · y           : int or int list, optional             / Source/s Y position/s
    · theta       : float or float list, optional         / Source/s orientation angle from the horizontal axis
    · peak_range  : float 2 elements list, optional       / Peak values range as a percentage of the maximum image value [min, max]
    · fwhm_range  : float 2 elements list, TYPE, optional / FWHM values range in pixels [min, max]
    · e_range     : float 2 elements list, optional       / Ellipticity values range [min, max]
    · theta_range : float 2 elements list, optional       / Orientation values range in degrees [min, max]
    · gal_size    : float, optional                       / Size of the galaxy relative to the image size so no sources are added within. If None, sources positions are completely random
    · simil       : bool, optional                        / Obtain the SSIM results between the original and source-added images
    · split       : bool, optional                        / Split the SSIM results into the three components
    · plot        : bool, optional                        / Show the image with the added sources (and SSIM results if simil is active)
    · circles     : bool, optional                        / Display circles surrounding the added sources and the galaxy region
    · cmap        : str or cmap, optional                 / Colormap used for the original and source-added images
    · SSIM_cmap   : str or cmap, optional                 / Colormap used for the SSIM-related maps
    · savepath    : str, optional                         / Path where to save the plot

    # Returns
    ---------
    * If simil:
        · synth_fits : Astropy FITS / FITS file containing the synthetic image
        · synth_ssim : int and np.array list / List containing the SSIM results
            - If not split: Contains mean SSIM and SSIM map
            - If split: Contains mean SSIM, SSIM map, 
                                 mean luminosity, luminosity map, 
                                 mean contrast, contrast map,
                                 mean structure and structure map
    * If not simil:
        · synth_fits : Astropy FITS / FITS file containing the synthetic image
    """
      
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r
    
    if n is None:  # If no number of sources is given but we have one of the quantities array
        for i in [peak, fwhm, e, x, y, theta]:
            try:
                n = len(i)
                break
            except:
                n = 1  # If there is no given array of any quantity or it is an integer, we have one object
                
    # Integer parameters are changed to a list of size 1
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
        
    band = hdu.header['FILTER']  # The filter is retrieved
    img = np.array(hdu.data, dtype=float);  head = hdu.header  # The image and header are read
    
    # The values that have not been given are random within their respective ranges
    if peak is None: peak = np.random.uniform(low=peak_range[0]*np.max(img), high=peak_range[1]*np.max(img), size=n)
    if fwhm is None: fwhm = np.random.uniform(low=fwhm_range[0], high=fwhm_range[1], size=n)
    if e is None: e = np.random.uniform(low=e_range[0], high=e_range[1], size=n)
    if gal_size and not x and not y:  # If the galaxy region is delimited, positions have to be located outside of it
        mask_image = np.ones(shape=np.shape(img))
        mask_image[int((0.5-gal_size/2)*np.shape(img)[1]):int((0.5+gal_size/2)*np.shape(img)[1]),
                   int((0.5-gal_size/2)*np.shape(img)[0]):int((0.5+gal_size/2)*np.shape(img)[0])] = 0
        
        x, y = np.where(mask_image == 1)  # Available positions outside of the selected galaxy radius
        ind = np.random.randint(low=0, high=len(x), size=n)
        x = x[ind];  y = y[ind]  # Sources positions are randomly selected from the available ones
       
    if x is None: x = np.random.randint(low=0, high=np.shape(img)[1], size=n)  # If no galaxy region has been defined, the coordinates are completely random
    if y is None: y = np.random.randint(low=0, high=np.shape(img)[0], size=n)
    if theta is None: theta = np.random.uniform(low=theta_range[0], high=theta_range[1], size=n)

    if e is not(None): fwhmx = fwhm;  fwhmy = fwhmx*(1-np.array(e))  # The ellipticity and FWHMX give the FWHMY value
    else: fwhmx = fwhm;  fwhmy = fwhm  # If no ellipticity, it is assumed to be 0 (circular sources)
    sigmax = fwhmx/(2*np.sqrt(2*np.log(2)))  # Sigma values for the Gaussian distribution
    sigmay = fwhmy/(2*np.sqrt(2*np.log(2)))

    synth_img = img.copy()  # Synthetic image where the sources will be added
    y_img, x_img = np.mgrid[0:np.shape(img)[1], 0:np.shape(img)[0]]  # Arrays of X and Y coordinates for the Gaussian distribution

    for j in range(len(x)):
        synth_img += Gaussian2D(peak[j], y[j], x[j], sigmax[j], sigmay[j], theta=theta[j]*np.pi/180)(y_img, x_img)  # Each Gaussian source is added to the synthetic image
    
    # If we have only added one object, its parameters are shown in the title. If there are more, only the total number of them is shown
    if len(x) == 1: synth_label = 'Synthetic object\n(Peak=%.2f, X=%i, Y=%i, \nFWHMX=%.2f, FWHMY=%.2f, θ=%.2fº)' % (peak[0], x[0], y[0], abs(fwhmx[0]), abs(fwhmy[0]), theta[0])
    else: synth_label = '%i Synthetic stars' % (n)
    
    # Comparison between original and source-added images if simil is active
    if simil: _, synth_ssim = __compare_images(img, synth_img, plot=plot, split=split, \
                                               suptitle=gal+' - '+band, title1='Original', title2=synth_label, cmap=cmap, SSIM_cmap=SSIM_cmap)
    if plot and not simil: 
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
        fig.suptitle(gal+' - '+band)
        im0 = ax[0].imshow(img, norm=LogNorm(), cmap=cmap);        ax[0].set_title('Original')  # Original image
        ax[1].imshow(synth_img, norm=LogNorm(vmin=im0.norm.vmin, vmax=im0.norm.vmax), cmap=cmap);  ax[1].set_title(synth_label)  # Image with added sources
        
        ax_cb = fig.add_axes(rect=[.079,-.025,0.87,.02], frameon=False)
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=0)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()
        if circles:
            for j in range(len(x)): ax[1].add_patch(Circle((x[j], y[j]), radius=3*np.mean([fwhmx[j], fwhmy[j]]), fill=None, color='r', lw=3))
                
   
    synth_fits = fits.PrimaryHDU(data=synth_img, header=head)  # Synthetic sources image FITS

    if plot and (savepath is not None): # Saving the plot
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_synth_'+str(n)+'.png', bbox_inches='tight')
    
    if simil: return synth_fits, synth_ssim
    return synth_fits