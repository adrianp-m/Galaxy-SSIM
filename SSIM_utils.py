3#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:46:24 2022

@author: adrian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean as cmo
import cmasher as cma
import astropy.io.fits as fits
import importlib
import skimage.metrics as metrics
metrics = importlib.reload(metrics)
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage import data, img_as_float

import os
import sys

sys.path.append('./FITS_manipulation')
from FITS_utils import __simpleaxis, __image_retrieve
     

'''_____»_____»_____»_____»_____» __compare_images «_____«_____«_____«_____«_____'''
def __compare_images(img1, img2, win_size=None, gaussian_weights=True, plot=False, log=True, decimals=4, suptitle='', title1='Image 1', title2='Image 2', cmap=None, SSIM_cmap=None, filename=None, 
                     alpha=1, beta=1, gamma=1, split=False):
    """
    # Description
    -------------
    The MSE and SSIM index are calculated for two given non-negative images of the same shape. The SSIM
    components can be obtained separately and their weights can be changed. A plot showing the two images 
    and the results can be performed

    # Parameters
    ------------
    · img1             : float np.array        / Reference image of non-negative values
    · img2             : float np.array        / Compared image of non-negative values
    · win_size         : int, optional         / Size of the used window for each local comparison
    · gaussian_weights : bool, optional        / Use a gaussian weighted window
    · plot             : bool, optional        / Perform plot of the results
    · log              : bool, optional        / Use logarithmic scale for the images imshow
    · decimals         : int, optional         / Number of decimals to use in plots' labels
    · suptitle         : str, optional         / Superior title of the plot
    · title1           : str, optional         / Title of the reference image
    · title2           : str, optional         / Title of the compared image
    · cmap             : str or cmap, optional / Colormap used for the reference and compared images
    · SSIM_cmap        : str or cmap, optional / Colormap used for the SSIM-related maps
    · filename         : str, optional         / Name of the saved plot file
    · alpha            : float, optional       / Alpha non-negative superindex (luminosity component)
    · beta             : float, optional       / Beta non-negative superindex (contrast component)
    · gamma            : float, optional       / Gamma non-negative superindex (structure component)
    · split            : bool, optional        / Obtain each SSIM component result

    # Returns
    ---------
    · mse    : float                 / Value of the Mean Squared Error
    · ssimil : int and np.array list / List containing the SSIM results
        * If not split: Contains mean SSIM and SSIM map 
        * If split    : Contains mean SSIM, SSIM map, 
                                 mean luminosity, luminosity map,
                                 mean contrast, contrast map,
                                 mean structure and structure map
    """
    
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r
    
    mse = np.nanmean((img1 - img2) ** 2, dtype=np.float64)  # Mean Squared Error calculation
    
    # Calculation of the SSIM quantities
    ssimil = ssim(img1, img2, win_size=win_size, gaussian_weights=gaussian_weights, alpha=alpha, beta=beta, gamma=gamma, split=split)
 
    ssimil_label = ssimil[0]  # Mean SSIM index value for labels

    # Superior title of figure
    if suptitle != '': suptitle = suptitle + ' / MSE = {:.{decimals}f}, SSIM = {:.{decimals}f}'.format(mse, ssimil_label, decimals=decimals)
    else: suptitle = 'MSE = {:.{decimals}f}, SSIM = {:.{decimals}f}'.format(mse, ssimil_label, decimals=decimals)

    if plot:
        suptitlesize=28;  titlesize=24;  labelsize=22;  ticksize=20  # Plot parameters
        
        if split:  # If components are obtained
            fig, ax = plt.subplots(2, 4, figsize=(24, 16.5), sharey=True, constrained_layout=True)
            fig.delaxes(ax[0, 0]);  fig.delaxes(ax[0, 3])

            if log:  # Logarithmic scale plots
                im1 = ax[0, 1].imshow(img1, norm=LogNorm(), cmap=cmap)  # Reference image
                ax[0, 2].imshow(img2, norm=LogNorm(vmin=im1.norm.vmin, vmax=im1.norm.vmax), cmap=cmap)  # Compared image

            else:    # Linear scale plots
                im1 = ax[0, 1].imshow(img1, cmap=cmap)
                ax[0, 2].imshow(img2, vmin=im1.norm.vmin, vmax=im1.norm.vmax, cmap=cmap)
    
            ax[0, 1].set_title(title1, fontsize=titlesize)
            ax[0, 2].set_title(title2, fontsize=titlesize)

            ax01_cb = fig.add_axes(rect=[.271,.512,0.479,.01], frameon=False)  # Common colorbar for the two images
            cbar01 = fig.colorbar(im1, orientation='horizontal', cax=ax01_cb, format='%.1e')
            cbar01.ax.tick_params(rotation=45)
            
            # SSIM map
            ax[1, 0].set_title('SSIM map (%.4f)' % (ssimil[0]), fontsize=titlesize)
            SSIM_map = ax[1, 0].imshow(ssimil[1], vmin=0, vmax=1, cmap=SSIM_cmap)  
            ax_cb  = fig.add_axes(rect=[.0268, .038, 0.9670, .01], frameon=False)
            cbar_full = fig.colorbar(SSIM_map, orientation='horizontal', cax=ax_cb, format='%.2f')            
            cbar_full.ax.tick_params(rotation=45, labelsize=22)

            # Luminosity map
            ax[1, 1].set_title(u'Luminosity map (%.4f)\nα=%.4f' % (ssimil[2], alpha), fontsize=titlesize)
            ax[1, 1].imshow(ssimil[3], vmin=0, vmax=1, cmap=SSIM_cmap)
            
            # Contrast map
            ax[1, 2].set_title(u'Contrast map (%.4f)\nβ=%.4f' % (ssimil[4], beta), fontsize=titlesize)
            ax[1, 2].imshow(ssimil[5], vmin=0, vmax=1, cmap=SSIM_cmap)
            
            # Structure map
            ax[1, 3].set_title(u'Structure map (%.4f)\nγ=%.4f' % (ssimil[6], gamma), fontsize=titlesize)
            ax[1, 3].imshow(ssimil[7], vmin=0, vmax=1, cmap=SSIM_cmap)
            
        else:  # Only SSIM map
            fig, ax = plt.subplots(1, 3, figsize=(16, 8.2), sharey=True, constrained_layout=True)
            ax01_cb = fig.add_axes(rect=[.042,.1,0.6285,.01], frameon=False)
            ax2_cb = fig.add_axes(rect=[.684,.1,0.306,.01], frameon=False)
            
            if log: # Logarithmic scale plots
                im1 = ax[0].imshow(img1, norm=LogNorm(), cmap=cmap)
                ax[1].imshow(img2, norm=LogNorm(vmin=im1.norm.vmin, vmax=im1.norm.vmax), cmap=cmap)

            else:   # Linear scale plots
                im1 = ax[0].imshow(img1, cmap=cmap)
                ax[1].imshow(img2, vmin=im1.norm.vmin, vmax=im1.norm.vmax, cmap=cmap)
            
            ax[0].set_title(title1, fontsize=titlesize)
            ax[1].set_title(title2, fontsize=titlesize)
            
            # SSIM map
            im = ax[2].imshow(ssimil[1], vmin=0, vmax=1, cmap=SSIM_cmap)
            cbar = fig.colorbar(im, orientation='horizontal', cax=ax2_cb)            
            cbar.ax.tick_params(rotation=45)
            ax[2].set_title('SSIM map', fontsize=titlesize)


        cbar0 = fig.colorbar(im1, orientation='horizontal', cax=ax01_cb)        
        cbar0.ax.tick_params(rotation=45)
            
        fig.suptitle(suptitle, fontsize=suptitlesize) # , y=0.89
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()
    if plot and filename: # Figure save
        filename = filename.replace('"SSIM"', '%.2f' % (ssimil[0]))
        if not os.path.isdir(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename))
        plt.savefig(filename+'.png', bbox_inches='tight')
    return mse, ssimil    