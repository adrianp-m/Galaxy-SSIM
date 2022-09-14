#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:43:51 2022

@author: adrian
"""

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean as cmo
import cmasher as cma
from PIL import Image
import os
import sys

if __name__ == '__main__':    
    if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))
    from FITS_utils import __image_retrieve
    from SSIM_utils import __compare_images


'''_____»_____»_____»_____»_____» Cut image «_____«_____«_____«_____«_____'''
def __image_cut(gal, hdu, size,  center=None, xsize=None, ysize=None, plot=False, log=True, cmap=None, savepath=None):
    """
    # Description
    -------------
    Cuts an image to a given size from a certain point.

    # Parameters
    ------------
    · gal      : str                           / Name of the galaxy
    · hdu      : Astropy FITS                  / FITS file containing the galaxy data
    · size     : int                           / Size of the cut region
    · center   : int 2 elements list, optional / Central coordinates of the galaxy. If not given, it is selected as the image center
    · xsize    : int, optional                 / Size of the cut in X. If None, uses size
    · ysize    : int, optional                 / Size of the cut in Y. If None, uses size
    · plot     : bool, optional                / Plot original and cut images
    · log      : bool, optional                / Use logarithmic scale for the images plot
    · cmap     : str or cmap, optional         / Colormap used for the original and cut images
    · savepath : str, optional                 / Path where to save the plot

    # Returns
    ---------
    · cut_fits : Astropy FITS / Cut FITS file
    """
    
    if cmap is None: cmap = cmo.cm.balance_r
    
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
    
    band = hdu.header['FILTER']    # The filter is retrieved
    img = np.array(hdu.data, dtype=float);  head = hdu.header  # The image and header are read
    if center is None: center = np.array(np.shape(img))/2      # If the galaxy center is not given, it is selected as the image center
    else: center = center[::-1]     # The center is inverted to match the plot axes (that are swapped)
    if xsize is None: xsize = size  # Cut size in X not given
    if ysize is None: ysize = size  # Cut size in Y not given
    cut_img = img[int(center[0]-xsize/2):int(center[0]+xsize/2), int(center[1]-ysize/2):int(center[1]+ysize/2)]  # Cut image
    

    if plot:
        cut_label = r'Cut (Center: [%i$\pm$%i, %i$\pm$%i] px)' % (center[0], xsize, center[1], ysize)
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.2))
        fig.suptitle(gal+' - '+band)
        if log:  # Logarithmic scale
            im0 = ax[0].imshow(img, norm=LogNorm(), cmap=cmap)  # Original image imshow
            ax[1].imshow(cut_img, norm=LogNorm(vmin=im0.norm.vmin, vmax=im0.norm.vmax), cmap=cmap)  # Cut image imshow
        else:    # Linear scale
            im0 = ax[0].imshow(img, cmap=cmap) 
            ax[1].imshow(cut_img, vmin=im0.norm.vmin, vmax=im0.norm.vmax, cmap=cmap)

        ax[0].set_title(r'Original (%i $\times$ %i px)' % (np.shape(img)[1], np.shape(img)[0]), fontsize=28)
        ax[1].set_title(cut_label, fontsize=28)
        
        ax_cb = fig.add_axes(rect=[.065,-.025,0.9,.02], frameon=False)  # Common colorbar for the two images
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=45)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()

    cut_fits = fits.PrimaryHDU(data=cut_img, header=head)  # Cut image FITS
    
    if plot and (savepath is not None):  # Saving the plot
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_cut.png', bbox_inches='tight')

    return cut_fits


'''_____»_____»_____»_____»_____» Rescale image «_____«_____«_____«_____«_____'''
def __image_rescale(gal, hdu, size, plot=False, log=True, cmap=None, savepath=None):
    """
    # Description
    -------------
    Rescalates an image to a given size.

    # Parameters
    ------------
    · gal      : str                   / Name of the galaxy
    · hdu      : Astropy FITS          / FITS file containing the galaxy data
    · size     : int                   / Size for the final image
    · plot     : bool, optional        / Plot original and resized images
    · log      : bool, optional        / Use logarithmic scale for the images plot
    · cmap     : str or cmap, optional / Colormap used for the original and resized images
    · savepath : str, optional         / Path where to save the plot

    # Returns
    ---------
    · res_fits : Astropy FITS / Resized FITS file
    """
    if cmap is None: cmap = cmo.cm.balance_r
    
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']  # The filter is retrieved
    img = np.array(hdu.data, dtype=float);  head = hdu.header  # The image and header are read
    res_img = Image.fromarray(img)
    res_img = np.array(res_img.resize(size=(size, size)), dtype=np.float64)  # Resized image in float array format
    
    if plot:
        res_label = r'Resized (%i $\times$ %i px)' % (size, size)
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.2))
        fig.suptitle(gal+' - '+band)
        if log:  # Logarithmic scale
            im0 = ax[0].imshow(img, norm=LogNorm(), cmap=cmap)  # Original image imshow
            ax[1].imshow(res_img, norm=LogNorm(), cmap=cmap)  # Cut image imshow
        else:    # Linear scale
            im0 = ax[0].imshow(img, cmap=cmap) 
            ax[1].imshow(res_img, cmap=cmap)
        
        ax[0].set_title(r'Original (%i $\times$ %i px)' % (np.shape(img)[1], np.shape(img)[0]), fontsize=28)
        ax[1].set_title(res_label, fontsize=28)
        
        ax_cb = fig.add_axes(rect=[.065,-.025,0.9,.02], frameon=False)  # Common colorbar for the two images
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=45)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()
    
    res_fits = fits.PrimaryHDU(data=res_img, header=head)  # Resized image FITS
    
    if plot and (savepath is not None):  # Saving the plot
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_res_'+str(2*size)+'x'+str(2*size)+'.png', bbox_inches='tight')
            
    return res_fits    



