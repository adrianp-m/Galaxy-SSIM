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


##### Cut of image
def __image_cut(gal, hdu, size,  center=None, xsize=None, ysize=None, plot=False, log=True, cmap=None, savepath=None):
    if cmap is None: cmap = cmo.cm.balance_r
    
    ########## Data preparation
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
    
    band = hdu.header['FILTER']
    img = np.array(hdu.data, dtype=float);  head = hdu.header
    if center is None: center = np.array(np.shape(img))/2
    else: center = center[::-1]
    if xsize is None: xsize = size
    if ysize is None: ysize = size
    cut_img = img[int(center[0]-xsize/2):int(center[0]+xsize/2), int(center[1]-ysize/2):int(center[1]+ysize/2)]
    
    cut_label = 'Cutted (Center: [%i$\pm$%i, %i\pm$%i])' % (center[0], xsize, center[1], ysize)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
        fig.suptitle(gal+' - '+band)
        if log:
            im0 = ax[0].imshow(img, norm=LogNorm(), cmap=cmap) 
            ax[1].imshow(cut_img, norm=LogNorm(vmin=im0.norm.vmin, vmax=im0.norm.vmax), cmap=cmap)
        else:
            im0 = ax[0].imshow(img, cmap=cmap) 
            ax[1].imshow(cut_img, vmin=im0.norm.vmin, vmax=im0.norm.vmax, cmap=cmap)
        
        ax[0].set_title('Original', fontsize=28)
        ax[1].set_title(cut_label, fontsize=28)
        
        ax_cb = fig.add_axes(rect=[.065,-.025,0.9,.02], frameon=False)
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=45)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()

    cut_fits = fits.PrimaryHDU(data=cut_img, header=head)
    
    if plot and (savepath is not None): 
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_cut.png', bbox_inches='tight')

    return cut_fits

##### Resizing of image
def __image_rescale(gal, hdu, size, simil=False, split=True, plot=False, savepath=None, cmap=None, SSIM_cmap=None):
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r
    
    ########## Data preparation    
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']    
    img = np.array(hdu.data, dtype=float);  head = hdu.header
    res_img = Image.fromarray(img)
    res_img = np.array(res_img.resize(size=(size, size)), dtype=np.float64)
    
    if plot:
        res_label = 'Cutted (%ix%i)' % (2*size, 2*size)
        if simil: _, res_ssim = __compare_images(img, res_img, plot=True, \
                                                        gradient=True, suptitle=gal+' - '+band, title1='Original', title2=res_label, cmap=cmap)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
            fig.suptitle(gal+' - '+band)
            ax[0].imshow(img, norm=LogNorm());      ax[0].set_title('Original')
            ax[1].imshow(res_img, norm=LogNorm());  ax[1].set_title(res_label)
                            
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()
            
    else:
        if simil: _, res_ssim = __compare_images(img, res_img, split=split)
                
    res_fits = fits.PrimaryHDU(data=res_img, header=head)
    
    if plot and (savepath is not None):
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_res_'+str(2*size)+'x'+str(2*size)+'.png', bbox_inches='tight')
            
    if simil: return res_fits, res_ssim
    return res_fits    



