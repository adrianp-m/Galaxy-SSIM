#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:52:21 2022

@author: adrian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Circle
import cmocean as cmo
import cmasher as cma
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
import photutils as phot
import os
import sys

if __name__ == '__main__':    
    if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))
    from SSIM_utils import __compare_images
    from FITS_utils import __moments_centroid, __ellipse_fit

##### Sky values calculation
def __Sky(image, nsigma=2):
    mask = phot.make_source_mask(image, nsigma=nsigma, npixels=5, dilate_size=11) 
    mean, sky, sigma = sigma_clipped_stats(image, sigma=3.0, mask=mask) 
    return sky, sigma

##### Ellipse masking
def __ellipse_mask(img, apertures=None, plot=False):
    if apertures is None: apertures = __ellipse_fit(img)
    ell_x, ell_y = apertures.positions
    overlap = apertures.to_mask().get_overlap_slices(np.shape(img))
    ell_mask = np.array(apertures.to_mask().data[overlap[1]], dtype=bool)
    
    if plot:
        import cmocean as cmo
        fig, ax = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
        fig.suptitle('NGC0214 - g', fontsize=30, y=0.82)
        ax[0].imshow(img, norm=LogNorm(), cmap=cmo.cm.deep)
        ax[0].set_title('Original image', fontsize=24)
        apertures.plot(axes=ax[0], color='r')
        mask = np.zeros(np.shape(img))
        mask[overlap[0]] = ell_mask
        ax[1].imshow(mask)
        ax[1].set_title('Mask from fitted ellipse', fontsize=24)
        plt.savefig('/home/adrian/Desktop/prueba.png', bbox_inches='tight')
        
    masked_img = img[overlap[0]]*ell_mask
    masked_img[np.where(masked_img == 0)] = np.nan    
    return masked_img, ell_mask


##### Masking of individual star
def __mask(img, x, y, rad, sky_rad, value_fill=None, give_mask=False):
    obj_img = img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)]
    center = sky_rad

    Y, X = np.ogrid[:obj_img.shape[0], :obj_img.shape[1]]
    
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = dist_from_center <= rad
    sky_img = img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)][~mask]
    sky = np.nanmean(sky_img);  sky_std = np.std(sky_img)
    size = len(np.where(mask)[0])
    
    if value_fill: img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)][mask] = value_fill
    else: img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)][mask] = abs(np.random.normal(loc=sky, scale=sky_std, size=size))
    if give_mask: return img, mask
    return img


##### Masks the center of the galaxy
def __mask_center(gal, hdu, moments_center=False, rel_rad=0.1, simil=False, split=False, plot=False, log=True, cmap=None, SSIM_cmap=None, savepath=None):
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r

    ########## Data preparation
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']
    img = np.array(hdu.data, dtype=float);  head = hdu.header

    ### Image center
    center = int(np.shape(img)[1]/2)
    rad = int(center*rel_rad)

    ### Centroid calculation for more precision
    if moments_center: y, x = __moments_centroid(img)
    else: x, y = center, center
    masked_img = img.copy()
    masked_img = __mask(masked_img, x, y, rad, 1.1*rad)

    if simil: _, masked_ssim = __compare_images(img, masked_img, split=split, plot=plot, suptitle=gal+' - '+band, title1='Original', title2='Masked center image', cmap=cmap, SSIM_cmap=SSIM_cmap)
    if plot and not simil:
        fig, ax = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
        fig.suptitle(gal+' - '+band, fontsize=18, y=0.8)
        ax[0].set_title('Original', fontsize=18)
        if log: 
            im = ax[0].imshow(img, norm=LogNorm(), cmap=cmap)
            ax[1].imshow(img, norm=LogNorm(), cmap=cmap)
            ax[2].imshow(masked_img, norm=LogNorm(vmin=im.norm.vmin, vmax=im.norm.vmax), cmap=cmap)

        else: 
            im = ax[0].imshow(img, cmap=cmap)
            ax[1].imshow(img, cmap=cmap)
            ax[2].imshow(masked_img, vmin=im.norm.vmin, vmax=im.norm.vmax, cmap=cmap)

        ax[1].set_title('Center location', fontsize=18)
        ax[1].add_patch(Circle((y, x), radius=rad, fill=None, color='g', lw=4))

        ax[2].set_title('Masked center image', fontsize=18)
        plt.tight_layout()

    masked_fits = fits.PrimaryHDU(data=masked_img, header=head)
        
    if plot and (savepath is not None): 
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_center_masked.png', bbox_inches='tight')
    
    if simil: return masked_fits, masked_ssim
    return masked_fits


##### Masking of all stars in image
def __mask_image(gal, hdu, FWHM=6, gal_size=0.5, FWHM_factor=1, border_factor=1, nsigma=3, simil=False, split=False, plot=False, log=True, circles=False, savepath=None, cmap=None, SSIM_cmap=None):
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r

    ########## Data preparation
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']
    img = np.array(hdu.data, dtype=float);  head = hdu.header
    
    no_gal_img = img.copy()
    no_gal_img = __mask(no_gal_img, np.shape(img)[1]/2, np.shape(img)[0]/2, gal_size*np.shape(img)[1]/2, 1.05*gal_size*np.shape(img)[1]/2, value_fill=np.nan)
    
    # no_gal_img[int((1-gal_size)*np.shape(img)[0]/2):int((1+gal_size)*np.shape(img)[0]/2), \
    #             int((1-gal_size)*np.shape(img)[1]/2):int((1+gal_size)*np.shape(img)[1]/2)] = np.nan
    
    mean, median, std = sigma_clipped_stats(img)
    sky, sky_std = __Sky(no_gal_img, nsigma=3)
    
    # Buenos resultados con threshold = sky/100
    try:
        daofind = phot.DAOStarFinder(fwhm=FWHM, threshold=nsigma*std, exclude_border=False)
        stars = daofind(no_gal_img - median)
        x_stars = stars['xcentroid'].value
        y_stars = stars['ycentroid'].value
    except:
        stars = None
        # print('No stars found. Masking borders.')
    
    if plot and not simil:
        fig, ax = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
        fig.suptitle(gal+' - '+band, fontsize=18, y=0.8)
        ax[0].set_title('Original', fontsize=18)
        if log: 
            im = ax[0].imshow(img, norm=LogNorm(), cmap=cmap)
            ax[1].imshow(img, norm=LogNorm(), cmap=cmap)
        else:
            im = ax[0].imshow(img, cmap=cmap)
            ax[1].imshow(img, cmap=cmap)
        ax[1].set_title('Star detection', fontsize=18)
        # ax[1].add_patch(Rectangle((int((1-gal_size)*np.shape(img)[0]/2), int((1-gal_size)*np.shape(img)[1]/2)), \
        #                         int(2*gal_size*np.shape(img)[0]/2), int(2*gal_size*np.shape(img)[1]/2), fill=None, color='g', lw=5))
        if circles: ax[1].add_patch(Circle((np.shape(img)[1]/2, np.shape(img)[1]/2), radius=gal_size*np.shape(img)[1]/2, fill=None, color='g', lw=5))

    masked_img = img.copy()
    if stars:
        for i in range(len(x_stars)):
            if plot and circles and not simil: ax[1].add_patch(Circle((x_stars[i], y_stars[i]), radius=2*FWHM_factor*FWHM, fill=None, color='r', lw=2))
    
            if (x_stars[i] < (FWHM_factor + 1)*FWHM) or (x_stars[i] > np.shape(img)[1] + (FWHM_factor + 1)*FWHM) or \
                (y_stars[i] < (FWHM_factor + 1)*FWHM) or (y_stars[i] > np.shape(img)[0] + (FWHM_factor + 1)*FWHM): 
                    try:
                        masked_img = __mask(masked_img, y_stars[i], x_stars[i], (FWHM_factor)*0.75*FWHM, (FWHM_factor + 1)*0.75*FWHM)
                        continue
                    except:
                        continue
            
            masked_img = __mask(masked_img, y_stars[i], x_stars[i], FWHM_factor*FWHM, (FWHM_factor + 1)*FWHM)
        
    masked_img[np.r_[0:int(border_factor*FWHM), int(np.shape(img)[1]-border_factor*FWHM):np.shape(img)[1]], :] \
                = abs(np.random.normal(loc=sky, scale=sky_std, size=np.shape(masked_img[np.r_[0:int(border_factor*FWHM), int(np.shape(img)[1]-border_factor*FWHM):np.shape(img)[1]], :])))
                
    masked_img[:, np.r_[0:int(border_factor*FWHM), int(np.shape(img)[0]-border_factor*FWHM):np.shape(img)[1]]] \
                = abs(np.random.normal(loc=sky, scale=sky_std, size=np.shape(masked_img[:, np.r_[0:int(border_factor*FWHM), int(np.shape(img)[0]-border_factor*FWHM):np.shape(img)[1]]])))       
    
    if simil: _, masked_ssim = __compare_images(img, masked_img, split=split, plot=plot, log=log, suptitle=gal+' - '+band, title1='Original', title2='Masked image', cmap=cmap, SSIM_cmap=SSIM_cmap)
                 

    if plot and not simil:
        ax[2].set_title('Masked image', fontsize=18)
        if log: ax[2].imshow(masked_img, norm=LogNorm(vmin=im.norm.vmin, vmax=im.norm.vmax), cmap=cmap)
        else: ax[2].imshow(masked_img, vmin=im.norm.vmin, vmax=im.norm.vmax, cmap=cmap)
        plt.tight_layout()
    
    masked_fits = fits.PrimaryHDU(data=masked_img, header=head)

    if plot and (savepath is not None): 
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_masked.png', bbox_inches='tight')
            
    if simil: return masked_fits, masked_ssim
    return masked_fits