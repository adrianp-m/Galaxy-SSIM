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


'''_____»_____»_____»_____»_____» Sky calculation «_____«_____«_____«_____«_____'''
def __Sky(img, nsigma=2):
    """
    # Description
    -------------
    Calculates the sky value and standard deviation from a given image.

    # Parameters
    ------------
    · img    : float np.array  / Image where to measure the sky
    · nsigma : float, optional / Sigma factor for the sources search in the image

    # Returns
    ---------
    · sky   : float / Sky median value
    · sigma : float / Sky standard deviation
    """
    mask = phot.make_source_mask(img, nsigma=nsigma, npixels=5, dilate_size=11)  # sources are masked
    mean, sky, sigma = sigma_clipped_stats(img, sigma=3.0, mask=mask)            # The mean, median and std of the sky are calculated
    return sky, sigma


'''_____»_____»_____»_____»_____» Single source masking «_____«_____«_____«_____«_____'''
def __mask(img, x, y, rad, sky_rad, value_fill=None, give_mask=False):
    """
    # Description
    -------------
    Masks a source centered in a given image.

    # Parameters
    ------------
    · img        : float np.array     / Full image where the source is contained
    · x          : int                / Source X position
    · y          : int                / Source Y position
    · rad        : float              / Source radius
    · sky_rad    : float              / Source image to cut, containing source and a sky annulus
    · value_fill : Any type, optional / Value to fill the mask with (e.g. nan or 0 values for a strong mask). If None, the object is masked with values from a gaussian distribution centered in the sky value
    · give_mask  : bool, optional     / Return the used mask

    # Returns
    ---------
    * If give_mask:
        · img  : float np.array / Masked image
        · mask : bool np.array  / Mask
    
    * If not give_mask:
        · img : float np.array / Masked image
    """
    obj_img = img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)]  # Source contained in the image
    center = sky_rad  # Center of the image

    Y, X = np.ogrid[:obj_img.shape[0], :obj_img.shape[1]]  # Coordinates arrays
    
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)  # Distance from the center of the image to delimit the source within a certain radius
    mask = dist_from_center <= rad  # Source pixels
    sky_img = img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)][~mask]  # Sky pixels
    sky = np.nanmedian(sky_img);  sky_std = np.std(sky_img)  # Sky values
    size = len(np.where(mask)[0])   # Size of the mask 
    
    if value_fill: img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)][mask] = value_fill  # All values are filled to the given one (could be nan)
    else: img[int(x-sky_rad):int(x+sky_rad), int(y-sky_rad):int(y+sky_rad)][mask] = abs(np.random.normal(loc=sky, scale=sky_std, size=size))  # Masking to a gaussian distribution
    if give_mask: return img, mask
    return img


'''_____»_____»_____»_____»_____» Ellipse masking «_____«_____«_____«_____«_____'''
def __ellipse_mask(img, apertures=None, plot=False, give_mask=False):
    """
    # Description
    -------------
    Performs an ellipse masking.

    # Parameters
    ------------
    · img       : float np.array                         / Image where to apply the mask
    · apertures : photutils EllipticalAperture, optional / Photutils' object containing the elliptical distribution parameters. If None, this is calculated for default aperture radius
    · plot      : bool, optional                         / Plot image and elliptical mask
    · give_mask : bool, optional                         / Return the used mask

    # Returns
    ---------
    * If give_mask:
        · masked_img : float np.array / Image of elliptical shape after the masking. Values outside of the ellipse are nan
        · ell_mask   : bool np.array  / Ellipse mask
    
    * If not give_mask:
        · masked_img : float np.array / Image of elliptical shape after the masking. Values outside of the ellipse are nan
    
    """
    if apertures is None: apertures = __ellipse_fit(img)  # If the ellipse parameters are not given, they are calculated
    ell_x, ell_y = apertures.positions  # Center of the ellipse
    overlap = apertures.to_mask().get_overlap_slices(np.shape(img))        # Values of the ellipse mask that lay inside of the image
    ell_mask = np.array(apertures.to_mask().data[overlap[1]], dtype=bool)  # Boolean mask
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
        fig.suptitle('NGC0214 - g', fontsize=30, y=0.82)
        ax[0].imshow(img, norm=LogNorm())  # Galaxy image with mask imshow
        ax[0].set_title('Original image', fontsize=24)
        apertures.plot(axes=ax[0], color='r')
        mask = np.zeros(np.shape(img))  # Mask with the same size as the image
        mask[overlap[0]] = ell_mask     # The ellipse mask region is added to the full mask
        ax[1].imshow(mask)              # Mask imshow
        ax[1].set_title('Mask from fitted ellipse', fontsize=24)
        plt.savefig('/home/adrian/Desktop/prueba.png', bbox_inches='tight')
        
    masked_img = img[overlap[0]]*ell_mask           # Masked image
    masked_img[np.where(masked_img == 0)] = np.nan  # Values outside of the mask are set as nan
    
    if give_mask: return masked_img, ell_mask
    return masked_img


'''_____»_____»_____»_____»_____» Center masking «_____«_____«_____«_____«_____'''
def __mask_center(gal, hdu, moments_center=False, rel_rad=0.1, simil=False, split=False, plot=False, log=True, cmap=None, SSIM_cmap=None, savepath=None):
    """
    # Description
    -------------
    Masks the center of the image.

    # Parameters
    ------------
    · gal            : str                   / Name of the galaxy
    · hdu            : Astropy FITS          / FITS file containing the galaxy data
    · moments_center : bool, optional        / Recalculate the center using moments
    · rel_rad        : flat, optional        / Size of the central mask relative to the full image size
    · simil          : bool, optional        / Obtain the SSIM results between the original and masked images
    · split          : bool, optional        / Split the SSIM results into the three components
    · plot           : bool, optional        / Show the masked image (and SSIM results if simil is active)
    · log            : bool, optional        / Use logarithmic scale for the plots
    · cmap           : str or cmap, optional / Colormap used for the original and masked images
    · SSIM_cmap      : str or cmap, optional / Colormap used for the SSIM-related maps
    · savepath       : str, optional         / Path where to save the plot

    # Returns
    ---------
    * If simil:
        · masked_fits : float np.array        / Masked FITS file
        · masked_ssim : int and np.array list / List containing the SSIM results
            - If not split: Contains mean SSIM and SSIM map
            - If split: Contains mean SSIM, SSIM map, 
                                 mean luminosity, luminosity map, 
                                 mean contrast, contrast map,
                                 mean structure and structure map
                                 
    * If not simil:
        · masked_fits : float np.array / Masked FITS file

    """
    
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r

    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']  # The filter is retrieved
    img = np.array(hdu.data, dtype=float);  head = hdu.header  # The image and header are read

    center = int(np.shape(img)[1]/2)  # Center of the image
    rad = int(center*rel_rad)         # Radius from the center to mask

    if moments_center: y, x = __moments_centroid(img)    # Center centroid calculation
    else: x, y = center, center                          # If not recalculated, image center is selected
    masked_img = img.copy()
    masked_img = __mask(masked_img, x, y, rad, 1.1*rad)  # Center of the image is masked

    # SSIM is calculated between original and center-masked image if simil is activated
    if simil: _, masked_ssim = __compare_images(img, masked_img, split=split, plot=plot, suptitle=gal+' - '+band, title1='Original', title2='Masked center image', cmap=cmap, SSIM_cmap=SSIM_cmap)
    if plot and not simil:
        fig, ax = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
        fig.suptitle(gal+' - '+band, fontsize=18, y=0.8)
        ax[0].set_title('Original', fontsize=18)
        if log: # Logarithmic scale plots
            im = ax[0].imshow(img, norm=LogNorm(), cmap=cmap)  # Original image
            ax[1].imshow(img, norm=LogNorm(), cmap=cmap)       # Original image with the center later marked
            ax[2].imshow(masked_img, norm=LogNorm(vmin=im.norm.vmin, vmax=im.norm.vmax), cmap=cmap)  # Masked image

        else:   # Linear scale plots
            im = ax[0].imshow(img, cmap=cmap)
            ax[1].imshow(img, cmap=cmap)
            ax[2].imshow(masked_img, vmin=im.norm.vmin, vmax=im.norm.vmax, cmap=cmap)

        ax[1].set_title('Center location', fontsize=18)
        ax[1].add_patch(Circle((y, x), radius=rad, fill=None, color='g', lw=4))  # Center region circle is added

        ax[2].set_title('Masked center image', fontsize=18)
        plt.tight_layout()

    masked_fits = fits.PrimaryHDU(data=masked_img, header=head)  # Masked image FITS
        
    if plot and (savepath is not None):  # Saving the plot
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_center_masked.png', bbox_inches='tight')
    
    if simil: return masked_fits, masked_ssim
    return masked_fits


'''_____»_____»_____»_____»_____» Masking all image sources «_____«_____«_____«_____«_____'''
def __mask_sources(gal, hdu, FWHM=6, gal_size=0.5, FWHM_factor=1, border_factor=1, nsigma=3, simil=False, split=False, 
                   plot=False, log=True, circles=False, cmap=None, SSIM_cmap=None, savepath=None):
    """
    # Description
    -------------
    Finds and masks all the sources external to the galaxy in a given image.

    # Parameters
    ------------
    · gal           : str                   / Name of the galaxy
    · hdu           : Astropy FITS          / FITS file containing the galaxy data
    · FWHM          : float, optional       / Seed value of the Full Width at Half-Maximum in pixels to find sources in the image
    · gal_size      : float, optional       / Relative size of the galaxy in the image
    · FWHM_factor   : float, optional       / Factor applied to the FWHM to select each source cut region for its masking
    · border_factor : float, optional       / Factor applied to the FWHM to select the border masking width
    · nsigma        : float, optional       / Sigma factor for the sources search in the image
    · simil         : bool, optional        / Obtain the SSIM results between the original and masked images
    · split         : bool, optional        / Split the SSIM results into the three components
    · plot          : bool, optional        / Show the masked image (and SSIM results if simil is active)
    · log           : bool, optional        / Use logarithmic scale for the plots
    · circles       : bool, optional        / Draw circles around the found sources
    · cmap          : str or cmap, optional / Colormap used for the reference and compared images
    · SSIM_cmap     : str or cmap, optional / Colormap used for the SSIM-related maps
    · savepath      : str, optional         / Path where to save the plot

    # Returns
    ---------
    * If simil:
        · masked_fits : float np.array / Masked FITS file
        · masked_ssim : int and np.array list / List containing the SSIM results
            - If not split: Contains mean SSIM and SSIM map
            - If split: Contains mean SSIM, SSIM map, 
                                 mean luminosity, luminosity map, 
                                 mean contrast, contrast map,
                                 mean structure and structure map
                                 
    * If not simil:
        · masked_fits : float np.array / Masked FITS file
    """
    
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r

    ########## Data preparation
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']  # The filter is retrieved
    img = np.array(hdu.data, dtype=float);  head = hdu.header  # The image and header are read
    
    no_gal_img = img.copy()  # Image with the galaxy masked in order to only find sources outside of this region
    no_gal_img = __mask(no_gal_img, np.shape(img)[1]/2, np.shape(img)[0]/2, gal_size*np.shape(img)[1]/2, 1.05*gal_size*np.shape(img)[1]/2, value_fill=np.nan)
    
    _, median, std = sigma_clipped_stats(img)   # Image values
    sky, sky_std = __Sky(no_gal_img, nsigma=3)  # Sky values
    
    try:
        daofind = phot.DAOStarFinder(fwhm=FWHM, threshold=nsigma*std, exclude_border=False)  # Search for the given FWHM and threshold parameters
        stars = daofind(no_gal_img - median)  # Searching sources in the image
        x_stars = stars['xcentroid'].value    # Sources coordinates
        y_stars = stars['ycentroid'].value
    except:
        stars = None  # No found stars
    
    if plot and not simil:
        fig, ax = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
        fig.suptitle(gal+' - '+band, fontsize=18, y=0.8)
        ax[0].set_title('Original', fontsize=18)
        if log: 
            im = ax[0].imshow(img, norm=LogNorm(), cmap=cmap)  # Original image
            ax[1].imshow(img, norm=LogNorm(), cmap=cmap)
        else:
            im = ax[0].imshow(img, cmap=cmap)
            ax[1].imshow(img, cmap=cmap)  # Image with detected sources
        ax[1].set_title('Star detection', fontsize=18)
        
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
            
            masked_img = __mask(masked_img, y_stars[i], x_stars[i], FWHM_factor*FWHM, (FWHM_factor + 1)*FWHM)  # Individual star masking
    
    # Border masking
    masked_img[np.r_[0:int(border_factor*FWHM), int(np.shape(img)[1]-border_factor*FWHM):np.shape(img)[1]], :] \
                = abs(np.random.normal(loc=sky, scale=sky_std, size=np.shape(masked_img[np.r_[0:int(border_factor*FWHM), int(np.shape(img)[1]-border_factor*FWHM):np.shape(img)[1]], :])))
                
    masked_img[:, np.r_[0:int(border_factor*FWHM), int(np.shape(img)[0]-border_factor*FWHM):np.shape(img)[1]]] \
                = abs(np.random.normal(loc=sky, scale=sky_std, size=np.shape(masked_img[:, np.r_[0:int(border_factor*FWHM), int(np.shape(img)[0]-border_factor*FWHM):np.shape(img)[1]]])))       
    # Border masking ends
    
    # SSIM results if simil is activaded    
    if simil: _, masked_ssim = __compare_images(img, masked_img, split=split, plot=plot, log=log, suptitle=gal+' - '+band, title1='Original', title2='Masked image', cmap=cmap, SSIM_cmap=SSIM_cmap)

    if plot and not simil:
        ax[2].set_title('Masked image', fontsize=18)
        if log: ax[2].imshow(masked_img, norm=LogNorm(vmin=im.norm.vmin, vmax=im.norm.vmax), cmap=cmap)  # Masked image
        else: ax[2].imshow(masked_img, vmin=im.norm.vmin, vmax=im.norm.vmax, cmap=cmap)
        plt.tight_layout()
    
    masked_fits = fits.PrimaryHDU(data=masked_img, header=head)  # Masked image FITS

    if plot and (savepath is not None):  # Saving the plot
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_masked.png', bbox_inches='tight')
            
    if simil: return masked_fits, masked_ssim
    return masked_fits