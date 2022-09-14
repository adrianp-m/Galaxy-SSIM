#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:56:07 2022

@author: adrian
"""

import numpy as np
from photutils.morphology import data_properties
from photutils.aperture import EllipticalAperture
import astropy.io.fits as fits
import astropy.units as u
import os
from PIL import Image
from masking import __mask_sources
from rescale import __image_cut, __image_rescale


'''_____»_____»_____»_____»_____» Image retrieve «_____«_____«_____«_____«_____'''
def __image_retrieve(gal, filespath, band=None, size=None, cut=None, non_neg=True, 
                     mask=False, norm=False, log=False, unpack=False):
    """
    # Description
    The image/s of the selected filter/s for a given galaxy is/are retrieved with
    different tools available, like resizing, cutting, masking, normalizing or 
    rescaling the image to logarithm.

    # Parameters
    ------------
    · gal       : str            / Name of the galaxy
    · filespath : str            / Path where the files are located
    · band      : str, optional  / Selected filter
    · size      : int, optional  / Change the size of the image to this one
    · cut       : int, optional  / Cut the image from the center to this size
    · non_neg   : bool, optional / Add the absolute value of the lowest signal to the image to avoid negative values
    · mask      : bool, optional / Search for stars in the image and mask them
    · norm      : bool, optional / Normalize the image to its maximum value (uses the masked image to find it)
    · log       : bool, optional / Apply logarithmic scale to the image
    · unpack    : bool, optional / Separate data and header in the output

    # Returns
    ---------
    * If unpack:
        · hdu_data   : float np.array / Image data values
        · hdu_header : Astropy Header / Image header
        
    * If not unpack:
        · hdu : Astropy FITS / Galaxy FITS containing image data and header
    """
    if band and np.shape(band) == ():  # If one single filter is given
        file = [f for f in os.listdir(filespath+band+'/') if gal in f]  # Find images of that filter, previously separated by folder
        if file == []:  # If no files have been found
            print('Filter %s hasn\'t been found for galaxy %s.' % (band, gal))
            if unpack: return None, None
            else: return None

        hdu = fits.open(filespath+band+'/'+file[0])[0]  # The found image for the galaxy is opened
        
        if size and hdu.header['NAXIS1'] != size: hdu = __image_rescale(gal, hdu, size)  # If a size has been given, the image is resized to it
        
        if cut and cut < size: hdu = __image_cut(gal, hdu, cut)  # If the cut option is active, the image is cutted from the center to the given size
        if non_neg: hdu.data[np.where(hdu.data < 0)] = 0         # If the non negative option is active, the maximum value of the lowest quantity is added
        if mask and not cut:  # The mask is applied if this option is active the galaxy has not been cutted to the galaxy region
            hdu = __mask_sources(gal, hdu, gal_size=0.75, FWHM=6, FWHM_factor=1, border_factor=1)
            if norm: hdu.data, _ = __normalization(hdu.data)     # If normalization is active
            
        # If normalization is active and no masking has been performed
        elif not mask and norm: _, maximum = __normalization(__mask_sources(gal, hdu, gal_size=0.75, FWHM=6, FWHM_factor=1, border_factor=1).data);  hdu.data=hdu.data/maximum
        if not norm and log: hdu.data = np.log10(hdu.data + abs(np.min(hdu.data)-1))  # If logarithmic rescalation is active and the image has not been normalized

        if unpack: return hdu.data, hdu.header  # If the unpack option is active, the output separates data and header
        else: return hdu                        # If the unpack option is deactivated, the full FITS object is given 
    
    bands = {'u', 'g', 'r', 'i', 'z'}  # All the SDSS filters
    if unpack: img_fits = {};  hdr_fits = {}  # Data and header dictionaries
    else: files_fits = {}                     # Full FITS dictionary
    
    for b in bands:  # For each given filter (or all if no filter has been given)
        if band and b not in band: continue
        file = [f for f in os.listdir(filespath+b+'/') if gal in f]  # Filter images are searched
        if file == []:  # If no files have been found
            print('Filter %s hasn\'t been found for galaxy %s.' % (b, gal))
            continue
        
        hdu = fits.open(filespath+b+'/'+file[0])[0]  # The found image for the galaxy is opened
        
        if size and hdu.header['NAXIS1'] != size: hdu = __image_rescale(gal, hdu, size)  # If a size has been given, the image is resized to it
        
        if cut and cut < size: hdu = __image_cut(gal, hdu, cut)  # If the cut option is active, the image is cutted from the center to the given size
        if non_neg: hdu.data[np.where(hdu.data < 0)] = 0         # If the non negative option is active, the maximum value of the lowest quantity is added
        if mask and not cut:  # The mask is applied if this option is active the galaxy has not been cutted to the galaxy region
            hdu = __mask_sources(gal, hdu, gal_size=0.75, FWHM=6, FWHM_factor=1, border_factor=1)
            if norm: hdu.data, _ = __normalization(hdu.data)     # If normalization is active
            
        # If normalization is active and no masking has been performed
        elif not mask and norm: _, maximum = __normalization(__mask_sources(gal, hdu, gal_size=0.75, FWHM=6, FWHM_factor=1, border_factor=1).data);  hdu.data=hdu.data/maximum
        if not norm and log: hdu.data = np.log10(hdu.data + abs(np.min(hdu.data)-1))  # If logarithmic rescalation is active and the image has not been normalized

        if unpack: img_fits[b] = hdu.data;  hdr_fits[b] = hdu.header  # Values are added to dictionaries
        else: files_fits[b] = hdu

    if unpack: return img_fits, hdr_fits  # Dictionaries are returned
    else: return files_fits

'''_____»_____»_____»_____»_____» Centroid recalculation «_____«_____«_____«_____«_____'''
def __moments_centroid(img):
    """
    # Description
    -------------
    Finds the center of a given image from the moments analysis.

    # Parameters
    ------------
    · img : float np.array / Image where the center has to be found

    # Returns
    ---------
    · Xc : int / Central X pixel of the image
    · Yc : int / Centray Y pixel of the image
    """
    
    ax,ay = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))  # The position arrays for X and Y
    X, Y, img = ax.ravel(), ay.ravel(), img.ravel()                        # The arrays are reshaped to 1D
    
    M00 = np.sum(img)       # Zero order moment
    M10 = np.sum(X*img)     # First order moment for X
    M01 = np.sum(Y*img)     # First order moment for Y
    
    Xc = M10/M00;  Yc = M01/M00  # Centroid calculation
    
    # Mxx = np.sum((X-Xc)*(X-Xc)*img)/M00  # Second order moments (not used)
    # Mxy = np.sum((X-Xc)*(Y-Yc)*img)/M00
    # Myy = np.sum((Y-Yc)*(Y-Yc)*img)/M00
    return Xc, Yc


'''_____»_____»_____»_____»_____» Ellipse fitting «_____«_____«_____«_____«_____'''
def __ellipse_fit(img, r=2.5):
    """
    # Description
    -------------
    Finds the elliptical distribution of a given image with a certain aperture radius.

    # Parameters
    ------------
    · img : float np.array  / Image where the elliptical distribution is calculated
    · r   : float, optional / Radius for the initial guess of the aperture

    # Returns
    ---------
    · apertures : photutils EllipticalAperture / Photutils' object containing the found elliptical distribution parameters

    """
    cat = data_properties(img)           # Basic properties obtained from the image
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma', 'semiminor_sigma', 'orientation']  # Desired parameters to obtain from the image properties
    tbl = cat.to_table(columns=columns)  # Table of ellipse properties

    position = (cat.xcentroid, cat.ycentroid)  # Central position of the ellipse
    a = cat.semimajor_sigma.value*r            # Major semiaxis of the ellipse
    b = cat.semiminor_sigma.value*r            # Minor semiaxis of the ellipse

    theta = cat.orientation.to(u.rad).value    # Orientation of the ellipse
    apertures = EllipticalAperture(position, a, b, theta=theta)  # Ellipse object containing all its information
    return apertures    


'''_____»_____»_____»_____»_____» Simple axis «_____«_____«_____«_____«_____'''
def __simpleaxis(ax):
    """
    # Description
    -------------
    Simplifies a figure axis by rermoving the borders' lines and labels.

    # Parameters
    ------------
    · ax : matplotlib AxesSubplot / Axis to simplify

    # Returns
    ---------
    None

    """
    ax.spines['top'].set_visible(False)     # Hiding the top, bottom, left and right line borders
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)  # Hiding all labels


'''_____»_____»_____»_____»_____» Image normalization «_____«_____«_____«_____«_____'''
def __normalization(img, gal_size=0.5, give_max=False):
    """
    # Description
    -------------
    Normalizes a given image to its maximum found value.

    # Parameters
    ----------
    · img      : float np.array  / Image to normalize
    · gal_size : float, optional / Percentage of the image considered as the galaxy (to not search sources within)
    · give_max : bool, optional  / Return the maximum found value along with the normalized image

    # Returns
    ---------
    * If give_max:
        · norm_img : float np.array / Normalized image 
        · maximum  : float          / Maximum found value
    
    * If not give_max:
        · norm_img : float np.array / Normalized image
    """
    gal_img = img[int((0.5-gal_size/2)*np.shape(img)[1]):int((0.5+gal_size/2)*np.shape(img)[1]),  # The image is cutted to the central region 
                  int((0.5-gal_size/2)*np.shape(img)[0]):int((0.5+gal_size/2)*np.shape(img)[0])]  # to find the maximum value of the galaxy
    
    maximum = np.max(gal_img)  # The maximum value is found
    if give_max: return img/maximum, maximum  # The full image is normalized to the found value and returned
    return img/maximum
    


