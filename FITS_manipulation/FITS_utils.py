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
from masking import __mask_image
from rescale import __image_cut, __image_rescale


'''_____»_____»_____»_____»_____» Image retrieve «_____«_____«_____«_____«_____'''
def __image_retrieve(gal, filespath, band=None, size=None, cut=None, non_neg=True, 
                     mask=False, norm=False, log=False, unpack=False):
    """
    # Description
    The image/s of the selected filter/s for a given galaxy is/are retrieved with
    different tools available, like resizing, cutting, masking, normalizing or 
    rescaling the image to logarithm

    # Parameters
    ------------
    · gal       : str            / Galaxy name 
    · filespath : str            / Path where the files are located
    · band      : str, optional  / Selected filter
    · size      : int, optional  / Change the size of the image to this one
    · cut       : int, optional  / Cut the image from the center to this size
    · non_neg   : bool, optional / Add the absolute value of the lowest signal to the image to avoid negative values
    · mask      : bool, optional / Search for stars in the image and mask them
    · norm      : bool, optional / Normalize the image to its maximum value
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
    if band and np.shape(band) == ():
        file = [f for f in os.listdir(filespath+band+'/') if gal in f]
        if file == []: 
            print('Filter %s hasn\'t been found for galaxy %s.' % (band, gal))
            if unpack: return None, None
            else: return None

        hdu = fits.open(filespath+band+'/'+file[0])[0]
        
        if size and hdu.header['NAXIS1'] != size: hdu = __image_rescale(gal, hdu, size)
            # img = Image.fromarray(hdu.data)
            # img = np.array(img.resize(size=(size, size)), dtype=np.float64)
            # hdu.data = img
        
        if cut and cut < size: hdu = __image_cut(gal, hdu, cut)
        if non_neg: hdu.data[np.where(hdu.data < 0)] = 0
        if mask and not cut: 
            hdu = __mask_image(gal, hdu, gal_size=0.75, FWHM=6, FWHM_factor=1, border_factor=1)
            if norm: hdu.data, _ = __normalization(hdu.data)
        elif not mask and norm: _, maximum = __normalization(__mask_image(gal, hdu, gal_size=0.75, FWHM=6, FWHM_factor=1, border_factor=1).data);  hdu.data=hdu.data/maximum
        if not norm and log: hdu.data = np.log10(hdu.data + abs(np.min(hdu.data)-1))

        if unpack: return hdu.data, hdu.header
        else: return hdu
    
    bands = {'u', 'g', 'r', 'i', 'z'}
    if unpack: img_fits = {};  hdr_fits = {}
    else: files_fits = {}
    
    for b in bands:
        if band and b not in band: continue
        file = [f for f in os.listdir(filespath+b+'/') if gal in f]
        if file == []: 
            print('Filter %s hasn\'t been found for galaxy %s.' % (b, gal))
            continue
        
        hdu = fits.open(filespath+b+'/'+file[0])[0]
        if mask: hdu = __mask_image(gal, hdu, gal_size=0.5, FWHM=2, FWHM_factor=1, border_factor=1)
        img = hdu.data

        if size:
            img = Image.fromarray(hdu.data)
            hdu.data = np.array(img.resize(size=(size, size)), dtype=np.float64)
            
        if non_neg: img[np.where(img < 0)] = 0;  hdu.data=img
        if norm: _, maximum = __normalization(__mask_image(gal, hdu).data);  img=img/maximum
    
        if unpack: img_fits[b] = hdu.data;  hdr_fits[b] = hdu.header
        else: files_fits[b] = hdu

    if unpack: return img_fits, hdr_fits
    else: return files_fits

'''_____»_____»_____»_____»_____» Centroid recalculation «_____«_____«_____«_____«_____'''
def __moments_centroid(img):
    """
    # Description
    -------------
    Finds the moments center of a given image

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    Xc : TYPE
        DESCRIPTION.
    Yc : TYPE
        DESCRIPTION.

    """
    ax,ay = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    X, Y, img = ax.ravel(), ay.ravel(), img.ravel()
    
    M00 = np.sum(img)
    M10 = np.sum(X*img)
    M01 = np.sum(Y*img)
    
    Xc = M10/M00;  Yc = M01/M00
    
    # Mxx = np.sum((X-Xc)*(X-Xc)*img)/M00
    # Mxy = np.sum((X-Xc)*(Y-Yc)*img)/M00
    # Myy = np.sum((Y-Yc)*(Y-Yc)*img)/M00
    return Xc, Yc


'''_____»_____»_____»_____»_____» Ellipse fitting «_____«_____«_____«_____«_____'''
def __ellipse_fit(img, r=2.5):
    cat = data_properties(img)
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma', 'semiminor_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)

    position = (cat.xcentroid, cat.ycentroid)
    a = cat.semimajor_sigma.value*r
    b = cat.semiminor_sigma.value*r

    theta = cat.orientation.to(u.rad).value
    apertures = EllipticalAperture(position, a, b, theta=theta)
    return apertures    


'''_____»_____»_____»_____»_____» Simple axis «_____«_____«_____«_____«_____'''
def __simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)


'''_____»_____»_____»_____»_____» Image normalization «_____«_____«_____«_____«_____'''
def __normalization(img, gal_size=0.5, give_max=False):
    gal_img = img[int((0.5-gal_size/2)*np.shape(img)[1]):int((0.5+gal_size/2)*np.shape(img)[1]),
                  int((0.5-gal_size/2)*np.shape(img)[0]):int((0.5+gal_size/2)*np.shape(img)[0])]
    
    maximum = np.max(gal_img)
    if give_max: return img/maximum, maximum
    return img/maximum, maximum
    


