#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""º
Created on Wed Jul 20 08:35:42 2022

@author: adrian
"""

import numpy as np
import astropy.io.fits as fits
from astropy import coordinates as coords
from astropy import units as u
from astropy.wcs import WCS
from astroquery.sdss import SDSS
from astroquery.ipac.ned import Ned
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import warnings
import os
import sys

warnings.filterwarnings('ignore')

plt.close('all')

# Path where the images will be saved
SAVEPATH = '../Data/'

# Dictionary of coordinates for the images retrieval
coordinates = [[ 10.36676542921, 25.49939825550], 
               [     100.924333,      65.206278], 
               [ 42.06719481784, 34.41981249351], 
               [232.65246902758, 42.71715818051], 
               [237.59468104597, 18.93925918637], 
               [123.19353284951, 26.36177424386], 
               [113.75939726951, 32.82212337845], 
               [213.70990419607, 15.14682752020], 
               [130.90861769543, 50.20553946989], 
               [  9.01707594696, 23.95791745197], 
               [143.02592844955,  8.44180182431], 
               [  1.81606776565, 27.70806114254], 
               [      241.80415,        7.97900], 
               [      228.56312,       20.47866], 
               [143.73997217066, 21.70526682911], 
               [      262.58926,       59.63989], 
               [      357.77607,        1.05672], 
               [      338.52830,        5.57033], 
               [      345.51443,       27.05260], 
               [      126.86050,       17.28408], 
               [      235.37684,       15.78767], 
               [        0.77352,       -1.91383], 
               [      115.63674,       49.80967], 
               [       42.36645,       -0.87301], 
               [      240.75469,       20.95593]]

only_gal = []  # Select only these galaxies
exclude = []   # Exclude these galaxies

overwrite = True   # Overwrite previously downloaded images
plot = False       # Plot downloaded images
spectro = False    # Look for spectroscopic match in SDSS query
blacklist = False  # Ignore blacklisted galaxies (blacklis.txt)
npetro = 3.5       # Rpetro50 factor for image cut 


'''_____»_____»_____»_____»_____» add_to_blaclist «_____«_____«_____«_____«_____'''
def add_to_blacklist(filename, galaxy, filt, message, savepath='./'):
    """
    # Description
    -------------
    Adds a galaxy to a given blacklist file. The galaxy name, 
    the filter and the reason of the addition are the colums of this file.
    
    # Parameters
    ------------
    · filename: str           / Name of the blacklist file (with extension)
    · galaxy:   str           / Name of the galaxy to add
    · filt:     str           / Filter in which the galaxy is blacklisted
    · message:  str           / Additional message (blacklist reason)
    · savepath: str, optional / Path where the file is/will be located. If None, selects current folder

    # Returns
    ---------
    None
    """
    # If the file is not found, columns titles are used
    # Else, the colums are read
    if not os.path.isfile(savepath+filename): galaxies = ['Galaxy'];  filters = ['Filter'];  messages = ['Error message']
    else: galaxies, filters, messages = np.loadtxt(savepath+filename, unpack=True, delimiter=', ', dtype=str)
    # New entry is added
    galaxies = np.append(galaxies, galaxy);  filters = np.append(filters, filt)
    messages = np.append(messages, message)
    # File is saved, overwritting if there was a previous one
    np.savetxt(savepath+filename, np.column_stack([galaxies, filters, messages]), fmt=['%s,', '%s,', '%s'])


'''_____»_____»_____»_____»_____» save_gal_images «_____«_____«_____«_____«_____'''
def save_gal_images(pos, savepath, name=None, filters = ['u', 'g', 'r', 'i', 'z'], spectro=True, 
                    dr=17, npetro=3.5, min_rad=5, plot=False, overwrite=False, timeout=120_000):
    """
    # Description
    -------------
    For a given set of coordinates, the images of the objects are retrieved
    and saved.
    
    # Parameters
    ------------
    · pos:       Astropy SkyCoord    / Astropy coordinates object containing RA and DEC
    · savepath:  str                 / Path where files will be saved separated by filters
    · name:      str, optional       / Name of the galaxy. If None, name will be selected as the coordinates
    · filters:   str list, optional  / List of filters for the image download
    · spectro:   bool, optional      / Perform spectroscopic match in the query
    · dr:        int, optional       / SDSS data release
    · npetro:    float, optional     / Factor for Rpetro50 to cut the image
    · min_rad:   float, optional     / Minimum Rpetro50 value for retrieval
    · plot:      bool, optinoal      / Plot retrieved images
    · overwrite: bool, optional      / Overwrite existing images
    · timeout:   float, optional     / Timeout for the query process  
    message: Additional message (blacklist reason)
    
    # Returns
    ---------
    None
    """    
    
    ra = pos.ra.value;  dec = pos.dec.value  # RA and DEC are obtained from SkyCoords object
    if name == '???': # If unknown name, coordinates are used as name
        name = 'RA=%.2f, DEC=%.2f' % (ra, dec)
    
    main = ['run', 'rerun', 'camcol', 'field']          # Selected values to retrieve from query
    petroR50 = ['petroR50_'+filt for filt in filters]   # Petrosian50 radii are added to retrieved values
    
    xid = SDSS.query_region(pos, spectro=spectro, photoobj_fields=main+petroR50, timeout=timeout)  # Query from coordinates
    # crossid = SDSS.query_crossid(pos, obj_names=[name], timeout=timeout)  # Alternative query from names (not used)

    img = {}  # Dictionary that will contain images by filter
    for filt in filters: # For each filter
        if blacklist:  # Blacklist filtering for undesired galaxies (add as blacklist.txt)
            if os.path.isfile('blacklist.txt'):
                txt_gal, txt_filt, _ = np.loadtxt('blacklist.txt', unpack=True, delimiter=',', dtype=str)
                txt_gal = np.array([x.replace(' ', '') for x in txt_gal])
                txt_filt = np.array([x.replace(' ', '') for x in txt_filt])
                verify = list(np.where((txt_gal == name) & (txt_filt == filt))[0])  # Searchs for the galaxy in the specific filter inside the blacklist
                if verify !=  []:  # If the galaxy has been found in this filter, it is ignored
                    print('  ∟ Galaxy %s with filter %s previously added to blacklist.' % (name, filt))
                    continue
        if not(overwrite) and os.path.isfile(SAVEPATH+'/'+filt+'/'+name+'_'+filt+'.fits'):  # If file is already download and overwrite is off
            print('  ∟ Galaxy %s image for filter %s has already been downloaded.' % (name, filt))
            continue
        img[filt] = SDSS.get_images(coordinates=pos, matches=xid, band=filt, data_release=dr, timeout=timeout)  # Filter image is obtained from previous query
        
    filters = list(img.keys())  # Redefinition of filters to avoid skipped ones  
    if filters == []: return    # If no filter left, returns

    if plot:  # Starts plot preparation
        fig, ax = plt.subplots(3, 2)
        fig.suptitle('RA=%.2f, DEC=%.2f' % (ra, dec))
    
        fig2, ax2 = plt.subplots(3, 2)
        fig.suptitle('RA=%.2f, DEC=%.2f' % (ra, dec))
    
    k = 0
    for i, filt in enumerate(filters):
        if len(img[filt]) == 0:  # If no images have been found
            print('  ∟ Galaxy %s image for filter %s hasn\'t been found.' % (name, filt))
            add_to_blacklist('blacklist.txt', name, filt, 'Filter not found')
            continue
        
        for j, img_filt in enumerate(img[filt]):  # For each filter with at least one found image
            if i > 2: i = i - 3; k = 1  # Change of index to split the plots in 3 columns maximum
            data = img_filt[0].data     # Data read from FITS
            head = img_filt[0].header   # Header read from FITS

            # Change from RADEC to pixels in the image to find the center
            w = WCS(head)
            x, y = w.world_to_pixel(pos)
            
            petro_R50 = xid['petroR50_'+filt][j]  # Petrosian50 radius for the filter
            # query = 'SELECT p.petroR50_'+filt+' FROM PhotoObj as p WHERE p.objID = '+str(crossid['objID'][0])
            # petro_rad = SDSS.query_sql(query, dr=dr, timeout=timeout)['petroR50_'+filt]  # Alternative query (not used)
            size = int(npetro*petro_R50/0.396)  # Change to pixels. SDSS scale from http://classic.sdss.org/dr3/instruments/imager/

            if (x < size) or (x > np.shape(data)[1] - size) or \
               (y < size) or (y > np.shape(data)[0] - size):  # If the cut image is too near to the full image border
                   if j + 1 == len(img[filt]):  # If this happens to the last image, no cut has been possible for any of the retrieved full images
                       print('  ∟ Galaxy %s is out of bounds in filter %s image for size=%i (x=%i, y=%i).' % (name, filt, 2*size, x, y))
                       add_to_blacklist('blacklist.txt', name, filt, 'Out of image bounds')
                   continue
               
            if petro_R50 < min_rad:  # If the petrosian radius is too small
                if j + 1 == len(img[filt]): 
                    print('  ∟ Galaxy %s\' Rp50(%s)=%.2f is too small.' % (name, filt, petro_R50))
                    add_to_blacklist('blacklist.txt', name, filt, 'Petrosian radius too small (%.2f)' % (petro_R50))
                continue
                
            gal_data = data[int(y-size):int(y+size), int(x-size):int(x+size)]  # The galaxy image is cut from the galaxy center
    
            if plot:  # The galaxy image is plotted in the figure
                ax[i, k].set_title(filt, fontsize=18)
                ax[i, k].imshow(data, norm=LogNorm())
                ax[i, k].plot(x, y, 'r.')
                ax[i, k].add_patch(Rectangle((x-size, y-size), 2*size, 2*size, fill=None, color='r'))
                ax[i, k].invert_yaxis()

                ax2[i, k].set_title(filt, fontsize=18)
                ax2[i, k].imshow(gal_data, norm=LogNorm())
                ax2[i, k].plot(np.shape(gal_data)[0]/2, np.shape(gal_data)[1]/2, 'r.')
                ax2[i, k].invert_yaxis()
    
            hdu = fits.PrimaryHDU(data=gal_data, header=head)  # The FITS file is created
            try:
                if not os.path.isdir(savepath+'/'+filt+'/'): os.makedirs(savepath+'/'+filt+'/')
                hdu.writeto(savepath+'/'+filt+'/'+name+'_'+filt+'.fits', overwrite=overwrite)  # The FITS file is saved
                print('  ∟ Galaxy %s image for filter %s has been saved. Image size: %i x %i' % (name, filt, 2*size, 2*size))
                break
            
            except Exception as e:
                print('  ∟ Problem with %s:' % (name), e)
                add_to_blacklist('blacklist.txt', name, filt, 'Problem when saving: '+e)
                continue    
    return


if __name__ == '__main__':
    for g, radec in enumerate(coordinates): # For each pair of RADEC coordinates
        ra, dec = radec
        pos = coords.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs', unit=u.deg)  # Astropy SkyCoord object
    
        try:
            ned_region = Ned.query_region(pos, radius=0.001*u.deg)  # NED query to obtain the name
            names = list(ned_region['Object Name'])
            name = [n.replace(' ', '') for n in names if ('NGC' in n or 'IC' in n or 'UGC' in n)][0]  # Galaxies should start with these 
            # ned_diameters = Ned.get_table(name, table='diameters')   #!!! For position angle (not used)
            # ned_ = Ned.get_table(name, table='positions')
        except:
            name = '???'  # If the NED query fails, the name is unknown
        
        print('\n● %i/%i) Working with %s || RA=%.2f  DEC=%.2f' % (g+1, len(coordinates), name, ra, dec))
        # The image download is called
        save_gal_images(pos, npetro=npetro, name=name, savepath=SAVEPATH, spectro=spectro, plot=plot, overwrite=overwrite)
    
