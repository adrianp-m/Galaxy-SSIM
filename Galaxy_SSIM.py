#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:52:21 2022

@author: adrian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean as cmo
import cmasher as cma
import astropy.io.fits as fits
import os
import sys
import warnings
import time


if os.getcwd()+'/FITS_manipulation' not in sys.path: sys.path.append(os.getcwd()+'/FITS_manipulation')
from FITS_utils import __image_retrieve, __normalization, __ellipse_fit
from rotation import __rotate_image, __find_orientation
from SSIM_utils import __compare_images
from masking import __mask_sources, __ellipse_mask

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    plt.close('all')

    '''_____»_____»_____»_____»_____» Parameters «_____«_____«_____«_____«_____'''
    ### Matplotlib parameters
    plt.rcParams["figure.figsize"] = (6,6)

    # Font sizes
    plt.rcParams['figure.titlesize'] = 26
    plt.rcParams['axes.titlesize'] =  18
    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 16

    # Borders
    # plt.rcParams['axes.spines.top'] = False
    # plt.rcParams['axes.spines.bottom'] = False
    # plt.rcParams['axes.spines.left'] = False
    # plt.rcParams['axes.spines.right'] = False
    
    filespath = '../Data/'         # Path where files are located
    file = 'SSIM'                  # Name of the csv file where the results are saved
    savepath = '../SSIM_results/'  # Path where results and images are saved
    
    band = ['g', 'r', 'i', 'z']    # Selected filters for the analysis
    band = 'g'
    sum_filters = ['g', 'r', 'i']  # Selected filters in the color composition
    
    flip = True         # Try flip in alignment
    permute = False     # Try permutation in alignment
    ellipse = True      # Use ellipse masking
    log = True          # Use logarithm scale in the final comparison
    plot = True         # Plot results
    
    
    min_size = 100      # Minimum allowed image size
    aper_rad = 2        # Aperture radius for ellipse masking
    threshold_SSIM = 0.9      # SSIM threshold for twins criterion
    overwrite = False   # Overwrite existing results
    
    # Changes in labels for saving purposes
    if ellipse: savepath += 'ell_';  file += '_ell'
    if log: savepath += 'log_';  file += '_log'
    if not log and not ellipse: savepath += 'raw_'
    if savepath.endswith('_'): savepath = savepath[:-1]
    savepath += '/'
    sup_label = savepath.split('/')[-2]
    
    
    '''_____»_____»_____»_____»_____» Galaxies «_____«_____«_____«_____«_____'''
    
    '''_____» Galaxy names «_____'''
    all_galaxies = ['NGC0001', 'NGC0160', 'NGC0214', 'NGC1093', 'NGC2253', 'NGC2410', \
                'NGC2540', 'NGC2596', 'NGC2639', 'NGC2906', 'NGC2916', 'NGC5522', \
                'NGC5947', 'NGC5980', 'NGC6004', 'NGC6032', 'NGC6063', 'NGC6394', \
                'NGC7311', 'NGC7466', 'UGC00005', 'UGC02311', 'UGC03973', 'UGC09777', \
                'UGC12810']
    all_galaxies.sort()
    
    '''_____» Active «_____'''
    AGN = dict(zip(all_galaxies, [False, False, True, True, False, True, False, False, \
                              True, True, True, False, False, False, False, False, \
                              False, True, True, True, True, False, True, False, False]))
    all_galaxies = [g+'*'*AGN[g] for g in all_galaxies]
        
    '''_____» Hubble classification «_____'''
    Hubble = dict(zip(all_galaxies, ['Sbc', 'Sa', 'SBbc', 'SBbc', 'SBbc', 'SBb', 'SBbc', \
                                  'Sbc', 'Sa', 'Sbc', 'Sbc', 'SBb', 'SBbc', 'Sbc', \
                                  'SBbc', 'SBbc', 'Sbc', 'SBbc', 'Sa', 'Sbc', 'Sbc', \
                                  'SBbc', 'SBbc', 'Sbc', 'SBbc']))
        
    '''_____» Subsample selection «_____'''
    # only_gal = ['NGC0214', 'NGC1093', 'NGC7311']  # Select only these galaxies
    only_gal = []
    exclude = []   # Exclude these galaxies
    
    galaxies = all_galaxies
    if only_gal != []:
        galaxies = [g for g in all_galaxies if g.replace('*', '') in only_gal]

        
    if exclude != []:
        exclude = [g for g in all_galaxies if g.replace('*', '') in exclude]
        for ex in exclude: 
            del(galaxies[galaxies.index(ex)])


'''_____»_____»_____»_____»_____» Analysis «_____«_____«_____«_____«_____'''

def main_analysis(galaxies, band, filespath, file, Hubble, flip=True, permute=False, ellipse=True, log=True, \
                    min_size=100, threshold_SSIM=0.9, aper_rad=2, \
                    overwrite=False, plot=True, sup_label=None, savepath=None, LaTeX=True):
    """
    # Description
    -------------
    For a given set of galaxies, the SSIM is calculated for each combination
    of them following the main pipeline from the already cut images (resize
    all images to the minimum size, mask the external sources, normalization, 
    alignment, rotation of masked non-normalized image, use of ellipse mask,
    logarithmic scale, final SSIM comparison). For further information, see
    the scholarship report.

    # Parameters
    ------------
    · galaxies       : str list        / List of galaxies names
    · band           : str             / Filter to be analysed
    · filespath      : str             / Path where original files are located
    · file           : str             / Name of the .csv file where the results will be saved
    · Hubble         : str, dict       / Dictionary with the Hubble type for each galaxy
    · flip           : bool, optional  / Flip the second galaxy image in each alignment
    · permute        : bool, optional  / Change the rotated galaxy in each alignment
    · ellipse        : bool, optional  / Use the ellipse masking for the final comparison
    · log            : bool, optional  / Use the logarithmic scale for the final comparison
    · min_size       : int, optional   / Minimum allowed galaxy size
    · threshold_SSIM : float, optional / SSIM threshold for twins criterion
    · aper_rad       : float, optional / Aperture radius for ellipse detection
    · overwrite      : bool, optional  / Overwrite existing results
    · plot           : bool, optional  / Plot each SSIM comparison
    · sup_label      : str, optional   / Label for the suptitle of each figure
    · savepath       : str, optional   / Path where the results are saved 
    · LaTeX          : bool, optional  / Create the LaTeX tables and figures

    # Returns
    ---------
    None

    """

    i = len(galaxies)
    total = 0
    while i > 1:  # Calculation of total comparisons
        i += -1
        total += i
        
    if not os.path.isdir(savepath): os.makedirs(savepath)  # If not existing, creates path to save results

    
    if overwrite: # If overwrite, all is deleted
        if os.path.isfile(savepath+file+'.csv'): os.remove(savepath+file+'.csv')
        for png in [f for f in os.listdir(savepath) if f.endswith('.png')]: os.remove(savepath+png)
    
    ini_time = time.time()
    size = None
    SSIM = np.ones((len(galaxies), len(galaxies)))

    not_found = []
    for g, gal in enumerate(galaxies):  # The lowest size from the sample is searched and galaxies with no data are not considered
        fitsfile = __image_retrieve(gal.replace('*', ''), filespath, band=band, unpack=False)
        
        if fitsfile is None:  # Galaxy image not found
            not_found.append(gal)
            continue
        if size is None: size = fitsfile.header['NAXIS1']  
        elif fitsfile.header['NAXIS1'] < size and fitsfile.header['NAXIS1'] > min_size: size = fitsfile.header['NAXIS1']
        elif fitsfile.header['NAXIS1'] < min_size: del(galaxies[g])
    
    print('Minimal size found: %ipx \n' % (size))
    print('List of used galaxies: %s (%i galaxies, %i comparisons)\n' % (galaxies, len(galaxies), total))
    remains = galaxies.copy()
    if os.path.isfile(savepath+file+'.csv'): # If there is a csv file, the analysis is resumed from where it stopped
        df = pd.read_csv(savepath+file+'.csv', index_col=0)
        done_galaxies = list(df.index)
        SSIM[0:len(done_galaxies), :] = df.values  # Data is rescued
        SSIM = np.triu(SSIM) + np.triu(SSIM, 1).T  # Diagonals symmetry is applied 
        remains= [g for g in galaxies if g not in done_galaxies]  # Galaxies not analyzed yet
        if remains == []: return
    
    count = 0
    unused_galaxies = galaxies.copy()  # Galaxies that have already been analysed with all their respective permutations
    for i, galaxy1 in enumerate(galaxies):
        del(unused_galaxies[unused_galaxies.index(galaxy1)])

        if galaxy1 in not_found:
            SSIM[i, :] = 0  # The galaxy has not been found, a 0 is established for all its comparisons
            df = pd.DataFrame(dict(zip(galaxies, SSIM[i,:])), columns=galaxies, index=[galaxy1])
            df.to_csv(savepath+file+'.csv', mode='a+', header=not(os.path.isfile(savepath+file+'.csv')))
            print('● Galaxy %s has not been found.' % galaxy1)
            print('\n'+70*'-'+'\n')
            count += len(unused_galaxies)
            continue
        
        if remains != [] and galaxy1 not in remains: # The csv existed and this galaxy has already been analysed
            print('● Galaxy %s already analysed.' % galaxy1)
            print('\n'+70*'-'+'\n')
            count += len(unused_galaxies)
            continue
        
        if i != len(galaxies) - 1: print('● Galaxy %s %s is being compared with:' % (galaxy1.ljust(10), ('('+Hubble[galaxy1]+')').ljust(7)))
        hdu1  = __image_retrieve(galaxy1.replace('*', ''),  filespath, band=band, size=size, mask=False, unpack=False)  # Galaxy 1 image is retrieved
        hdu1_masked = __mask_sources(galaxy1, hdu1, border_factor=0)                                                    # Masking of the image
        hdu1_norm = fits.PrimaryHDU(__normalization(hdu1_masked.data), hdu1.header)                                  # Normalized image copy

        for galaxy2 in unused_galaxies:
            count += 1
            j = galaxies.index(galaxy2)

            if galaxy2 in not_found:
                SSIM[i, j] = 0;  SSIM[j, i] = 0  # If the galaxy has not been found, a 0 is established for the two permutations with galaxy1
                print('       ∟ %s (Not found, skipping).' % galaxy2)
                continue
            
            
            sys.stdout.write(r'       ∟ %s %s | ' % ((galaxy2.ljust(10)), ('('+Hubble[galaxy2]+')').ljust(7)))
            filename = savepath+'%s_%s_"SSIM"' % (galaxy1, galaxy2)  # Name of the saved plot

            hdu2 = __image_retrieve(galaxy2.replace('*', ''), filespath, band=band, size=size, mask=False, unpack=False)  # Galaxy 2 image is retrieved
            hdu2_masked = __mask_sources(galaxy2, hdu2, border_factor=0)                                                 # Masking of the image
            hdu2_norm = fits.PrimaryHDU(__normalization(hdu2_masked.data), hdu2.header)                               # Normalized image copy
            
            # The relative orientation between the two galaxies is searched. This process takes most of the computational time for each iteration. It can be 
            # reduced by turning off the flip and permutation options.
            _, rot_angle, [flipped, permuted] = __find_orientation(hdu1_norm, hdu2_norm, angle_range=180, precision=1, flip=flip, permute=permute, split=False)
            
            hdu1_rot = hdu1.copy();  hdu1_masked_rot = hdu1_masked.copy()  # Rotated copies of image 1
            hdu2_rot = hdu2.copy();  hdu2_masked_rot = hdu2_masked.copy()  # Rotated copies of image 2
            
            label = ''  # Additional labels will be added if the alignment process has required flip and/or permutation
            if not permuted:
                if flipped: 
                    label = ' (flipped)'
                    hdu2_rot.data = hdu2_rot.data[::-1]  # If the flip was required, the second image (because permuted was used) copies are flipped
                    hdu2_masked_rot.data = hdu2_masked_rot.data[::-1]
                    
                hdu2_rot = __rotate_image(galaxy2, hdu2_rot, rot_angle)  # The second image (because permuted was used) copies are rotated
                hdu2_masked_rot = __rotate_image(galaxy2, hdu2_masked, rot_angle)
                
            else:
                label = '(permuted)'
                if flipped: 
                    label = ' (flipped, permuted)'
                    hdu1_rot.data = hdu1_rot.data[::-1]  # If the flip was required, the first image copies are flipped
                    hdu1_masked_rot.data = hdu1_masked_rot.data[::-1]

                hdu1_rot = fits.PrimaryHDU(__rotate_image(galaxy1, hdu1, rot_angle), hdu1.header)  # The first image copies are rotated
                hdu1_masked_rot = __rotate_image(galaxy1, hdu1_masked, rot_angle)
            
            if ellipse:  # Ellipse masking
                gal1_apert = __ellipse_fit(hdu1_masked_rot.data, r=aper_rad)            # The ellipse parameters are searched for the first galaxy
                img1_final, mask = __ellipse_mask(hdu1_rot.data, apertures=gal1_apert, give_mask=True)  # The ellipse is applied to both images
                img2_final       = __ellipse_mask(hdu2_rot.data, apertures=gal1_apert)
            else:
                img1_final = hdu1_masked_rot.data
                img2_final = hdu2_masked_rot.data
            
            if log:  # Logarithm rescalation. We add the remaining value to reach 1 for the original image, so the minimum value after rescalating is 0
                img1_final = np.log10(img1_final + abs(np.nanmin(img1_final)-1))
                img2_final = np.log10(img2_final + abs(np.nanmin(img2_final)-1))


            # Both images are compared after they have been processed
            SSIM[i, j] = __compare_images(img1_final, img2_final, plot=True, gaussian_weights=True, \
                                          filename=filename, suptitle=sup_label, title1=galaxy1, title2=galaxy2)[1][0]
            plt.close()
            # We copy this value for the opposite comparison, as we ignore the effect of rotating the first or second image
            # (If we have allowed permutation, we are assuming the result that raises the highest SSIM index value)
            SSIM[j, i] = SSIM[i, j]    
                        
            SSIM_label = ('SSIM=%.2f %s ' % (SSIM[i, j], label)).center(29)
            sys.stdout.write('%s| (%i/%i)\n' % (SSIM_label, count, total))  # The result is shown in the terminal
        
        df = pd.DataFrame(dict(zip(galaxies, SSIM[i,:])), columns=galaxies, index=[galaxy1])  # A new row is inserted in the table
        df.to_csv(savepath+file+'.csv', mode='a+', header=not(os.path.isfile(savepath+file+'.csv')))
            
        if i!= len(galaxies) - 1: print('\n'+70*'-'+'\n')

    print('Elapsed time: %im %.2fs' % (divmod(time.time() - ini_time, 60)))
    
    # The LaTeX tables and figures are obtained after the analysis if this option is activated
    if LaTeX and os.path.isfile(savepath+file+'.csv'): LaTeX_results(galaxies=galaxies, Hubble=Hubble, band=band, threshold_SSIM=threshold_SSIM, \
                                    filespath=savepath, file=file, savepath=savepath)


'''_____»_____»_____»_____»_____» LaTeX tables and figures «_____«_____«_____«_____«_____'''
def LaTeX_results(galaxies, Hubble, band, threshold_SSIM, filespath, file, savepath):
    """
    # Description
    -------------
    Creates LaTeX tables and figures from previously obtained SSIM results 
    saved in a csv file for each filter

    # Parameters
    ------------
    · galaxies        : str list / List of galaxies names
    · Hubble          : str dict / Dictionary with the Hubble type for each galaxy
    · band            : str      / Selected filter
    · threshold_SSIM  : float    / SSIM threshold for twins criterion
    · filespath       : str      / Path where the results .csv file is located
    · file            : str      / Name of the results .csv file
    · savepath        : str      / Path where the .tex files are saved 
    
    # Returns
    ---------
    None

    """
   
    '''_____»_____»_____»_____» Full table «_____«_____«_____«_____'''
    df = pd.read_csv(filespath+file+'.csv', index_col=0, header=0)  # Data read
    
    # Table parameters and structure
    LaTeX_table = r'\footnotesize'+'\n' \
    # r'\addtolength{\tabcolsep}{-0.075 cm}'+'\n' \
    r'\begin{landscape}'+'\n' \
    r'\begin{tabularx}{\textwidth}{@{\extracolsep{\fill}}c|%s|c}' % ((len(galaxies))*'c')+'\n'
    r'\large (SDSS%s) \footnotesize' % (band)

    for gal in galaxies:  # Each found galaxy is established in the first row as the columns' headers
        LaTeX_table += r' & \rotatebox{270}{\textbf{%s}} ' % gal
        if gal == galaxies[-1]: LaTeX_table += r' & \rotatebox{270}{\textbf{Mean}} \\ \hline'+'\n'  # Last one ends the row
        
    
    for i, gal in enumerate(galaxies):
        indexes = list(df[gal].values)  # Indexes are obtained from the table
        sorted_indexes = indexes.copy()
        sorted_indexes.sort(reverse=True)               # We sort indexes (descending)
        max_SSIM = str(np.round(sorted_indexes[1], 2))  # Hence, the maximum found value (ignoring exactly 1, same galaxy) is the second one
    
        # Each row is associated with a galaxy. The index values are rounded to 2 decimals and the diagonals as integers
        row = r' \textbf{%s} & ' % (gal)+str([np.round(f, 2) for f in indexes]).replace(',', ' & ')[1:-1]+r' & %.2f \\' % \
            (np.mean(np.array(indexes)[np.where(np.array(indexes) != 1)]))+'\n'  # Mean value of whole row
        row = row.replace('1.0', '1').replace(max_SSIM, r'\textcolor{red}{%s} ' % (max_SSIM))  # The maximum value (1 excluded) is written in red
    
        row_split = row.split()  # The row is splitted to facilitate column separations
        Htype = Hubble[gal]      # Hubble type of the galaxy for this row
        same = np.where(np.array(list(Hubble.values())) == Htype)[0]
        amper_pos = np.where((np.array(row_split) == '&'))[0]  # Ampersands are used to locate columns positions
        
        # These are the indexes above the established threshold (threshold_SSIM). They are written in green
        above_thres = np.where((np.array(indexes) > threshold_SSIM) & (np.array(indexes) != 1 & (np.array(indexes) != sorted_indexes[1])))[0] 
        for a in above_thres:
            if r'\textcolor{red}' not in row_split[amper_pos[a]+1]: row_split[amper_pos[a]+1] = r'\textcolor{ForestGreen}{%s} ' % (row_split[amper_pos[a]+1])
    
        for s in same:  # Galaxies with the same Hubble type have their cells painted in light blue
            if list(Hubble.keys())[s] != gal: row_split[amper_pos[s+1]] = '\cellcolor{SkyBlue}'+row_split[amper_pos[s+1]]
          
        row = ''.join(row_split).replace('&', ' & ')+' \n'  # The row is joined again as a single string variable 
        row = row.replace(' 0 ', ' - ')  # 0 values are replaced with -
        LaTeX_table += row  # The row is added to the whole table text
    
    # After all galaxies' rows have been aded, some final considerations are added
    LaTeX_table += r'\hline'+(len(galaxies) + 1)*' & ' + r'\large %.2f \\' % np.mean(df.values[df.values != 1])+'\n'  # Mean value of all SSIM indexes
    LaTeX_table += r"\caption{\footnotesize Found SSIM index values for all the galaxy sample in filter SDSS%s." % band + \
    r"The first column and row are the galaxies' names. The names with a star indicate an active galaxy." \
    r"The SSIM index values above %.2f are written in green," % threshold_SSIM +\
    r' while the highest value for each row is written in red. The cells between galaxies of the same Hubble type are in blue.' \
    r' The last column shows the mean SSIM index for each row and for all the sample galaxies.}'+'\n' \
    r'\label{Tab: All galaxies SSIM %s}' % (band) +'\n'+ \
    r'\end{tabularx}'+'\n' \
    r'\end{landscape}'+'\n' \
    
    # The table is created, saved and closed
    table = open(savepath+'LaTeX_table.tex','w+')
    table.write(LaTeX_table)
    table.close()


    '''_____»_____»_____»_____» Table by type «_____«_____«_____«_____'''
    df_or = pd.read_csv(filespath+file+'.csv', index_col=0, header=0)  # Data read

    table = open(savepath+'LaTeX_table_type.tex','w+')
    LaTeX_results = {}
    Htypes = np.unique(list(Hubble.values()))
    Htypes.sort()
    for Htype in Htypes:  # Iterations separated by Hubble type
        Hgalaxies = [g for g in galaxies if Hubble[g] == Htype]  # Lisft of galaxies for this Hubble type
        df = df_or[Hgalaxies].loc[Hgalaxies]  # DataFrame for this Hubble type galaxies
    
        LaTeX_table = r'\footnotesize'+'\n' \
        r'\begin{tabularx}{\textwidth}{@{\extracolsep{\fill}}c|%s|c}' % ((len(Hgalaxies))*'c')+'\n'+\
        r'\normalsize \begin{tabular}[c]{@{}c@{}} \textbf{Type %s} \\ \footnotesize{(}SDSS%s{)}\end{tabular} ' % (Htype, band)
        
        for gal in Hgalaxies:
            LaTeX_table += r' & \rotatebox{45}{\textbf{%s}}' % gal
            if gal == Hgalaxies[-1]: LaTeX_table += r' & \textbf{Mean} \\ \hline'+'\n'
            
        for i, gal in enumerate(Hgalaxies):
            indexes = list(df[gal].values)
            sorted_indexes = indexes.copy()
            sorted_indexes.sort(reverse=True)
            max_SSIM = str(np.round(sorted_indexes[1], 2))
    
            row = r' \textbf{%s} & ' % (gal)+str([np.round(f, 2) for f in indexes]).replace(',', ' & ')[1:-1]+r' & %.2f \\' % \
                (np.mean(np.array(indexes)[np.where(np.array(indexes) != 1)]))+'\n' 
            row = row.replace('1.0', '1').replace(max_SSIM, r'\textcolor{red}{%s}' % (max_SSIM))
    
            row_split = row.split()
            amper_pos = np.where((np.array(row_split) == '&'))[0]
            above_thres = np.where((np.array(indexes) > threshold_SSIM) & (np.array(indexes) != 1 & (np.array(indexes) != sorted_indexes[1])))[0]
            for a in above_thres:
                if r'\textcolor{red}' not in row_split[amper_pos[a]+1]: row_split[amper_pos[a]+1] = r'\textcolor{ForestGreen}{%s}' % (row_split[amper_pos[a]+1])
    
            row = ''.join(row_split).replace('&', ' & ')+' \n'
            row = row.replace(' 0 ', ' - ')
            LaTeX_table += row
    
        LaTeX_table += r'\hline'+(len(Hgalaxies) + 1)*' & ' + r'\normalsize %.2f \\' % np.mean(df.values[df.values != 1])+'\n'
        LaTeX_table += r"\caption{\footnotesize Found SSIM index values for %s type galaxies of the total sample in filter SDSS%s. " % (Htype, band) + \
        r"The first column and row are the galaxies' names.  The names with a star indicate an active galaxy. " +\
        r"The SSIM index values above %.2f are written in green," % threshold_SSIM +\
        r' while the highest value for each row is written in red. The last column shows the mean SSIM index for each row and for all the %s galaxies.}' % Htype +'\n' \
        r'\label{Tab: %s galaxies SSIM %s}' % (Htype, band)+'\n' \
        r'\end{tabularx}'+'\n' \
        r'\vspace{-0.5 cm}'
         
        # The table is created, saved and closed
        LaTeX_results[Htype] = LaTeX_table
        table.write(LaTeX_table+'\n')
    
    table.close()
    
    '''_____»_____»_____»_____» Table by type «_____«_____«_____«_____'''
    df = pd.read_csv(filespath+file+'.csv', index_col=0, header=0)
    
    for gal1 in galaxies:
        i = 0
        if not '*' in gal1: continue  # Only active galaxies are selected
        
        # The figure is created with certain parameters
        LaTeX_fig = r'\begin{figure}[H]'+'\n' \
                    r'\subfloat[%s - %s, Htype=%s]{\includegraphics[width=0.2\textwidth]{Figures/images/%s.jpg}}' % (gal1, band, Hubble[gal1], gal1.replace('*', ''))+'\n'
            
        df_gal = df.loc[gal1]  # The galaxy-related indexes are searched
        del(df_gal[gal1])      # The index with itself is deleted
        max_SSIM = np.where(df_gal == np.max(df_gal))[0][0]  # The maximum value is searched
        best_twin = df_gal.keys()[max_SSIM]                  # so the best twin is found
        
        # The best twin has its labels in red and is next to the active galaxy
        LaTeX_fig += r'\subfloat[\textcolor{red}{%s - %s,\newline SSIM=%.2f,\newline Htype=%s}]{\includegraphics[width=0.2\textwidth]{Figures/images/%s.jpg}} \\' % \
                        (best_twin, band, df_gal[best_twin], Hubble[best_twin], best_twin.replace('*', '')) +'\n'
        del(df_gal[best_twin])  # The best twin is now deleted from the list
        
        df_gal = df_gal.sort_values(ascending=False)  # Values are sorted descending
        for gal2 in df_gal.keys():
            if df_gal[gal2] > threshold_SSIM: # If the galaxy is above the threshold, it is considered a twin and its figure is added
                LaTeX_fig += r'\subfloat[%s - %s,\newline SSIM=%.2f,\newline Htype=%s]{\includegraphics[width=0.2\textwidth]{Figures/images/%s.jpg}}' % \
                                (gal2, band, df_gal[gal2], Hubble[gal2], gal2.replace('*', ''))
                i += 1
                if i == 5: 
                    LaTeX_fig += r'\\'
                    i = 0
                LaTeX_fig += '\n'
        
        # Once all the twins' images have been added, we add the figure caption and end it
        LaTeX_fig += r'\caption{SSIM results for %s}' % gal1 +'\n' \
                     r'\end{figure}'+'\n'+r'\newpage'+'\n\n'
        
        # The figure is created, saved and closed
        fig = open(savepath+'LaTeX_figs.tex','w+')
        fig.write(LaTeX_fig)  # 
    
    fig.close()


'''_____»_____»_____»_____»_____» Main calls «_____«_____«_____«_____«_____'''

if __name__ == '__main__':
    if np.shape(band) == (): band = [band]
    
    SSIM_filt = {} # Dictionary to save all the SSIM index matrix by filter
    for i, b in enumerate(band):  # Results for each filter
        # try:
        #     print(70*'='+'\n')
        #     print('Working with filter SDSS%s\n'.center(70) % b)
        #     print(70*'='+'\n')
            
        main_analysis(galaxies, b, filespath, file, Hubble, flip=flip, permute=permute, ellipse=ellipse, log=log,
                        min_size=min_size, threshold_SSIM=threshold_SSIM, aper_rad=aper_rad,
                        overwrite=overwrite, plot=plot, sup_label=sup_label, savepath=savepath+b+'/', LaTeX=True)
        # except Exception as e:
        #     print('Filter %s failed: %s' % (b, e))
        #     continue
        
        if sum_filters and b in sum_filters:  # Average of the selected filters
            df = pd.read_csv(savepath+b+'/'+file+'.csv', index_col=0)
            SSIM_filt[b] = df.values
            if len(SSIM_filt) == 1: SSIM_total = SSIM_filt[b]  # If the dictionary has only one entry, the total matrix is initialized
            else:  # If it already exists, the filter matrix is added
                SSIM_total += SSIM_filt[b]
        
    
    if sum_filters is not None and SSIM_filt != {}:  # If we have selected some filters
        SSIM_total = SSIM_total/len(SSIM_filt.keys())  # The average is calculated
    
        galaxies = list(df.index)  # The used galaxies
        
        # The table is created and saved in the main folder
        df = pd.DataFrame(SSIM_total, columns=galaxies, index=galaxies)
        df.to_csv(savepath+file+'_all.csv', mode='w+', header=True)
        LaTeX_results(galaxies, Hubble, ''.join(sum_filters), threshold_SSIM, savepath, file+'_all', savepath)    
    