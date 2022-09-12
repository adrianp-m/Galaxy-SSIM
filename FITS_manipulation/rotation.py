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
import cmasher as cma
import astropy.io.fits as fits
from skimage.transform import rotate
import os
import sys

if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))
from FITS_utils import __simpleaxis, __image_retrieve
from SSIM_utils import __compare_images


##### Rotation of the image
def __rotate_image(gal, hdu, angle, log_rot=False, simil=False, split=True, plot=False, log=True, savepath=None, cmap=None, SSIM_cmap=None):
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r
    
    ########## Data preparation
    try:
        hdu.header['FILTER']
    except:
        hdu = hdu[0]
        
    band = hdu.header['FILTER']
    img = np.array(hdu.data, dtype=float);  head = hdu.header
    rot_img = rotate(img, -angle)
    if log_rot:
        val = abs(np.min(img)-1)
        img = np.log10(img + val)
        log = False
    
    rot_label = 'Rotated (%.2fº)' % (angle)
    if simil: _, rot_ssim = __compare_images(img, rot_img, split=split, plot=plot, log=log, suptitle=gal+' - '+band, title1='Original', title2=rot_label, cmap=cmap, SSIM_cmap=SSIM_cmap)
    if plot and not simil:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
        fig.suptitle(gal+' - '+band)
        if log:
            im0 = ax[0].imshow(img, norm=LogNorm(), cmap=cmap) 
            ax[1].imshow(rot_img, norm=LogNorm(vmin=im0.norm.vmin, vmax=im0.norm.vmax), cmap=cmap)
        else:
            im0 = ax[0].imshow(img, cmap=cmap) 
            ax[1].imshow(rot_img, vmin=im0.norm.vmin, vmax=im0.norm.vmax, cmap=cmap)
        
        ax[0].set_title('Original', fontsize=28)
        ax[1].set_title(rot_label, fontsize=28)
        
        ax_cb = fig.add_axes(rect=[.065,-.025,0.9,.02], frameon=False)
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=45)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout()

    rot_fits = fits.PrimaryHDU(data=rot_img, header=head)
    
    if plot and (savepath is not None): 
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+gal+'_'+band+'_rot_'+str(np.round(angle, 2))+'.png', bbox_inches='tight')
            
    if simil: return rot_fits, rot_ssim
    return rot_fits



def __find_orientation(hdu1, hdu2, log_rot=False, angle_range=180, precision=5, flip=True, permute=True, split=True, \
                       plot=False, log=True, suptitle='', title1='Galaxy 1', title2='Galaxy 2', \
                       all_range=True, cmap=None, SSIM_cmap=None, savepath=None):    
    if cmap is None: cmap = cmo.cm.balance_r
    if SSIM_cmap is None: SSIM_cmap = cmo.cm.balance_r
    angle_range = int(angle_range)

    try:
        hdu1.header['FILTER']
    except:
        hdu1 = hdu1[0]
    try:
        hdu2.header['FILTER']
    except:
        hdu2 = hdu2[0]
    
    img1  = hdu1.data.copy();  img2 = hdu2.data.copy()
    head2 = hdu2.header.copy()
    if log_rot: 
        img1 = np.log10(img1 + abs(np.min(img1)-1))
        log = False
    
    all_angle = np.arange(0, angle_range + precision, precision)
    all_img2 = [];  all_mse = [];  all_ssim = []
    all_ssim_map = [];  all_L = [];  all_C = [];  all_S = []
    i = 0;  angle = 0;  half = len(all_angle)
    while angle <= angle_range + precision:
        rot_img2 = rotate(np.array(img2, dtype=float), -angle)
        if log_rot: rot_img2 = np.log10(rot_img2 + abs(np.min(img2)-1))
            
        rot_mse, rot_ssim = __compare_images(img1, rot_img2, split=split)
        all_img2.append(rot_img2)
        all_mse.append(rot_mse);   all_ssim.append(rot_ssim[0])
        all_ssim_map.append(rot_ssim[1])
        if split:
            all_L.append(rot_ssim[2])
            all_C.append(rot_ssim[4])
            all_S.append(rot_ssim[6])
                
        if angle == angle_range and flip:
            angle = 0
            img2 = img2[::-1]
            flip = False
            half = i + 1

        angle = angle + precision;  i = i + 1

    ori = np.where(all_ssim == np.max(all_ssim))[0][0]
    ori_angle = np.concatenate([all_angle, all_angle])[ori]
    ori_img2 = all_img2[ori]
    ori_mse = all_mse[ori];  ori_ssim = all_ssim[ori];  ori_ssim_map = all_ssim_map[ori]

    if permute:
        ori_fits_p, ori_angle_p, [flip, permute] = __find_orientation(hdu2, hdu1, log_rot=log_rot, angle_range=angle_range, precision=precision, flip=flip, permute=False, split=split, \
                                                                      plot=plot, log=log, suptitle='(permuted)'+suptitle, title1=title2, title2=title1, \
                                                                      all_range=all_range, cmap=cmap, SSIM_cmap=SSIM_cmap, savepath=None)
        
        _, ssim_1 = __compare_images(img1, ori_img2)
        _, ssim_2 = __compare_images(img2, ori_fits_p.data)
        if ssim_2[0] > ssim_1[0]: 
            if savepath is not None: 
                if not os.path.isdir(savepath): os.makedirs(savepath)
                plt.savefig(savepath+title2+'_'+title1+'_rot.png', bbox_inches='tight')    

            return ori_fits_p, ori_angle_p, [flip, True]
        plt.close()

    if ori > half: flip = True

    if plot:
        if suptitle != '': suptitle = suptitle + ' / MSE = %.2f, SSIM = %.2f' % (ori_mse, ori_ssim)
        else: suptitle = 'MSE = %.2f, SSIM = %.2f' % (ori_mse, ori_ssim)
        
        fig, ax = plt.subplots(2, 3, figsize=(18, 16), sharey=True, gridspec_kw={'height_ratios': [0.4, 0.6]})
        plt.suptitle(suptitle, y=0.93)
        
        if log: 
            im0 = ax[0, 0].imshow(img1, norm=LogNorm(), cmap=cmap)
            ax[0, 1].imshow(ori_img2, norm=LogNorm(vmin=im0.norm.vmin, vmax=im0.norm.vmax), cmap=cmap)
            
        else: 
            im0 = ax[0, 0].imshow(img1, cmap=cmap)
            ax[0, 1].imshow(ori_img2, vmin=im0.norm.vmin, vmax=im0.norm.vmax, cmap=cmap)

        ax[0, 0].set_title(title1)
        
        ax[0, 1].set_title(title2)
        ax_cb = fig.add_axes(rect=[.124,.565, 0.505,.01], frameon=False)
        cbar01 = fig.colorbar(im0, orientation='horizontal', cax=ax_cb, format='%.1e')
        cbar01.ax.tick_params(rotation=0)
        
        im3 = ax[0, 2].imshow(ori_ssim_map, vmin=0, vmax=1, cmap=SSIM_cmap)
        ax[0, 2].set_title('SSIM map')
        
        ax3_cb = fig.add_axes(rect=[.671,.565,0.229,.01], frameon=False)
        cbar = fig.colorbar(im3, orientation='horizontal', cax=ax3_cb)            
        cbar.ax.tick_params(rotation=45)
        
        ax4 = plt.subplot(212)
        if angle_range/precision > 35: ls='-';  marker=''
        else: ls='';  marker='o'


        if flip: ax4.plot(all_angle, all_ssim[half:], c='k', ls=ls, marker=marker, label='SSIM')
        else: ax4.plot(all_angle, all_ssim[:half], c='k', ls=ls, marker=marker, label='SSIM') #!!!
        
        if split:
            if flip:
                ax4.plot(all_angle, all_L[half:], c='r', ls=ls, marker=marker, label='Luminosity')
                ax4.plot(all_angle, all_C[half:], c='g', ls=ls, marker=marker, label='Contrast')
                ax4.plot(all_angle, all_S[half:], c='b', ls=ls, marker=marker, label='Structure')
            else:
                ax4.plot(all_angle, all_L[:half], c='r', ls=ls, marker=marker, label='Luminosity')
                ax4.plot(all_angle, all_C[:half], c='g', ls=ls, marker=marker, label='Contrast')
                ax4.plot(all_angle, all_S[:half], c='b', ls=ls, marker=marker, label='Structure')
            
        if split: ax4.legend(loc='best', prop={'size':22})

        ax4.axvline(ori_angle, c='k', ls='--')
        ax4.set_xlim(0, angle_range)
        if flip: ax4.set_title('Relative position angle rPA=%.2fº + X flip' % (ori_angle))
        else: ax4.set_title('Relative position angle rPA=%.2fº' % (ori_angle))
        ax4.set_xlabel('Rotation angle [deg]', fontsize=22)
        ax4.set_ylabel('SSIM index', fontsize=22)
        ax4.grid()
        if all_range: ax4.set_ylim(0, 1)
    
        plt.subplots_adjust(hspace=0.05)
    if savepath is not None: 
        if not os.path.isdir(savepath): os.makedirs(savepath)
        plt.savefig(savepath+title1+'_'+title2+'_rot.png', bbox_inches='tight')    
    
    ori_fits = fits.PrimaryHDU(data=ori_img2, header=head2)

    return ori_fits, ori_angle, [flip, False]
