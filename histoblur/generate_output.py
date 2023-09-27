#Code by Rahul Nair and Petros Liakopoulos
#Import packages
import numpy as np
import logging

import sklearn.feature_extraction.image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank
import warnings
from multiprocessing import Pool
from WSI_handling import wsi
import time
import math

import matplotlib.cm
import cv2
import torch
import pandas as pd

from torchvision.models import DenseNet

from tqdm import tqdm

from  skimage.color import rgb2gray
import os
import sys
from .utils import *


import openslide
from tifffile import TiffWriter

from typing import NamedTuple

### logger

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# ### Args class

class Args_Detect(NamedTuple):
    """ Command-line arguments """
    mode: str
    input_wsi: str
    outdir: str
    batchsize: int
    model: str
    gpuid: int
    cpus: int
    enablemask: bool
    white_ratio: float
    binmask: bool

def generate_output(images, gpuid, model, outdir, enablemask, ratio_white, binmask, batch_size, cpus):
    """"Function that generates output """

    ########## Muting non conformant TIFF warning

    warnings.filterwarnings(action='ignore', module='tifffile')


    ##### import model and get training magnification

    model, training_mag, device = load_densenet_model(gpuid, model, logger)

    ##### create results dictionary
    results_dict = {}
    
    failed_slides = 0
    ##### Iterate through images and generate output
    for slide in images:

        try:
            osh  = openslide.OpenSlide(slide)
        except openslide.lowlevel.OpenSlideUnsupportedFormatError as er:
            failed_slides+=1
            logger.error(f"ERROR: {slide} not Openslide compatible, skipping. Detail: {str(er)}")
            continue
        except openslide.lowlevel.OpenSlideError as e:
            failed_slides+=1
            logger.error(f"ERROR: {slide} cannot be read properly using OpenSlide, skipping. Detail: {str(e)}")
            continue


        level, native_mag, targeted_mag = get_level_for_targeted_mag_openslide(osh, slide, training_mag, logger)
        
        logger.info(f"processing slide: {slide}")
        start = time.time()

        if openslide.OpenSlide.detect_format(slide) == "dicom": #if dicom, use accession number to name file
            sample = osh.properties["dicom.AccessionNumber"]
        else:
            samplebase = os.path.basename(slide)
            sample = os.path.splitext(samplebase)[0]
        
        try:
            img, mask_level = get_mask_and_level(slide, 8)
        except openslide.lowlevel.OpenSlideError as e:
            failed_slides += 1
            logger.error(f"ERROR: {slide} Mask extraction failed due to issues with OpenSlide compatibility, skipping. Detail: {str(e)}")
            continue      
        
                    
        
        if(enablemask):
            mask = resize_mask(slide, img)
        
        else:
            print("Generating mask")
            mask_final = generate_mask_loose(img)
            mask = mask_final
            
        logger.info((f"Mask level used: {mask_level}"))
        
        print("Estimating tissue size")
        tissue_size_pixels = int(sum(sum(mask)))
        print(f"{tissue_size_pixels} at 8mpp magnification")

        ### Initializing variables
        
        patch_size = 256
        shape=osh.level_dimensions[level]
        shaperound=[((d//patch_size)+1)*patch_size for d in shape]
        
        ds_mask = osh.level_downsamples[mask_level]
        ds_level = osh.level_downsamples[level]
        t_s_mask = int(patch_size*ds_level//ds_mask)
        
        ##Initialize empty blur mask
        npmm=np.zeros((shaperound[1]//patch_size,shaperound[0]//patch_size,3),dtype=np.uint8)
        
        ##Get coordinates inside mask
        coordsx, coordsy = extract_coords_inside_mask(patch_size=patch_size, level=level, osh=osh, mask=mask, ds_mask = ds_mask, t_s_mask=t_s_mask)
        
        ##Extract all patches in parallel using multithreading and save into list
        patches, xs, ys = extract_patches_from_coordinates(slide, coordsx, coordsy, patch_size, ratio_white, level, cpus)
        patches_np = np.array(patches) #final np array of all patches

        ## Instantiate variables for blur percentage calculations
        total_patches = len(patches)

        not_blurry = 0
        mildly_blurry = 0
        highly_blurry = 0

        total_batches = len(patches_np) // batch_size + (len(patches_np) % batch_size != 0) #total batches for tqdm progress bar

        for i, batch_arr in tqdm(enumerate(divide_batch(patches_np, batch_size)), total=total_batches, desc="Processing batches"):

            batch_xs = xs[i*batch_size:(i+1)*batch_size]
            batch_ys = ys[i*batch_size:(i+1)*batch_size]

            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
            with torch.no_grad():
                output_batch = model(arr_out_gpu)
                output_batch = output_batch.detach().cpu().numpy()

                #Any other processing, for instance:
                output_batch = output_batch.argmax(axis=1)
                output_batch_color = custom_cmap(output_batch)

            not_blurry += sum(output_batch == 0)
            mildly_blurry += sum(output_batch == 1)
            highly_blurry += sum(output_batch == 2)

            # Assuming output_batch_color has the processed patches
            for patch, x, y in zip(output_batch_color, batch_xs, batch_ys):
                npmm[y//patch_size//int(ds_level):y//patch_size//int(ds_level)+1, x//patch_size//int(ds_level):x//patch_size//int(ds_level)+1] = patch

        blur_perc = ((mildly_blurry+highly_blurry) * 100) / total_patches
        mildly_blur_perc = (mildly_blurry * 100) / total_patches
        highly_blur_perc = (mildly_blurry * 100) / total_patches


        end = time.time()
        total = asMinutes(end - start)
        logger.info((f"Slide processed in {total}"))


        #add results to dictionary
        results_dict[sample] = [blur_perc, mildly_blur_perc, highly_blur_perc, native_mag, targeted_mag, tissue_size_pixels, total, slide]
        
        if not (enablemask):        
            bin_mask = mask_final*255
            bin_mask = bin_mask.astype(np.uint8)
            #Write binary mask to png if one not provided

            cv2.imwrite(f'{outdir}/tissue_masks/output_tissue_mask_{sample}.png', bin_mask)
        
        if(binmask):
            blur_binmask = npmm[:,:,1]
            blur_binmask[blur_binmask != 0] = 255

            cv2.imwrite(f'{outdir}/blurmask_{sample}.png', blur_binmask)

        #write mask to output
        with TiffWriter(f'{outdir}/output_{sample}.tif', bigtiff=True, imagej=True) as tif:
        
            tif.save(npmm, compress=6, tile=(16,16) ) 
        
    

    try:
        ###### Format output dictionary and save to csv file
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        results_df.columns = ["total_blurry_perc", "mildly_blurry_perc", "highly_blurry_perc", "native_magnification_level",
        "blur_detection_magnification", "npixels_at_8mpp", "processing_time", "file_path"]
        if failed_slides > 0:
            logger.warning(f"WARNING: {failed_slides} slides were skipped due to lack of openslide compatibility")
    except ValueError:
        logger.error(f"ERROR: No slide in glob pattern is openslide compatible")
        sys.exit(1)

    return results_df
