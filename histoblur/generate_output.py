#Code by Petros Liakopoulos
#Import packages
import numpy as np
import logging
import csv
import sklearn.feature_extraction.image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk, remove_small_objects
from scipy.ndimage import minimum_filter
from multiprocessing import Pool, cpu_count
import time
import math

import matplotlib.cm
import cv2
import torch

from torchvision.models import DenseNet

from tqdm import tqdm

from  skimage.color import rgb2gray
import os
import sys
from .utils import *


import openslide

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
    min_size_object: int
    binmask: bool

def generate_output(images, gpuid, model, outdir, enablemask, ratio_white, min_size_object, binmask, batch_size, cpus):
    """"Function that generates output """

    
    ##### import model and get training magnification

    model, training_mag, device = load_densenet_model(gpuid, model, logger)

    ##### create results dictionary
    results_dict = {}
    
    failed_slides = 0

    ##### Lookup table to convert DL prediction to pixel vals:

    lookup_table = {
            -1: np.array([0, 0, 0]),    # No prediction
            0: np.array([0, 255, 0]),    # Class 0 
            1: np.array([255, 0, 0]),    # Class 1 blue BGR
            2: np.array([0, 0, 255])     # Class 2 red BGR
        }

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
            img, mask_level = get_mask_and_level(osh, 8)
        except openslide.lowlevel.OpenSlideError as e:
            failed_slides += 1
            logger.error(f"ERROR: {slide} Mask extraction failed due to issues with OpenSlide compatibility, skipping. Detail: {str(e)}")
            continue      
        
                    
        
        if(enablemask):
            mask = resize_mask(slide, img)
        
        else:
            print("Generating mask")
            mask_final = generate_mask(img, min_size_object)
            mask = mask_final
            
        logger.info((f"Mask level used: {mask_level}"))
        
        print("Estimating tissue size")
        tissue_size_pixels = mask.sum()
        print(f"{tissue_size_pixels} at 8mpp magnification")

        ### Initializing variables
        
        patch_size = 256
        shape = osh.level_dimensions[level]
        shaperound = [math.ceil(d / patch_size) * patch_size for d in shape]


        ds_mask = osh.level_downsamples[mask_level]
        ds_level = osh.level_downsamples[level]
        t_s_mask = int(patch_size*ds_level//ds_mask)

        # Initialize empty blur mask
        npmm = -1 * np.ones((shaperound[1]//patch_size, shaperound[0]//patch_size), dtype=np.int8)

        # Get coordinates inside mask
        coords = extract_coords_inside_mask(patch_size=patch_size, level=level, osh=osh, mask=mask, ds_mask=ds_mask, t_s_mask=t_s_mask)

        # Extract all patches in parallel using multithreading and save into list
        patches, coords_list = extract_patches_from_coordinates(slide, coords, patch_size, ratio_white, level, cpus)
        patches_np = np.array(patches) # final np array of all patches

        # Instantiate variables for blur percentage calculations
        total_patches = len(patches)
        total_batches = len(patches_np) // batch_size + (len(patches_np) % batch_size != 0) # total batches for tqdm progress bar


        for batch_arr, batch_coords in tqdm(divide_batch(patches_np, coords_list, batch_size=batch_size), total=total_batches, desc="Processing batches"):
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
            
            with torch.no_grad():
                output_batch = model(arr_out_gpu)
                output_batch = output_batch.argmax(dim=1)
                output_batch = output_batch.detach().cpu().numpy()

            # Convert batch_coords to numpy array and perform vectorized operations
            batch_coords = np.asarray(batch_coords)
            batch_coords = (batch_coords // patch_size // int(ds_level)).astype(int)
            
            # Place the DL outputs onto the matrix
            npmm[batch_coords[:, 1], batch_coords[:, 0]] = output_batch


        npmm_final = np.vectorize(lookup_table.get, signature='()->(n)', otypes=[np.uint8])(npmm)

        # Count based on non-zero pixels in each channel in BGR
        mildly_blurry, _, highly_blurry = np.sum(npmm_final != 0, axis=(0, 1))

        # Compute percentages
        blur_perc = round(((mildly_blurry + highly_blurry) * 100) / total_patches, 3)
        mildly_blur_perc = round((mildly_blurry * 100) / total_patches, 3)
        highly_blur_perc = round((highly_blurry * 100) / total_patches, 3)

        end = time.time()
        total = asMinutes(end - start)
        logger.info(f"Slide processed in {total}")



        #add results to dictionary
        results_dict[sample] = [samplebase, blur_perc, mildly_blur_perc, highly_blur_perc, native_mag, targeted_mag, tissue_size_pixels, total, slide]
        
        if not (enablemask):        
            bin_mask = mask_final*255
            bin_mask = bin_mask.astype(np.uint8)
            #Write binary mask to png if one not provided

            cv2.imwrite(f'{outdir}/tissue_masks/output_tissue_mask_{sample}.png', bin_mask)
        
        if(binmask):
            blur_binmask = npmm_final[:,:,1]
            blur_binmask[blur_binmask != 0] = 255

            cv2.imwrite(f'{outdir}/blurmask_{sample}.png', blur_binmask)

        #write blur mask to output
        cv2.imwrite(f'{outdir}/output_{sample}.png', npmm_final)
        
    

    try:
        ###### Format output dictionary and save to csv file
        column_names = ["basename", "total_blurry_perc", "mildly_blurry_perc", "highly_blurry_perc", "native_magnification_level",
        "blur_detection_magnification", "npixels_at_8mpp", "processing_time", "file_path"]
        with open(f"{outdir}/results_overview.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writeheader()
            for key, values in results_dict.items():
                # Use the key as the identifier for the slide (or whatever 'key' represents)
                row_data = dict(zip(column_names, values))
                writer.writerow(row_data)

        if failed_slides > 0:
            logger.warning(f"WARNING: {failed_slides} slides were skipped due to lack of openslide compatibility")

    except ValueError as err:
            logger.error(f"ERROR: {slide} Mask extraction failed due to issues with OpenSlide compatibility, skipping. Detail: {str(err)}")
            sys.exit(1)
            

    return csvfile
