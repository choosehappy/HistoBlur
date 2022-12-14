#Code by Rahul Nair and Petros Liakopoulos
#Import packages
import numpy as np
from WSI_handling import wsi
import logging

import sklearn.feature_extraction.image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank
import warnings

import time
import math

import matplotlib.cm
import cv2
import torch
import pandas as pd

from torchvision.models import DenseNet

from tqdm.autonotebook import tqdm

from  skimage.color import rgb2gray
import os
import sys
from .dataset_creation_train import getMag, asMinutes

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
    model: str
    gpuid: int
    enablemask: bool
    white_ratio: float
    binmask: bool

def generate_mask_loose(image):
    """generates a mask with thresholding, does not apply erosion"""

    disk_size = 5
    threshold = 200
    img = rgb2gray(image)
    img = (img * 255).astype(np.uint8)
    selem = disk(disk_size)
    imgfilt = rank.minimum(img, selem)
    up_tresh = imgfilt < threshold
    down_thresh =  imgfilt > 0
    mask= np.float32(np.logical_and(up_tresh, down_thresh))
    
    return mask

def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 




def generate_output(images, gpuid, model, outdir, enablemask, perc_white, binmask):
    """"Function that generates output """

    ########## Muting non conformant TIFF warning

    warnings.filterwarnings(action='ignore', module='tifffile')

    ########## Loading Densenet model

    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    batch_size = 16  #we currently use patch_size * 4 for tile size, so each iteration only generates 16 patches at a time (room for optimization if necessary)

    if str(device) == 'cpu':
        logger.warning("Generating output with CPU, this might take longer. Switching to GPU is recommended")

    logger.info(f"model used: {model}")
    checkpoint = torch.load(model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

    training_mag = checkpoint["magnification"]

    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["nclasses"]).to(device)




    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    print("Loading model")
    print(f"total model params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")


    ##### create results dictionary
    results_dict = {}
    
    failed_slides = 0
    ##### Iterate through images and generate output
    for slide in images:

        try:
            osh  = openslide.OpenSlide(slide)
        except openslide.lowlevel.OpenSlideUnsupportedFormatError:
            failed_slides+=1
            logger.error(f"ERROR: {slide} not Openslide compatible, skipping")
            continue

    
 
        ##### getting openslide level for requested magnification
        native_mag = getMag(osh, slide)
        targeted_mag = training_mag
        down_factor = native_mag / targeted_mag
        relative_down_factors_idx=[np.isclose(x/down_factor,1,atol=.01) for x in osh.level_downsamples]
        level=np.where(relative_down_factors_idx)[0]

        if level.size:
            level=level[0]

        else:
            level = osh.get_best_level_for_downsample(down_factor)


        logger.info(f"processing slide: {slide}")
        start = time.time()
        samplebase = os.path.basename(slide)
        sample = os.path.splitext(samplebase)[0]
        fname=slide
        osh_mask  = wsi(fname)
        mask_level_tuple = osh_mask.get_layer_for_mpp(8)
        mask_level = mask_level_tuple[0]
        img = osh_mask.read_region((0, 0), mask_level, osh_mask["img_dims"][mask_level])
                    
        
        if(enablemask):
            mask_img=cv2.imread(os.path.splitext(slide)[0]+'.png', cv2.IMREAD_GRAYSCALE) #--- assume mask has png ending in same directory
            width = int(img.shape[1] )
            height = int(img.shape[0])
            dim = (width, height)
            mask_resize = cv2.resize(mask_img,dim)
            mask = np.float32(mask_resize)
            mask /= 255
    
            
        
        
        else:
            #add mask creation which skips parts of image
                
                print("Generating mask and estimating tissue size")
                img = np.asarray(img)[:, :, 0:3]
                mask_final = generate_mask_loose(img)
                mask = mask_final
    
           
        #estimating tissue 
        tissue_size_pixels = int(sum(sum(mask)))
        print(f"{tissue_size_pixels} at 8 ??m per pixel")

        #adjusting patch size and magnification based on tissue quantity
        if tissue_size_pixels == 0:
            logger.error(f"No tissue detected on slide {slide}")
            continue
        if 0 < tissue_size_pixels < 100000:
            if level == 0:
                patch_size = 64
                logger.info(f"Low tissue quantity detected, unable to use higher magnification cause already at max, patch size selected {patch_size}, openslide level {level}")
            else:
                level = level - 1
                targeted_mag = native_mag/osh.level_downsamples[level]
                patch_size = 128
                logger.info(f"Low tissue quantity detected, using higher magnification, patch size selected {patch_size}, openslide level {level}")
        elif 99999 < tissue_size_pixels < 200000:
            patch_size = 64
            logger.info(f"patch size selected {patch_size}, openslide level {level}")
        elif 199999 < tissue_size_pixels < 500000:
            patch_size = 128
            logger.info(f"patch size selected {patch_size}, openslide level {level}")
        elif tissue_size_pixels > 499999:
            patch_size = 256
            logger.info(f"patch size selected {patch_size}, openslide level {level}")


        cmap= matplotlib.cm.tab10

        osh.level_dimensions
        ds=int(osh.level_downsamples[level])

        
        #change the stride size to speed up the process, must equal factor between levels 
        stride_size = patch_size #Use non overlapping patches
        tile_size=patch_size*4 #the bigger the tile size the smaller the resolution of the mask. x4 the patch size is a good tradeoff
        nclasses=3

        #Variables for percentage of white pixels in each patch
        seventy_perc_thresh = int(patch_size*patch_size * 0.7)
        fifty_perc_thresh = int(patch_size*patch_size * 0.5)
        thirty_perc_thresh = int(patch_size*patch_size * 0.3)
        ten_perc_thresh = int(patch_size*patch_size * 0.1)


        #counters for percentage of white area

        count_seventy = 0
        count_fifty = 0
        count_thirty = 0
        count_ten = 0


        shape=osh.level_dimensions[level]
        shaperound=[((d//tile_size)+1)*tile_size for d in shape]
        ds_mask = osh.level_downsamples[mask_level]
        ds_level = osh.level_downsamples[level]
        t_s_mask = int(tile_size*ds_level//ds_mask)


        npmm=np.zeros((shaperound[1]//stride_size,shaperound[0]//stride_size,3),dtype=np.uint8)
        for y in tqdm(range(0,osh.level_dimensions[0][1],round(tile_size * osh.level_downsamples[level])), desc="outer"):
            for x in tqdm(range(0,osh.level_dimensions[0][0],round(tile_size * osh.level_downsamples[level])), desc=f"innter {y}", leave=False):

                #if skip
                
                
                
                maskx=int(x//ds_mask)
                masky=int(y//ds_mask)
                
                
                if(maskx >= mask.shape[1] or masky >= mask.shape[0]) or mask[masky:masky+t_s_mask,maskx:maskx+t_s_mask].mean() < 0.1:
                    continue
                
                
                output = np.zeros((0,nclasses,1,1))
                io = np.asarray(osh.read_region((x, y), level, (tile_size,tile_size)))[:,:,0:3] #trim alpha
                
                arr_out=sklearn.feature_extraction.image._extract_patches(io,(patch_size,patch_size,3),stride_size)
                arr_out_shape = arr_out.shape
                arr_out = arr_out.reshape(-1,patch_size,patch_size,3)

                #checking white perc
                idx_to_remove = []
                g_arr_out = list(map(rgb2gray, arr_out))
                for i, patch in enumerate(g_arr_out):
                    gray_mask = patch > 0.94 #threshold for white
                    white_pixel_sum = np.sum(gray_mask) #total pixels above that threshold
                    if white_pixel_sum > int(patch_size*patch_size * perc_white):#get index of patches above certain white percentage
                        idx_to_remove.append(i)
                        continue
                    if white_pixel_sum > seventy_perc_thresh: #At least 70% of patch made up of white pixels
                        count_seventy += 1
                        count_fifty += 1
                        count_thirty += 1
                        count_ten += 1
                    elif white_pixel_sum > fifty_perc_thresh: #At least 50%
                        count_fifty += 1
                        count_thirty += 1
                        count_ten += 1
                    elif white_pixel_sum > thirty_perc_thresh: #At least 30%
                        count_thirty += 1
                        count_ten += 1
                    elif white_pixel_sum > ten_perc_thresh: #At least 10%
                        count_ten += 1
                    else:
                        continue
                
                
                for batch_arr in divide_batch(arr_out,batch_size):
                
                    arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
                    with torch.no_grad():
                        # ---- get results
                        output_batch = model(arr_out_gpu)

                        # --- pull from GPU and append to rest of output 
                        output_batch = output_batch.detach().cpu().numpy()

                        output_batch_color=cmap(output_batch.argmax(axis=1), alpha=None)[:,0:3]
                        output_batch_color[idx_to_remove] = [0,0,0] #take indexes of patches beyond the white threshold and turn them black

                    output = np.append(output,output_batch_color[:,:,None,None],axis=0)
                    
                
                output = output.transpose((0, 2, 3, 1))
                
                #turn from a single list into a matrix of tiles
                output = output.reshape(arr_out_shape[0],arr_out_shape[1],patch_size//patch_size,patch_size//patch_size,output.shape[3])
                
                
                #turn all the tiles into an image
                output=np.concatenate(np.concatenate(output,1),1)
                
                
                
                npmm[y//stride_size//ds:y//stride_size//ds+tile_size//stride_size,x//stride_size//ds:x//stride_size//ds+tile_size//stride_size,:]=output*255 #need to save uint8

        #change the color to red green and blue. red signifies high blur, green signifies no blur, blue signifies medium blur
        data = npmm
        data1 = data
        data2 = data
        data3 = data
        r1, g1, b1 = 255, 127 ,14  # Original value
        r2, g2, b2 = 0, 0, 255 # Value that we want to replace it with


        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data1[:,:,:3][mask] = [r2, g2, b2]

        r1, g1, b1 = 44, 160 ,44  # Original value
        r2, g2, b2 = 255, 0, 0 # Value that we want to replace it with


        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data2[:,:,:3][mask] = [r2, g2, b2]

        r1, g1, b1 = 31, 119 ,180  # Original value
        r2, g2, b2 = 0, 255, 0 # Value that we want to replace it with


        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data3[:,:,:3][mask] = [r2, g2, b2]
        final_output = data1+data2+data3
        
        #total patches
        total_classified_patches = sum(sum(final_output[:,:,2] == 253)) + sum(sum(final_output[:,:,1] == 253)) + sum(sum(final_output[:,:,0] == 253)) #count number of patches with 253 in one channel

        #Perc white pixels calculations
        perc_seventy = round(count_seventy/total_classified_patches *100, 3)
        perc_fifty = round(count_fifty/total_classified_patches *100, 3)
        perc_thirty = round(count_thirty/total_classified_patches *100, 3)
        perc_ten = round(count_ten/total_classified_patches *100, 3)

        #Calculate perc of bluriness  
        total_patches_blurry = sum(sum(final_output[:,:,0] == 253)) + sum(sum(final_output[:,:,2] == 253))
        total_very_blurry = sum(sum(final_output[:,:,0] == 253))
        total_mildly_blurry = sum(sum(final_output[:,:,2] == 253))
        perc_tot = round(total_patches_blurry/total_classified_patches*100, 3)
        perc_very_blurry = round(total_very_blurry/total_classified_patches*100, 3)
        perc_mildly_blurry = round(total_mildly_blurry/total_classified_patches*100, 3)
                                                                            
        end = time.time()
        total = asMinutes(end - start)
        logger.info((f"Slide processed in {total}"))


        #add results to dictionary
        results_dict[samplebase] = [perc_tot, perc_mildly_blurry, perc_very_blurry, perc_seventy, perc_fifty, perc_thirty, perc_ten, patch_size, native_mag, targeted_mag, tissue_size_pixels, total, slide]
        
        if not (enablemask):        
            bin_mask = mask_final*255
            bin_mask = bin_mask.astype(np.uint8)
            #Write binary mask to png if one not provided

            cv2.imwrite(f'{outdir}/tissue_masks/output_tissue_mask_{sample}.png', bin_mask)
        
        if(binmask):
            blur_binmask = final_output[:,:,1]
            blur_binmask[blur_binmask != 0] = 255

            cv2.imwrite(f'{outdir}/blurmask_{sample}.png', blur_binmask)

        #write mask to output
        with TiffWriter(f'{outdir}/output_{sample}.tif', bigtiff=True, imagej=True) as tif:
        
            tif.save(final_output, compress=6, tile=(16,16) ) 
        
    

    try:
        ###### Format output dictionary and save to csv file
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        results_df.columns = ["total_blurry_perc", "mildly_blurry_perc", "highly_blurry_perc", "70_perc_white", "50_perc_white", "30_perc_white", "10_perc_white", "patch_size_used", "native_magnification_level",
        "blur_detection_magnification", "npixels_at_8??pp", "processing_time", "file_path"]
        if failed_slides > 0:
            logger.warning(f"WARNING: {failed_slides} slides were skipped due to lack of openslide compatibility")
    except ValueError:
        logger.error(f"ERROR: No slide in glob pattern is openslide compatible")
        sys.exit(1)

    return results_df
