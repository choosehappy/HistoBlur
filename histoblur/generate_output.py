#Code by Rahul Nair and Petros Liakopoulos
#Import packages
import numpy as np
import matplotlib.pyplot as plt
from WSI_handling import wsi

import sklearn.feature_extraction.image
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank

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
from os import path

import openslide
from tifffile import TiffWriter

import ttach as tta
from typing import NamedTuple

# ### Args class

class Args_Detect(NamedTuple):
    """ Command-line arguments """
    mode: str
    input_wsi: str
    patchsize: int
    batchsize: int
    outdir: str
    model: str
    gpuid: int
    enablemask: bool

    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def generate_output(images, gpuid, model, outdir, enablemask, batch_size, patch_size):
    """"Function that generates output """


    print("Launching blur detection analysis")
    ########## Loading Densenet model

    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

    level_training = checkpoint["level"]

    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["nclasses"]).to(device)




    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    print("Loading model")
    print(f"total model params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")



    ##### create results dictionary
    results_dict = {}
    
    
    ##### Iterate through images and generate output
    for slide in images:
        level = level_training
        start = time.time()
        samplebase = os.path.basename(slide)
        sample = os.path.splitext(samplebase)[0]
        print(f"processing file: {slide}")
        fname=slide
        osh_mask  = wsi(fname)
        mask_level_tuple = osh_mask.get_layer_for_mpp(8)
        mask_level = mask_level_tuple[0]
        img = osh_mask.read_region((0, 0), mask_level, osh_mask["img_dims"][mask_level])


        def divide_batch(l, n): 
            for i in range(0, l.shape[0], n):  
                yield l[i:i + n,::] 
                    
        
        if(enablemask):
            mask=cv2.imread(os.path.splitext(slide)[0]+'.png') #--- assume mask has png ending in same directory 
            width = int(img.shape[1] )
            height = int(img.shape[0])
            dim = (width, height)
            mask = cv2.resize(mask,dim)
            mask = np.float32(mask)
            
        
        
        else:
            #add mask creation which skips parts of image
                
                print("Generating mask and estimating tissue size")
                img = np.asarray(img)[:, :, 0:3]
                
                disk_size = 5
                threshold = 200
                img = rgb2gray(img)
                img = (img * 255).astype(np.uint8)
                selem = disk(disk_size)
                imgfilt = rank.minimum(img, selem)
                mask= imgfilt < threshold
                mask_final = imgfilt < threshold
                tissue_size_pixels = sum(sum(mask))
                print(f"{tissue_size_pixels} at 8 μm per pixel")
           
        
        if tissue_size_pixels == 0:
            print("No tissue detected, skipping file")
            continue
        if 0 < tissue_size_pixels < 100000:
            level = 0
            patch_size = 128
            print(f"Low tissue quantity detected, using higher magnification, patch size selected {patch_size}, openslide level {level}")
        elif 99999 < tissue_size_pixels < 200000:
            patch_size = 64
            print(f"patch size selected {patch_size}, openslide level {level}")
        elif 199999 < tissue_size_pixels < 500000:
            patch_size = 128
            print(f"patch size selected {patch_size}, openslide level {level}")
        elif tissue_size_pixels > 499999:
            patch_size = 256
            print(f"patch size selected {patch_size}, openslide level {level}")
         
    

        cmap= matplotlib.cm.tab10

        #ensure that this level is the same as the level used in training
        osh  = openslide.OpenSlide(fname)
        osh.level_dimensions
        ds=int(osh.level_downsamples[level])

        
        #change the stride size to speed up the process, must equal factor between levels 
        stride_size = patch_size #Use non overlapping patches
        tile_size=patch_size*4 #the bigger the tile size the smaller the resolution of the mask. x4 the patch size is a good tradeoff
        nclasses=3


        shape=osh.level_dimensions[level]
        shaperound=[((d//tile_size)+1)*tile_size for d in shape]

        
        npmm=np.zeros((shaperound[1]//stride_size,shaperound[0]//stride_size,3),dtype=np.uint8)
        for y in tqdm(range(0,osh.level_dimensions[0][1],round(tile_size * osh.level_downsamples[level])), desc="outer"):
            for x in tqdm(range(0,osh.level_dimensions[0][0],round(tile_size * osh.level_downsamples[level])), desc=f"innter {y}", leave=False):

                #if skip
                
                
                
                maskx=int(x//osh.level_downsamples[mask_level])
                masky=int(y//osh.level_downsamples[mask_level])
                
                
                if((np.any(maskx>= mask.shape[1])) or np.any(masky>= mask.shape[0]) or not np.any(mask[masky,maskx])): # need to handle rounding error.
                    continue
                
                
                output = np.zeros((0,nclasses,1,1))
                io = np.asarray(osh.read_region((x, y), level, (tile_size,tile_size)))[:,:,0:3] #trim alpha
                
                arr_out=sklearn.feature_extraction.image._extract_patches(io,(patch_size,patch_size,3),stride_size)
                arr_out_shape = arr_out.shape
                arr_out = arr_out.reshape(-1,patch_size,patch_size,3)
                
                    
                
                for batch_arr in divide_batch(arr_out,batch_size):
                
                    arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
                    with torch.no_grad():
                        # ---- get results
                        output_batch = model(arr_out_gpu)

                        # --- pull from GPU and append to rest of output 
                        output_batch = output_batch.detach().cpu().numpy()

                        output_batch_color=cmap(output_batch.argmax(axis=1), alpha=None)[:,0:3]
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
        
        #Calculate perc of bluriness  
        total_classified_patches = sum(sum(final_output[:,:,2] == 253)) + sum(sum(final_output[:,:,1] == 253)) + sum(sum(final_output[:,:,0] == 253)) #count number of patches with 253 in one channel
        total_patches_blurry = sum(sum(final_output[:,:,0] == 253)) + sum(sum(final_output[:,:,2] == 253))
        total_very_blurry = sum(sum(final_output[:,:,0] == 253))
        total_mildly_blurry = sum(sum(final_output[:,:,2] == 253))
        perc_tot = round(total_patches_blurry/total_classified_patches*100, 3)
        perc_very_blurry = round(total_very_blurry/total_classified_patches*100, 3)
        perc_mildly_blurry = round(total_mildly_blurry/total_classified_patches*100, 3)
                                                                            
        
        #add results to dictionary
        results_dict[samplebase] = [perc_tot, perc_mildly_blurry, perc_very_blurry, slide]
        
        #write binary mask
        with TiffWriter(f'{outdir}/tissue_masks/output_tissue_mask_{sample}.tif', bigtiff=True) as tif:
        
            tif.save(np.invert(mask_final), tile=(16,16) ) 
        
        #write mask to output
        with TiffWriter(f'{outdir}/output_{sample}.tif', bigtiff=True, imagej=True) as tif:
        
            tif.save(final_output, compress=6, tile=(16,16) ) 
        end = time.time()
        total = asMinutes(end - start)
        print(f"Slide processed in {total}")
        
    ###### Format output dictionary and save to csv file
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df.columns = ["total_blurry_perc", "mildly_blurry_perc", "highly_blurry_perc", "file_path"]

    return results_df
