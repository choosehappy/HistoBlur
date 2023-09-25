import numpy as np
from  skimage.color import rgb2gray
import torch
import time
import math
from torchvision.models import DenseNet
from skimage.morphology import disk
from skimage.filters import rank
import cv2
import os
from WSI_handling import wsi
from tqdm import tqdm
import openslide
from multiprocessing import Pool

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

def generate_mask_stringent(image):
    """Function that generates mask without disk expansion"""
    imgg=rgb2gray(image)
    mask=np.bitwise_and(imgg>0 ,imgg <230/255)
    mask = np.float32(mask)
    
    return mask


def get_magnification_from_mpp(native_mpp):
    """Takes native mpp and extracts magnification"""

    #slide.levels._levels[0].mpp.width

    reference_values_mpp = [0.125, 0.25, 0.5, 1.0, 2.0]
    mag_values = [80.0, 40.0, 20.0, 10.0, 5.0]

    # Find the index of the nearest reference value
    nearest_index = min(range(len(reference_values_mpp)), key=lambda i: abs(reference_values_mpp[i] - native_mpp))

    mag = mag_values[nearest_index]

    return mag


def getMag(osh, slide, logger):
    """Extracts magnification from openslide WSI"""
    mag = osh.properties.get("openslide.objective-power", "NA")
    if (mag == "NA"):  # openslide doesn't set objective-power for all SVS files: https://github.com/openslide/openslide/issues/247
        mag = osh.properties.get("aperio.AppMag", "NA")
    if (mag == "NA"):
        try:
            logger.warning(f"{slide} - No magnification found, estimating using mpp value ")
            mag = get_magnification_from_mpp(float(osh.properties["openslide.mpp-x"]))
        except:
            logger.warning(f"{slide} - No magnification or openslide mpp value found, assuming 40X")
            mag = 40
    
    mag = float(mag)

    return mag


def get_mask_and_level(slide, mpp):
    """" Takes slide opened with openslide and returns image array at specified mpp level"""
    fname=slide
    osh_mask  = wsi(fname)
    mask_level_tuple = osh_mask.get_layer_for_mpp(mpp)
    mask_level = mask_level_tuple[0]
    img = osh_mask.read_region((0, 0), mask_level, osh_mask["img_dims"][mask_level])

    return [np.asarray(img)[:, :, 0:3], mask_level]


def divide_batch(arr, batch_size):
    # Function to divide an array into batches
    return [arr[i:i + batch_size] for i in range(0, len(arr), batch_size)]

def custom_cmap(labels):
    colors = {
        0: [0, 255, 0],    # Class 0
        1: [0, 0, 255],    # Class 1
        2: [255, 0, 0]     # Class 2
    }
    return np.array([colors[label] for label in labels])
def random_subset(a, b, nitems):
    assert len(a) == len(b)
    idx = np.random.randint(0,len(a),nitems)
    return a[idx], b[idx]

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def generate_mask_stringent(image):
    """Function that generates mask with erosion, this removes very small regions of spur pixels"""
    imgg=rgb2gray(image)
    mask=np.bitwise_and(imgg>0 ,imgg <230/255)
    mask = np.float32(mask)
    

    return mask

def check_if_within(max_coords, new_coords, tile_size):
    max_x, max_y = max_coords
    new_x, new_y = new_coords

    # Check if the new coordinates and tile size are within the max coordinates
    if (max_x > new_x + tile_size) and (max_y > new_y + tile_size):
        return True
    else:
        return False

def get_level_for_targeted_mag_openslide(osh, slide, training_mag, logger):
    """
    Get the appropriate level for the requested magnification.

    Args:
        osh (OpenSlide): An OpenSlide object representing the slide.
        slide (string): path to slide file
        training_mag (float): The requested magnification level.

    Returns:
        int: The best level for the requested magnification.
        float: native magnification level of slide.
    """
    # Getting openslide level for requested magnification
    native_mag = getMag(osh, slide, logger)
    targeted_mag = training_mag
    down_factor = native_mag / targeted_mag
    relative_down_factors_idx = [np.isclose(x/down_factor, 1, atol=.01) for x in osh.level_downsamples]
    level = np.where(relative_down_factors_idx)[0]

    if level.size:
        level = level[0]
    else:
        level = 0

    return [level, native_mag, targeted_mag]


def load_densenet_model(gpuid, model, logger):
    """load densent model and extract magnification used during training"""

    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    if str(device) == 'cpu':
        logger.warning("Generating output with CPU, this might take longer. Switching to GPU is recommended")
    
    logger.info(f"model used: {model}")

    checkpoint = torch.load(model, map_location=lambda storage, loc: storage)

    training_mag = checkpoint["magnification"]

    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["nclasses"]).to(device)




    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    return [model, training_mag, device]

def resize_mask(slide, img):
    """Takes .png mask with same name as slide and resizes it to mask at given mpp"""

    mask_img=cv2.imread(os.path.splitext(slide)[0]+'.png', cv2.IMREAD_GRAYSCALE) #--- assume mask has png ending in same directory
    width = int(img.shape[1] )
    height = int(img.shape[0])
    dim = (width, height)
    mask_resize = cv2.resize(mask_img,dim)
    mask = np.float32(mask_resize)
    mask /= 255

    return mask


def extract_coords_inside_mask(patch_size, level, osh, mask, ds_mask, t_s_mask):
    coordsx = []
    coordsy = []
    for y in tqdm(range(0,osh.level_dimensions[0][1],round(patch_size * osh.level_downsamples[level])), desc="outer"):
        for x in tqdm(range(0,osh.level_dimensions[0][0],round(patch_size * osh.level_downsamples[level])), desc=f"innter {y}", leave=False):

            maskx=int(x//ds_mask)
            masky=int(y//ds_mask)


            if(maskx >= mask.shape[1] or masky >= mask.shape[0]) or mask[masky:masky+t_s_mask,maskx:maskx+t_s_mask].mean() < 0.2:
                continue
            coordsx.append(x)
            coordsy.append(y)
            
        
    return coordsx, coordsy

def process_coordinates(args):
    coords, slide_path, patch_size, level, ratio_white = args  # Unpacking arguments

    osh = openslide.OpenSlide(slide_path)
    patches_local = []
    xs_local = []
    ys_local = []

    for (x, y) in coords:
        io = np.asarray(osh.read_region((x, y), level, (patch_size, patch_size)))[:, :, 0:3]
        white_pixel_mask = (io[:, :, 0] > 232) & (io[:, :, 1] > 232) & (io[:, :, 2] > 232)
        white_pixel_sum = np.sum(white_pixel_mask)

        if white_pixel_sum <= int(patch_size * patch_size * ratio_white):
            patches_local.append(io)
            xs_local.append(x)
            ys_local.append(y)

    return patches_local, xs_local, ys_local

def extract_patches_from_coordinates(slide_path, coordsx, coordsy, patch_size, ratio_white, level, num_cores=None):
    # Group coordinates into chunks for multiprocessing
    total_coords = list(zip(coordsx, coordsy))
    chunk_size = len(total_coords) // num_cores
    chunks = [total_coords[i:i+chunk_size] for i in range(0, len(total_coords), chunk_size)]
    
    args_list = [(chunk, slide_path, patch_size, level, ratio_white) for chunk in chunks]

    with Pool(processes=num_cores or cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_coordinates, args_list), total=len(chunks), desc="Processing Chunks"))

    # Gather results
    patches = sum((r[0] for r in results), [])
    xs = sum((r[1] for r in results), [])
    ys = sum((r[2] for r in results), [])

    return patches, xs, ys
