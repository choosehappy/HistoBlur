import numpy as np
from  skimage.color import rgb2gray
import torch
import time
import math
from torchvision.models import DenseNet
from skimage.morphology import disk, remove_small_objects
from scipy.ndimage import minimum_filter
import cv2
import os
from tqdm import tqdm
import openslide
from multiprocessing import Pool, cpu_count

def generate_mask(image, min_size_object):
    """generates a mask with thresholding, does not apply erosion"""

    disk_size = 5
    threshold = 200
    img = rgb2gray(image)
    img = (img * 255).astype(np.uint8)
    selem = disk(disk_size)
    imgfilt = minimum_filter(img, footprint=selem)
    up_tresh = imgfilt < threshold
    down_thresh =  imgfilt > 0
    mask= np.float32(remove_small_objects(np.logical_and(up_tresh, down_thresh), min_size = min_size_object))
    
    return mask


def get_magnification_from_mpp(native_mpp):
    """Takes native mpp and extracts magnification"""

    reference_values_mpp = [0.125, 0.25, 0.5, 1.0, 2.0]
    mag_values = [80.0, 40.0, 20.0, 10.0, 5.0]

    # Use np.isclose to find the closest matching reference mpp value
    is_close_to_reference = [np.isclose(ref_mpp, native_mpp, atol=0.05) for ref_mpp in reference_values_mpp]

    # If there's a close match, take its corresponding magnification
    mag = mag_values[is_close_to_reference.index(True)] if any(is_close_to_reference) else None

    return mag


def get_layer_for_mpp(osh, desired_mpp):
    """
    Finds the highest-MPP layer with an MPP > desired_mpp, rescales dimensions to match that layer.
    """
    
    # If mpp is not provided in file
    
    mpp = float(osh.properties.get('openslide.mpp-x', 0.25))  # Default to 0.25 if mpp-x property does not exist
    
    downsamples = osh.level_downsamples
    img_dims = osh.level_dimensions
        
    mpps = [ds*mpp for ds in downsamples]
    
    diff_mpps = [float(desired_mpp) - m for m in mpps]
    valid_layers = [(index, diff_mpp) for index, diff_mpp in enumerate(diff_mpps) if diff_mpp >= 0]
    valid_diff_mpps = [v[1] for v in valid_layers]
    valid_layers= [v[0] for v in valid_layers]
    
    if not valid_layers:
        warn_message = 'Desired_mpp is lower than minimum image MPP of ' + str(min(mpps))
        print(warn_message)
        target_layer = mpps.index(min(mpps))
    else:
        target_layer = valid_layers[valid_diff_mpps.index(min(valid_diff_mpps))]
        
        
    return target_layer



def getMag(osh, slide, logger):
    """Extracts magnification from openslide WSI"""
    mag = osh.properties.get("openslide.objective-power", osh.properties.get("aperio.AppMag", "NA"))
    if (mag == "NA"):
        try:
            logger.warning(f"{slide} - No magnification found, estimating using mpp value ")
            mag = get_magnification_from_mpp(float(osh.properties["openslide.mpp-x"]))
        except:
            logger.warning(f"{slide} - No magnification or openslide mpp value found, assuming 40X")
            mag = 40
    
    mag = float(mag)

    return mag


def get_mask_and_level(osh, mpp):
    """" Takes slide opened with openslide and returns image array at specified mpp level"""
    mask_level = get_layer_for_mpp(osh, mpp)
    dims = osh.level_dimensions[mask_level]
    img = osh.read_region((0, 0), mask_level, dims)

    return [np.asarray(img)[:, :, 0:3], mask_level]

def divide_batch(*lists, batch_size):
    for i in range(0, len(lists[0]), batch_size):
        yield [lst[i:i + batch_size] for lst in lists]


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

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpuid}')
    else:
        logger.warning("GPU not found, generating output with CPU.")
        device = torch.device('cpu')
    
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
    coords = []
    for y in tqdm(range(0, osh.level_dimensions[0][1], round(patch_size * osh.level_downsamples[level])), desc="outer"):
        for x in tqdm(range(0, osh.level_dimensions[0][0], round(patch_size * osh.level_downsamples[level])), desc=f"innter {y}", leave=False):

            maskx = int(x // ds_mask)
            masky = int(y // ds_mask)

            if (maskx >= mask.shape[1] or masky >= mask.shape[0]) or mask[masky:masky+t_s_mask, maskx:maskx+t_s_mask].mean() < 0.2:
                continue
            coords.append((x, y))

    return coords


def process_coordinates(args):
    coords, slide_path, patch_size, level, ratio_white = args  # Unpacking arguments

    osh = openslide.OpenSlide(slide_path)
    patches_local = []
    coords_local = []  # A list of tuples (x, y) instead of two separate lists

    for coord in coords:
        io = np.asarray(osh.read_region(coord, level, (patch_size, patch_size)))[:, :, 0:3]
        white_pixel_mask = (io[:, :, 0] > 232) & (io[:, :, 1] > 232) & (io[:, :, 2] > 232)
        white_pixel_sum = white_pixel_mask.sum()

        if white_pixel_sum <= int(patch_size * patch_size * ratio_white):
            patches_local.append(io)
            coords_local.append(coord)  # Appending the coordinate tuple

    return patches_local, coords_local


def extract_patches_from_coordinates(slide_path, coords, patch_size, ratio_white, level, num_cores=None):
    # Group coordinates into chunks for multiprocessing
    chunk_size = math.ceil(len(coords) / num_cores)
    chunks = [chunk[0] for chunk in divide_batch(coords, batch_size=chunk_size)]
    
    args_list = [(chunk, slide_path, patch_size, level, ratio_white) for chunk in chunks]

    with Pool(processes=num_cores or cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_coordinates, args_list), total=len(chunks), desc="Processing Chunks"))

    # Gather results
    patches = [item for r in results for item in r[0]]
    coords = [item for r in results for item in r[1]]

    return patches, coords

