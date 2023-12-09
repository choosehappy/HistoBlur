import tables
import numpy as np
import logging

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from  skimage.color import rgb2gray
import cv2

import os
import openslide
import random
import os,sys
from skimage.morphology import disk, remove_small_objects
from scipy.ndimage import minimum_filter
from typing import NamedTuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from albumentations import *
from albumentations.pytorch import ToTensor
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from .utils import *
import scipy

import time
import math

# coding: utf-8

### logger

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

class Args_train(NamedTuple):
    """ Command-line arguments """
    mode: str
    input_wsi: str
    batchsize: int
    outdir: str
    gpuid: int
    magnification: float
    min_size_object: int
    dataset_name: str
    mask_path: str
    trainsize: int
    valsize: int
    epochs: int
    training: str
    validation: str
    





class Dataset(object):
    """Class for the Dataloader object"""
    def __init__(self, fname, blur_params ,img_transform=None):
        #nothing special here, just internalizing the constructor parameters
        self.fname=fname
        self.blur_params=blur_params
        self.img_transform=img_transform
        
        with tables.open_file(self.fname,'r') as db:
            self.nitems=db.root.imgs.shape[0]
        
        self.imgs = None
        
    def __getitem__(self, index):
        #opening should be done in __init__ but seems to be
        #an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

        with tables.open_file(self.fname,'r') as db:
            self.imgs=db.root.imgs
            ##apply blur here
            #get the requested image  from the pytable
            label = random.choice([0,1,2])
            img=self.imgs[index,:,:,:]
           
            rand_no=random.randint(self.blur_params[label][0],self.blur_params[label][1])
            new_img =  gaussian_filter(img,sigma=(rand_no,rand_no,0)) if rand_no>0 else img
            
        if self.img_transform:
            img_new = self.img_transform(image=new_img)['image']

        return img_new, label, img
    def __len__(self):
        return self.nitems
    
######### PYTABLES creation function

def create_pytables(slides, phases, dataname, trainsize, valsize, magnification_level, min_size_object, output_dir, mask_path):
    """ Create one pytables file for training set and one for validation"""
    #global variables for thresholding
    mask_threshold_white = 0.9          
    white_area_threshold_in_patch = 0.7
    # parameters for pytables creation
    patch_size = 256
    img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
    filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just incase we need it later,

    block_shape=np.array((patch_size,patch_size,3)) #block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
    filters=tables.Filters(complevel=6, complib='blosc') #we can also specify filters, such as compression, to improve storage speed

    max_number_samples={"train":trainsize,"val":valsize}  #Sample numbers for train and validation set

    storage={} #holder for future pytables
    for phase in phases: #now for each of the phases, we'll loop through the files
        storage[phase]={}
        hdf5_file = tables.open_file(f"{output_dir}/{dataname}_{phase}.pytable", mode='w') #open the respective pytable
        storage[phase]["filenames"] = hdf5_file.create_earray(hdf5_file.root, 'filenames', filenameAtom, (0,)) #create the array for storage

        storage[phase]["imgs"]= hdf5_file.create_earray(hdf5_file.root, "imgs", img_dtype,  
                                                  shape=np.append([0],block_shape), 
                                                  chunkshape=np.append([1],block_shape),
                                                 filters=filters)



    for slide in tqdm(slides): #now for each of the files
        try:
            osh  = openslide.OpenSlide(slide)
        except ValueError:
            logger.exception(f"ERROR: {slide} not Openslide compatible")
            sys.exit(1)


        level, native_mag, targeted_mag = get_level_for_targeted_mag_openslide(osh, slide, magnification_level, logger)
        
        img, mask_level = get_mask_and_level(osh, 8)
        
        samplebase = os.path.basename(slide)
        sample = os.path.splitext(samplebase)[0]
        

        if mask_path != "":
            mask = resize_mask(sample, mask_path, img) #if mask is provided, use it
        
        else:
            print("Generating mask")
            mask_final = generate_mask(img, min_size_object) #call mask generation function
            mask = mask_final 
    

        tissue_size_pixels = mask.sum()
        logger.info(f"{tissue_size_pixels} at 8mpp")

        if int(tissue_size_pixels) == 0:
            logger.critical("No tissue remaining after mask generation, please select a slide with sufficient tissue")
            sys.exit(1)
        elif int(tissue_size_pixels) < 500000:
            logger.warning("Warning: low tissue quantitiy detected. We recommend using a slide with more tissue")

        [rs,cs]=mask.nonzero() # mask coords in which tissue is present

        for phase in phases:
            [prs,pcs]=random_subset(rs,cs,min(max_number_samples[phase],len(rs))) #randomly shuffle coords of areas that have tissue


            for i, (r,c) in tqdm(enumerate(zip(prs,pcs)),total =len(prs), desc=f"innter2-{phase}", leave=False):

                io = np.asarray(osh.read_region((int(c*osh.level_downsamples[mask_level]), int(r*osh.level_downsamples[mask_level])), level,
                                                (patch_size, patch_size)))[:, :, 0:3]
            

                imgg=rgb2gray(io)
                mask2=np.bitwise_and(imgg>0 ,imgg <mask_threshold_white) #



                if np.count_nonzero(mask2 == True) > (patch_size * patch_size) * white_area_threshold_in_patch:
                     storage[phase]["imgs"].append(io[None,::])
                     storage[phase]["filenames"].append([f'{samplebase}_{r}_{c}']) #add the filename to the storage array

        bin_mask = mask*255
        bin_mask = bin_mask.astype(np.uint8)
        cv2.imwrite(f'{output_dir}/tissue_masks/output_tissue_mask_{sample}.png', bin_mask)

        osh.close()
    tables.file._open_files.close_all()


    return [["train", f"{output_dir}/{dataname}_train.pytable"],["val", f"{output_dir}/{dataname}_val.pytable"]]
    



def train_model(path_to_pytables_list, dataname, gpuid, batch_size, phases, num_epochs, output_dir, validation_phases, magnification_level):
    ############## Model training
    
    ## Model params
    patch_size = 256
    in_channels= 3  #input channel of the data, RGB = 3
    growth_rate=16 #change from 32 
    block_config=(2, 2, 2, 2)
    num_init_features=64
    bn_size=4
    drop_rate=0

    
    #sigma limits to gaussian noise being applied during augmentation
    blurparams = {0:[0,0],
                  1:[1,3],
                  2:[5,7]}
    
    nclasses=len(blurparams)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpuid}')
    else:
        logger.warning("GPU not found, generating output with CPU.")
        device = torch.device('cpu')

    model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                     num_init_features=num_init_features, 
                     bn_size=bn_size, 
                     drop_rate=drop_rate, 
                     num_classes=nclasses).to(device)

    ######## Image transformations
    img_transform = Compose([
            RandomScale(scale_limit=0.05,p=.9),
            PadIfNeeded(min_height=patch_size,min_width=patch_size),        
            VerticalFlip(p=.5),
            HorizontalFlip(p=.5),
            GaussNoise(p=.5, var_limit=(10.0, 50.0)),
            GridDistortion(p=.5, num_steps=5, distort_limit=(-0.2, 0.2), 
                            border_mode=cv2.BORDER_REFLECT),
            ISONoise(p=.5, intensity=(0.2, 0.4), color_shift=(0.01, 0.05)),
            RandomBrightness(p=.5, limit=(-0.1, 0.1)),
            RandomContrast(p=.5, limit=(-0.1, 0.1)),
            RandomGamma(p=.5, gamma_limit=(90, 110), eps=1e-07),
            MultiplicativeNoise(p=.5, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
            HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10,val_shift_limit=10,p=.9),
            Rotate(p=1, border_mode=cv2.BORDER_REFLECT),
            RandomCrop(patch_size,patch_size),
            ToTensor()
        ])

    ####### Preparing the dataLoaders for each phase
    dataset={}
    dataLoader={}

    for phase in path_to_pytables_list: #each element of this list contains a string "train" or "test" at element 1 and the path to the pytables as element 2

        dataset[phase[0]]=Dataset(phase[1], blurparams, img_transform=img_transform) #dataset
        dataLoader[phase[0]]=DataLoader(dataset[phase[0]], batch_size=batch_size, 
                                    shuffle=True, num_workers=12,pin_memory=True) 
        logger.info(f"{phase} dataset size:\t{len(dataset[phase[0]])}")

    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-5) #adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
    criterion = nn.CrossEntropyLoss()
    
    writer=SummaryWriter(log_dir=f"{output_dir}/logs") #open the tensorboard visualiser
    best_loss_on_test = np.Infinity
    
    ####### Training the model
    logger.info(device)
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
        #zero out epoch based performance variables 
        all_acc = {key: 0 for key in phases} 
        all_loss = {key: torch.zeros(0).to(device) for key in phases} #keep this on GPU for greatly improved performance
        cmatrix = {key: np.zeros((nclasses,nclasses)) for key in phases}

        for phase in phases: #iterate through both training and validation states

            model.train(phase == 'train')

            for ii , (X, label, img_orig) in enumerate(dataLoader[phase]): #for each of the batches
                X = X.to(device)  # [Nbatch, 3, H, W]
                label = label.type('torch.LongTensor').to(device)  # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)

                with torch.set_grad_enabled(phase == 'train'): #dynamically set gradient computation, in case of validation, this isn't needed
                                                                #disabling is good practice and improves inference time

                    prediction = model(X)  # [N, Nclass]
                    loss = criterion(prediction, label)


                    if phase=="train": #in case we're in train mode, need to do back propogation
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        train_loss = loss


                    all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))

                    if phase in validation_phases: #if this phase is part of validation, compute confusion matrix
                        p=prediction.detach().cpu().numpy()
                        cpredflat=np.argmax(p,axis=1).flatten()
                        yflat=label.cpu().numpy().flatten()

                        confusion_matrix = scipy.sparse.coo_matrix( (np.ones(yflat.shape[0], dtype=np.int64), (yflat, cpredflat)),
                                shape=(nclasses, nclasses), dtype=np.int64, ).toarray()

                        cmatrix[phase] = cmatrix[phase] + confusion_matrix  # confusion_matrix(yflat, cpredflat, labels=range(2))



            all_acc[phase]=(cmatrix[phase]/cmatrix[phase].sum()).trace()
            all_loss[phase] = all_loss[phase].cpu().numpy().mean()

            #save metrics to tensorboard
            writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
            if phase in validation_phases:
                writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
                for r in range(nclasses):
                    for c in range(nclasses): #essentially write out confusion matrix
                        writer.add_scalar(f'{phase}/{r}{c}', cmatrix[phase][r][c],epoch)

        logger.info('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                     epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]))    

        #if current loss is the best we've seen, save model state with all variables
        #necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            logger.info("  **")
            state = { 'magnification': magnification_level,
             'epoch': epoch + 1,
             'model_dict': model.state_dict(),
             'optim_dict': optim.state_dict(),
             'best_loss_on_test': all_loss,
             'in_channels': in_channels,
             'growth_rate':growth_rate,
             'block_config':block_config,
             'num_init_features':num_init_features,
             'bn_size':bn_size,
             'drop_rate':drop_rate,
             'nclasses':nclasses}


            torch.save(state, f"{output_dir}/{dataname}_densenet_best_model_{magnification_level}X.pth")
        else:
            logger.info("")



    


    return f"{output_dir}/{dataname}_densenet_best_model_{magnification_level}X.pth"


