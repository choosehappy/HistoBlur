
import tables
import numpy as np

import sklearn.feature_extraction.image
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from  skimage.color import rgb2gray
import cv2

from skimage.morphology import disk
from skimage.filters import rank
import os
import glob
import openslide
import random
import os,sys

from WSI_handling import wsi
from typing import NamedTuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from albumentations import *
from albumentations.pytorch import ToTensor
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian

from tensorboardX import SummaryWriter
from torchsummary import summary

import scipy

import time
import math

# coding: utf-8

class Args_train(NamedTuple):
    """ Command-line arguments """
    mode: str
    input_wsi: str
    patchsize: int
    batchsize: int
    outdir: str
    gpuid: int
    level: int
    dataset_name: str
    enablemask: bool
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
    
    

#Created by Rahul Nair and Petros Liakopoulos

########## Helper functions

#this is the level the WSI will be read, keep this level constant in output generation

def divide_size(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 

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

######### PYTABLES creation function

def create_pytables(files, phases, dataname, patch_size, trainsize, valsize, sample_level, output_dir, mask_bool=True):
    """ Create one pytables file for training set and one for validation"""
                    
    #ensure that no table is open
    #tables.file._open_files.close_all()
    seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the cross validation setup
    random.seed(seed) # set the seed
    print(f"random seed (note down for reproducibility): {seed}")




    # parameters for pytables creation
    img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
    filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just incase we need it later,

    block_shape=np.array((patch_size,patch_size,3)) #block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
    filters=tables.Filters(complevel=6, complib='zlib') #we can also specify filters, such as compression, to improve storage speed

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



    for filei in tqdm(files): #now for each of the files
        osh  = openslide.OpenSlide(filei)
        osh_mask  = wsi(filei)
        mask_level_tuple = osh_mask.get_layer_for_mpp(8)
        mask_level = mask_level_tuple[0]


        if(mask_bool):
            mask=cv2.imread(os.path.splitext(filei)[0]+'.png') #--- assume mask has png ending in same directory

        else:

            img = osh.read_region((0, 0), mask_level, osh.level_dimensions[mask_level])
            img = np.asarray(img)[:, :, 0:3]
                
            disk_size = 5
            threshold = 200
            img = rgb2gray(img)
            img = (img * 255).astype(np.uint8)
            selem = disk(disk_size)
            imgfilt = rank.minimum(img, selem)
            mask= imgfilt < threshold   



        [rs,cs]=mask.nonzero() # mask coords in which tissue is present

        for phase in phases:
            [prs,pcs]=random_subset(rs,cs,min(max_number_samples[phase],len(rs))) #randomly shuffle coords of areas that have tissue


            for i, (r,c) in tqdm(enumerate(zip(prs,pcs)),total =len(prs), desc=f"innter2-{phase}", leave=False): #iterate over coords. Coords in np array are Height x width vs width x Height in Openslide

                io = np.asarray(osh.read_region((int(c*osh.level_downsamples[mask_level]), int(r*osh.level_downsamples[mask_level])), sample_level,
                                                (patch_size, patch_size)))
                img = np.asarray(io)[:, :, 0:3]
                io = io[:, :, 0:3]  # remove alpha channel


                imgg=rgb2gray(img)
                mask2=np.bitwise_and(imgg>0 ,imgg <230/255) #
                plt.imshow(imgg,cmap="gray")



                if np.count_nonzero(mask2 == True) > (patch_size * patch_size) /8:
                     storage[phase]["imgs"].append(io[None,::])
                     storage[phase]["filenames"].append([f'{filei}_{r}_{c}']) #add the filename to the storage array

        osh.close()
    tables.file._open_files.close_all()
    return [["train", f"{output_dir}/{dataname}_train.pytable"],["val", f"{output_dir}/{dataname}_val.pytable"]]
    



def train_model(path_to_pytables_list, dataname, gpuid, batch_size, patch_size, phases, num_epochs, output_dir, validation_phases, sample_level):
    ############## Model training
    
    ## Model params
    in_channels= 3  #input channel of the data, RGB = 3
    growth_rate=16 #change from 32 
    block_config=(2, 2, 2, 2)
    num_init_features=64
    bn_size=4
    drop_rate=0

    

    blurparams = {0:[0,0],
                  1:[1,3],
                  2:[5,7]}
    nclasses=len(blurparams)
    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                     num_init_features=num_init_features, 
                     bn_size=bn_size, 
                     drop_rate=drop_rate, 
                     num_classes=nclasses).to(device)

    ######## Image transformations
    img_transform = Compose([
            RandomScale(scale_limit=0.1,p=.9),
            PadIfNeeded(min_height=patch_size,min_width=patch_size),        
            VerticalFlip(p=.5),
            HorizontalFlip(p=.5),
            GaussNoise(p=.5, var_limit=(10.0, 50.0)),
            GridDistortion(p=.5, num_steps=5, distort_limit=(-0.3, 0.3), 
                            border_mode=cv2.BORDER_REFLECT),
            ISONoise(p=.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
            RandomBrightness(p=.5, limit=(-0.2, 0.2)),
            RandomContrast(p=.5, limit=(-0.2, 0.2)),
            RandomGamma(p=.5, gamma_limit=(80, 120), eps=1e-07),
            MultiplicativeNoise(p=.5, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
            HueSaturationValue(hue_shift_limit=20,sat_shift_limit=10,val_shift_limit=10,p=.9),
            Rotate(p=1, border_mode=cv2.BORDER_REFLECT),
            RandomCrop(patch_size,patch_size),
            ToTensor()
        ])

    ####### Preparing the dataLoaders for each phase
    dataset={}
    dataLoader={}

    for phase in path_to_pytables_list: #now for each of the phases, we're creating the dataloader
                         #interestingly, given the batch size, i've not seen any improvements from using a num_workers>0

        dataset[phase[0]]=Dataset(phase[1], blurparams, img_transform=img_transform)
        dataLoader[phase[0]]=DataLoader(dataset[phase[0]], batch_size=batch_size, 
                                    shuffle=True, num_workers=12,pin_memory=True) 
        print(f"{phase} dataset size:\t{len(dataset[phase[0]])}")

    optim = torch.optim.Adam(model.parameters()) #adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
    criterion = nn.CrossEntropyLoss()
    
    writer=SummaryWriter() #open the tensorboard visualiser
    best_loss_on_test = np.Infinity
    
    ####### Training the model
    print(device)
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
        #zero out epoch based performance variables 
        all_acc = {key: 0 for key in phases} 
        all_loss = {key: torch.zeros(0).to(device) for key in phases} #keep this on GPU for greatly improved performance
        cmatrix = {key: np.zeros((nclasses,nclasses)) for key in phases}

        for phase in phases: #iterate through both training and validation states

            if phase == 'train':
                model.train()  # Set model to training mode
            else: #when in eval mode, we don't want parameters to be updated
                model.eval()   # Set model to evaluate mode

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

        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                     epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]),end="")    

        #if current loss is the best we've seen, save model state with all variables
        #necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **")
            state = { 'level': sample_level,
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


            torch.save(state, f"{output_dir}/{dataname}_densenet_best_model.pth")
        else:
            print("")


    return f"{output_dir}/{dataname}_densenet_best_model.pth"


