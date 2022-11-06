## HistoBlur


## Warning:
This tool is still under development and has only been tested on Linux Ubuntu. A docker container and full documentation will be made available shortly.


# Introduction

HistoBlur is a deep learning based tool that allows for the fast and accurate detection of blurry regions in Whole Slide Images.

HistoBlur has two modes.

Training mode:

```
usage: HistoBlur train [-h] [-f INPUT_WSI] [-p PATCHSIZE] [-s BATCHSIZE] [-o OUTDIR] [-d DATASET_NAME] [-i GPUID] [-l LEVEL] [-m] [-t TRAINSIZE] [-v VALSIZE] [-e EPOCHS] [-y TRAINING] [-z VALIDATION]

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_WSI, --input_wsi INPUT_WSI
                        WSI to use for training. ex: path/to/wsi.svs
  -p PATCHSIZE, --patchsize PATCHSIZE
                        patchsize to use for training, default 256
  -s BATCHSIZE, --batchsize BATCHSIZE
                        batchsize for controlling GPU memory usage, default 128
  -o OUTDIR, --outdir OUTDIR
                        output directory, default 'histoblur_output'
  -d DATASET_NAME, --dataset_name DATASET_NAME
                        name of dataset, default 'blur_detection'
  -i GPUID, --gpuid GPUID
                        id of gpu to use, default 0
  -l LEVEL, --level LEVEL
                        openslide level to use, default 1
  -m, --enablemask      provide external mask (assumes .png extension file with same slide name)
  -t TRAINSIZE, --trainsize TRAINSIZE
                        size of training set, default 10000
  -v VALSIZE, --valsize VALSIZE
                        size of validation set, default 2000
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train for, default 100
  -y TRAINING, --training TRAINING
                        pre-generated pytables file with training set
  -z VALIDATION, --validation VALIDATION
                        pre-generated pytables file with training set
```

Detect mode:

```
usage: HistoBlur detect [-h] -f INPUT_WSI [-s BATCHSIZE] [-o OUTDIR] -m MODEL [-i GPUID] [-t]

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_WSI, --input_wsi INPUT_WSI
                        directory to WSI image file(s)
  -s BATCHSIZE, --batchsize BATCHSIZE
                        batchsize for controlling GPU memory usage ,default 16
  -o OUTDIR, --outdir OUTDIR
                        output directory, default 'histoblur_output'
  -m MODEL, --model MODEL
                        model
  -i GPUID, --gpuid GPUID
                        id of gpu to use, default 0
  -t, --enablemask      provide external mask (assumes .png extension file with same slide name)
```


# Training

In order to detect blurry regions, a DenseNet model needs to be trained for the specific stain being analyzed.
For that, the user needs to provide one clear non-blurry Whole Slide Image (WSI) for the staining of choice.

In training mode, patches from the clear non-blurry WSI will be extracted and gaussian blur will be applied to some of them. A dataset will be created
in which a patch can either be clear, mildly blurry or highly blurry. The associated patches and labels with the artificially applied levels of blurriness
are used as a training set for the model.

The model then trains for the indicated number of epochs and the saved model weights for the best performing model are outputed in the output directory.
The trained model can then be used to detect blurry areas in other WSI of the same staining type.

# Detection

Once the model has been trained, the user can then provide other file(s) along with the previous trained model in order to estimate bluriness.

The detection mode will output a tissue mask with non blurry regions appearing in green, mildly blurry regions in blue and highly blurry regions in red.

The tool will also output a csv formatted file with the percentage of mildly and highly blurry regions on the slide.

# Installation
Before installing HistoBlur, the following dependencies need to be installed:

HDF5 (available here https://www.hdfgroup.org/downloads/hdf5/)

Openslide (available here https://openslide.org/download/)

Additionally, a CUDA installation is also required.

Once these dependencies have been setup, Histoblur can be installed.

To install HistoBlur, first clone the directory:

```
git clone https://github.com/choosehappy/HistoBlur.git
cd HistoBlur
```

If you wish to install the tool in a separate conda environment (recommended):

```
conda create --name HistoBlur python=3.8
conda activate HistoBlur
```

Then, run the following:

```
pip install .
```


To ensure that HistoBlur has been succesfully installed, run the following:

```
HistoBlur --help
```



# Examples

Since the tool has been tested at a specific magnification and patch size, if the WSI at your disposal are 40x magnification, we recomend using
default parameters.

To train a model, all you need is a Whole Slide Image of your staining of choice with no blur, we recommend picking a represntative slide with sufficient tissue:
```
HistoBlur train -f "path/to/wsi.svs" -o blur_detection -d blur_detection_histoblur
```

The following command will create a training and testing set from the provided WSI that will be saved in the 'blur_detection' output folder.
The two datasets will be used to train a Densenet for 100 epochs, the model will also be outputed in the designated output folder.

After training has been completed, the training statistics can be visualized using tensordboard:

Inside the output directory, run the following
```
tensorboard --logdir logs/
```

As a rule of thumb, a decreasing loss and an increasing accuracy curve suggests that the model has converged.


Once the model has been trained, new output can be generated:

```
Histoblur detect -f "path/to/images_to_test/*.svs" -o blur_quality_control -m blur_detection/blur_detection_histoblur_best_model.pth
```

This command will analyze all the WSIs matching the provided glob pattern and output a csv file with the percentage of bluriness for each
in a csv file in the designated output folder. A mask of the blurry regions for each slide will also be outputted.

# Output

*HistoBlur train will output the following files in the output directory:*

Two pytables file (train and val): _these contain the training and validation set used for the DenseNet model training, they can be reused with the flags -y and -z to retrain the model_

The model (ends with *best_model.pth): _the model that was trained and should be entered after the -m flag of the "detect" mode_

A logs directory containing the training summary: _the tensorboard summary graphs that can be visualized using tensorboard_

A binary tissue mask in a subdirectory named "tissue_masks": _binary tissue mask to ensure that the area used for training is reasonable_


*HistoBlur detect will output the following files in the output directory:*

A binary tissue mask in a subdirectory named "tissue_masks": _these can be used as binary masks for downstream analysis or as a sanity check to ensure that the tissue areas analyzed are reasonable._

A three color blur mask: _This mask gives a visual overview of where the blurry areas are located (Red denotes high blur, blue denotes medium blur and green denotes no blur)._
_These can also be used avoid blurry areas in downstream analysis._

A csv file: _containing the file names, blur percentages, openslide magnification level and patch size used for the analysis._


