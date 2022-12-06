![Histo_blur_git](https://user-images.githubusercontent.com/70012389/200196423-f7764a78-dee8-40f5-be7d-c9ed709af707.png)


## Warning:
This tool is still under development and has only been tested on Linux Ubuntu. A docker container and full documentation will be made available shortly.


# Introduction

HistoBlur is a deep learning based tool that allows for the fast and accurate detection of blurry regions in Whole Slide Images.

HistoBlur has two modes: train and detect

![HistoBlur_animation](https://user-images.githubusercontent.com/70012389/200196599-b6740098-db1a-43e5-8760-03e238ff9ebe.gif)

# How does HistoBlur work?

First, Histoblur takes a non-blurry representative WSI provided by the user and artificially blurs some patches to train a deep learning model.
Then, the trained model can be used to detect blur on any slide with the same stain/scanner combination.

# Installation

**1. Manual installation:**

HistoBlur has the following dependencies:

HDF5 (available here https://www.hdfgroup.org/downloads/hdf5/)

Openslide (available here https://openslide.org/download/)

Additionally, a CUDA installation is also required.


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

To install dependencies:

```
conda install -c anaconda hdf5
conda install -c conda-forge openslide
```

Then, run the following:

```
pip install .
```


**2. Docker container:**

Alternatively, the latest docker can be pulled using the following command:

(note that this also requires local installation of Nvidia CUDA > 11.0 on your machine)

```
docker pull petroslk/histoblur:latest
```


# Simple tutorial

To train a model, just provide a WSI that is large enough and representative (ideally includes tumor, stroma regions etc...)

```
HistoBlur train -f "path/to/wsi.svs" -o blur_detection -d blur_detection_histoblur -l 10.0
```

This will train a model at 10X magnification, which is usually the best tradeoff between accuracy and speed. The model can be found in the output folder (file with .pth extension)

Now, to run HistoBlur on a batch of slides:

```
Histoblur detect -f "path/to/images_to_test/*.svs" -o blur_quality_control -m blur_detection/blur_detection_histoblur_best_model.pth
```

Here, a glob pattern is given with the path to the directory with the slides to analyse **Path has to be given in quotation marks " "**

# Output

**HistoBlur train will output the following files in the output directory:**

Two pytables file (train and val): _these contain the training and validation set used for the DenseNet model training, they can be reused with the flags -y and -z to retrain the model_

The model (ends with *.pth): _the model that was trained and should be entered after the -m flag of the "detect" mode_

A logs directory containing the training summary: _the tensorboard summary graphs that can be visualized using tensorboard_

A binary tissue mask in a subdirectory named "tissue_masks": _binary tissue mask to ensure that the area used for training is reasonable_

The run log ("histoblur.log")

**HistoBlur detect will output the following files in the output directory:**

A binary tissue mask in a subdirectory named "tissue_masks": _these can be used as binary masks for downstream analysis or as a sanity check to ensure that the tissue areas analyzed are reasonable._

A three color blur mask: _This mask gives a visual overview of where the blurry areas are located (Red denotes high blur, blue denotes medium blur and green denotes no blur)._
_These can also be used avoid blurry areas in downstream analysis._

The run log ("histoblur.log")

A csv file: _containing the file names, blur percentages, magnification level and patch size used for the analysis. (for more details see below)_

# Understanding the output csv file

The output csv file has 13 columns with data that can be used for downstream analysis and allow for blurry slides to be detected and flagged without manual input.

The **three first columns (total_blurry_perc, mildly_blurry_perc, highly_blurry_perc)** hold the bluriness percentages that act as the main metric to detect blurry slides

The **next four columns (ending with perc_white)** are the % of patches that were processed for blur detection and that had above N% of white areas in them.

For example, if 70_perc_white = 13.6%, it means that 13.6% of the total patches processed were _at least_ 70% white. 
One thing to note is that for tissue with a lot of white space, white areas introduce additional noise that can be perceived as blur by the model.
Thus, if an entry has a large percentage of patches with high white area percentages, it is likely that more areas will be detected as blurry. This information can
be incorporated to assess the error rate of the result.

Alternatively, the user can select a more stringent white area filter ratio with **-w** and exclude patches above a certain ratio from the results (see More examples section).

The **patch_size_used** column contains the patch size that was used for blur detection, this one is adjusted based on the quantity of tissue detected.

The **native_magnification_level** is the magnification at which the slide was scanned. Note that you can give slides scanned at different magnification, HistoBlur will always analyse blur
at the magnification at which the level was trained.

**blur_detection_magnification** is the magnification level at which the blur was detected. Note that, although HistoBlur will mostly use the same magnification as the training, if a 
small quantity of tissue is detected, it will go to a higher magnification level to allow for more accurate results.

**npixels_at_8μpp** is the number of pixels of tissue detected at 8 micros per pixel. This gives a quick overview of the quantity of tissue present.

**processing_time** is the time it took for the slide to be processed in mins/secs.

**file_path** is the path to the file analyzed.


# More Examples

**Using pretrained models**

If you are working with H&E staining, the pretrained models might work well for you. They can be found in the pretrained_model directory.

```
Histoblur detect -f "path/to/images_to_test/*.svs" -o blur_quality_control -m pretrained_model/blur_detection_densenet_best_model_10.0X.pth
```

**White area thresholding**

By default, HistoBlur will report results on patches with white ratio > 0.9. Nevertheless, white areas can be misinterpreted as blur by the model. 
If you want to have a high accuracy blur detection and exclude patches with white area, you can use a stricter white ratio threshold. Note that
this might mean that some tissue on edges of white regions might be missed.

```
Histoblur detect -f "path/to/images_to_test/*.svs" -o blur_quality_control -m pretrained_model/blur_detection_densenet_best_model_10.0X.pth -w 0.2
```

Here, any patches with a white ratio above 0.2 will not be reported in the output, yielding a more precise blur annotation but with some edge areas of tissue being potentially missed.

**Output generation with external mask**

If binary masks for the slides being analyzed have already been generated with other tools, these can be used with
the flag **--enablemask**. Note that HistoBlur will search for a file with an identical name and a .png extension in the same directory as the slide.

```
Histoblur detect -f "path/to/images_to_test/*.svs" -o blur_quality_control -m blur_detection/blur_detection_histoblur_best_model.pth --enablemask
```


**Training model with external mask**

If you already have a specific slide and binary mask available that you wish to use for training, you can skip the built-in masking and just provide a mask in png format
with the flag **--enablemask**. Note that HistoBlur will search for a file with an identical name and a .png extension in the same directory as the slide.

```
HistoBlur train -f "path/to/wsi.svs" -o blur_detection -d blur_detection_histoblur -l 10.0 --enablemask
```

**Training model with more data**

By default, HistoBlur will take 15k overlapping patches for training and 4k for validation. Nevertheless, some of those will be filtered if there is too much white area, meaning that the actual
size will often be smaller.
Nevertheless, If you wish to train models at higher magnifications and more data, you can modify
the parameters accordingly.

```
HistoBlur train -f "path/to/wsi.svs" -o blur_detection -d blur_detection_histoblur -l 20.0 -t 20000 -v 8000 -e 120
```

Here, the model will be trained at 20.0X magnification and will take a training size of 20k and a validation size of 8k. Additionally, the **-e** flag indicates the number of
epochs used (100 by default, here 120).

**Visualizing training graphs**

After training has been completed, the training statistics can be visualized using tensordboard:

Inside the output directory, run the following

```
tensorboard --logdir logs/
```

**Running HistoBlur with Docker**

In a directory organized like this:
```
└── dir
    └── slides/
        ├── slide1.svs
        └── slide2.svs
    └── model/
        └── blur_detection_densenet_best_model.pth
```
You can run HistoBlur using the container by mounting the path to your input directory into docker:

```
docker run --gpus all -t -i -v /path/to/dir/:/app petroslk/histoblur:latest HistoBlur detect -f 'slides/*svs'  -m model/blur_detection_densenet_best_model.pth -o results
```

Pretrained models are also accessible from inside the docker container:

```
docker run --gpus all -t -i -v /path/to/dir/:/app petroslk/histoblur:latest HistoBlur detect -f 'slides/*svs'  -m /HistoBlur/pretrained_model/blur_detection_densenet_best_model_10.0X.pth -o results
```


