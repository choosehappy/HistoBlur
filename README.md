##HistoBlur

##Warning:
This tool is still under development and has only been tested on Linux Ubuntu


#Introduction

HistoBlur is a deep learning based tool that allows for the fast and accurate detection of blurry regions in Whole Slide Images.

HistoBlur has two modes, a training and a detection mode. 


#Training

In order to detect blurry regions, a DenseNet model needs to be trained for the specific stain being analyzed.
For that, the user needs to provide one clear non-blurry Whole Slide Image (WSI) for the staining of choice.
In training mode, patches from the clear non-blurry WSI will be extracted and gaussian blur will be applied to some of them. A dataset will be created
in which a patch can either be clear, mildly blurry or highly blurry. The associated patches and labels with the artificially applied levels of blurriness
are used as a training set for the model.
The model then trains for the indicated number of epochs and the saved model weights for the best performing model are outputed in the output directory.
These trained model weights can then be used to detect blurry areas in other WSI of the same straining type.

#Detection

Once the model has been trained, the user can then provide other file(s) along with the previous trained model weights in order to estimate bluriness.
The detection mode will output a tissue mask with non blurry regions appearing in green, mildly blurry regions in blue and highly blurry regions in red.
The tool will also output a csv formatted file with the percentage of mildly and highly blurry regions on the slide.

#Installation

To install HistoBlur, first clone the directory:

```
git clone https://github.com/choosehappy/HistoBlur.git
```

If you wish to install the tool in a separate conda environment:

```
conda create --name HistoBlur python=3.8
conda activate HistoBlur
```

Then, run the following:

```
pip install .
```

If you do not wish to install the tool in a separate conda environment, you can directly run this in the cloned github repo:

```
pip install .
```

To ensure that HistoBlur has been succesfully installed, run the following:

```
HistoBlur --help
```

#Examples

Since the tool has been tested at a specific magnification and patch size, if the WSI at your disposal are 40x magnification, we recomend using
default paramenters.

To train a model, all you need is a Whole Slide Image with no blur:
```
HistoBlur train -f path/to/wsi.svs -o blur_detection -d blur_detection_histoblur
```

The following command will create a training and testing set from the provided WSI that will be saved in the 'blur_detection' output folder.
The two datasets will be used to train a Densenet for 80 epochs, the model will also be outputed in the designated output folder.

Once the model has been trained, new output can be generated:

```
Histoblur detect -f path/to/images_to_test/*.svs -o blur_quality_control -m blur_detection/blur_detection_histoblur_best_model.pth
```

This command will analyze all the WSIs matching the provided glob pattern and output a csv file with the percentage of bluriness for each
in a csv file in the designated output folder. A mask of the blurry regions for each slide will also be outputted.





