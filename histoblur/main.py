#!/usr/bin/env python

from .dataset_creation_train import *
from .generate_output import *
from pathlib import Path
import argparse
import time
import datetime


    
def get_args():
    """Parsing command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='HistoBlur:  a flexible, deep learning based, Whole Slide Image blur detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(title="mode", help="Two modes: Train and Blur_Detection", dest='mode')
    
    ##### First subparser args
    Train_Parser = subparsers.add_parser("train", help="Create pytables with tiles and train a Densenet model to detect blurry regions")

    Train_Parser.add_argument('-f', '--input_wsi', help="WSI to use for training", default="", type=str)

    Train_Parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
    
    Train_Parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 128",
                        default=128, type=int)
    
    Train_Parser.add_argument('-o', '--outdir', help="output directory, default '.' ", default=".", type=str)
    
    Train_Parser.add_argument('-d', '--dataset_name', help="name of dataset, default 'blur_detection'", default="blur_detection", type=str)
    
    Train_Parser.add_argument('-i', '--gpuid', help="id of gpu to use, default 0", default=0, type=int)
    
    Train_Parser.add_argument('-l', '--level', help="openslide level to use, default 1", default=1, type=int)
    
    Train_Parser.add_argument('-m', '--enablemask', help="calculate tissue mask", action="store_true")
    
    Train_Parser.add_argument('-t', '--trainsize', help="size of training set, default 10000", default=10000, type=int)
    
    Train_Parser.add_argument('-v', '--valsize', help="size of validation set, default 2000", default=2000, type=int)
    
    Train_Parser.add_argument('-e', '--epochs', help="size of training set, default 80", default=80, type=int)

    Train_Parser.add_argument('-y', '--training', help="pre-generated pytables file with training set", default="", type=str)

    Train_Parser.add_argument('-z', '--validation', help="pre-generated pytables file with training set", default="", type=str)

    ##### Second subparser

    Detect_parser = subparsers.add_parser("detect", help="Detect blurry regions on WSI(s)")
    
    Detect_parser.add_argument('-f', '--input_wsi', help="directory to WSI image file(s)", required=True, default="", type=str)

    Detect_parser.add_argument('-p', '--patchsize', help="patchsize, default 16", default=16, type=int)

    Detect_parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage ,default 128", default=128, type=int)

    Detect_parser.add_argument('-o', '--outdir', help="output directory, default '.' ", default=".", type=str)

    Detect_parser.add_argument('-m', '--model', help="model", required=True, default="", type=str)

    Detect_parser.add_argument('-i', '--gpuid', help="id of gpu to use, default 0", default=0, type=int)

    Detect_parser.add_argument('-t', '--enablemask', action="store_true")

    Detect_parser.add_argument('-k', '--mask_level',help="level of input mask", default=2, type=int)

    Detect_parser.add_argument('-l', '--extraction_level',help="openslide level at which to extract patches (should match the one used in training)", default=1, type=int)


    args = parser.parse_args()
    
    if args.mode == "train" and args.input_wsi == "" and (args.training == "" or args.validation == ""):
        print(args.input_wsi)
        parser.error("No WSI provided and no validation/training set pytables provided.")

    if args.mode == "train":
        return Args_train(args.mode, args.input_wsi, args.patchsize, args.batchsize, args.outdir, args.gpuid, args.level, args.dataset_name,
        args.enablemask, args.trainsize, args.valsize, args.epochs, args.training, args.validation)

    elif args.mode == "detect":
        return Args_Detect(args.mode, args.input_wsi, args.patchsize, args.batchsize, args.outdir, args.model, args.gpuid,
         args.enablemask, args.mask_level,args.extraction_level)
    else:
        parser.error("Mode not provided - please select between train and detect")




        


    

def main() -> None:
    """the Juicy part"""
    
    ########## Parse args
    args = get_args()


    files = glob.glob(args.input_wsi) # WSI files to use

    phases = ["train","val"] #how many phases did we create databases for?
    validation_phases= ["val"]

    #create outdir if it does not exist

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ############## TRAINING 
        
    if args.mode == "train" and args.input_wsi == "" and args.training != "" and args.validation != "":
        print("Skipping dataset creation, using provided pytables files for training")
        dataset_train_val = [["train", args.training], ["val", args.validation]]
        model_path = train_model(path_to_pytables_list=dataset_train_val, dataname=args.dataset_name, gpuid=args.gpuid, batch_size=args.batchsize, 
        patch_size=args.patchsize,phases=phases, num_epochs=args.epochs, output_dir=args.outdir, validation_phases=validation_phases)
        print(f"Training complete, model can be found at '{model_path}' and can be used to detect blurry regions")
    elif args.mode == "train" and args.input_wsi != "":
        print("WSI was provided, creating dataset for training and validation")
        dataset_train_val = create_pytables(files=files, phases=phases, dataname=args.dataset_name,patch_size=args.patchsize, trainsize=args.trainsize,
        valsize=args.valsize, sample_level=args.level, output_dir=args.outdir, mask_bool=args.enablemask)
        print("Dataset preparation complete")
        print("Beginning model training")
        model_path = train_model(path_to_pytables_list=dataset_train_val, dataname=args.dataset_name, gpuid=args.gpuid, batch_size=args.batchsize,
         patch_size=args.patchsize, phases=phases, num_epochs=args.epochs, output_dir=args.outdir, validation_phases=validation_phases,sample_level=args.level)
        print(f"Training complete, model can be found at '{model_path}' and can be used to detect blurry regions")

    ############## GENERATE OUTPUT

    if args.mode == "detect":
        results_df = generate_output(images=files, gpuid=args.gpuid, model=args.model, outdir=args.outdir, enablemask=args.enablemask, mask_level=args.mask_level,
        batch_size=args.batchsize, patch_size=args.patchsize)
        results_df.to_csv(f"{args.outdir}/results_overview.csv", sep=",")
        print("Analysis complete")

if __name__ == "__main__":
    
    main()
