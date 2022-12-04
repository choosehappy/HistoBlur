#!/usr/bin/env python

from .dataset_creation_train import *
from .generate_output import *
from pathlib import Path
import argparse
import logging
import logging.config
import glob



    
def get_args():
    """Parsing command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='HistoBlur:  a flexible, deep learning based, Whole Slide Image blur detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(title="mode", help="Two modes: Train and Blur_Detection", dest='mode')
    
    ##### First subparser args
    Train_Parser = subparsers.add_parser("train", help="Create pytables with tiles and train a Densenet model to detect blurry regions")

    Train_Parser.add_argument('-f', '--input_wsi', help="WSI to use for training. ex: path/to/wsi.svs ", default="", type=str)

    Train_Parser.add_argument('-p', '--patchsize', help="patchsize to use for training, default 256", default=256, type=int)
    
    Train_Parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 128",
                        default=128, type=int)
    
    Train_Parser.add_argument('-o', '--outdir', help="output directory, default 'histoblur_output' ", default="histoblur_output", type=str)
    
    Train_Parser.add_argument('-d', '--dataset_name', help="name of dataset, default 'blur_detection'", default="blur_detection", type=str)
    
    Train_Parser.add_argument('-i', '--gpuid', help="id of gpu to use, default 0", default=0, type=int)
    
    Train_Parser.add_argument('-l', '--magnification_level', help="objective magnification level to use, default 10.0", default=10.0, type=float)
    
    Train_Parser.add_argument('-m', '--enablemask', help="provide external mask (assumes .png extension file with same slide name)", action="store_true")
    
    Train_Parser.add_argument('-t', '--trainsize', help="size of training set, default 10000", default=10000, type=int)
    
    Train_Parser.add_argument('-v', '--valsize', help="size of validation set, default 2000", default=2000, type=int)
    
    Train_Parser.add_argument('-e', '--epochs', help="number of epochs to train for, default 100", default=100, type=int)

    Train_Parser.add_argument('-y', '--training', help="pre-generated pytables file with training set", default="", type=str)

    Train_Parser.add_argument('-z', '--validation', help="pre-generated pytables file with training set", default="", type=str)

    ##### Second subparser

    Detect_parser = subparsers.add_parser("detect", help="Detect blurry regions on WSI(s)")
    
    Detect_parser.add_argument('-f', '--input_wsi', help="input WSI file(s) ex: 'path/to/wsi/*svs' ", required=True, default="", type=str)

    Detect_parser.add_argument('-o', '--outdir', help="output directory, default 'histoblur_output' ", default="histoblur_output", type=str)

    Detect_parser.add_argument('-m', '--model', help="model generated by HistoBlur train to use for blur detection ex: path/to/model.pth", required=True, default="", type=str)

    Detect_parser.add_argument('-i', '--gpuid', help="id of gpu to use, default 0", default=0, type=int)

    Detect_parser.add_argument('-t', '--enablemask', help="provide external mask (assumes .png extension file with same slide name)", action="store_true")

    Detect_parser.add_argument('-w', '--white_ratio', help="the ratio of white area to allow in each patch", default=0.9, type=float)



    args = parser.parse_args()
    
    if args.mode == "train" and args.input_wsi == "" and (args.training == "" or args.validation == ""):
        print(args.input_wsi)
        parser.error("No WSI provided and no validation/training set pytables provided.")

    if args.mode == "train":
        return Args_train(args.mode, args.input_wsi, args.patchsize, args.batchsize, args.outdir, args.gpuid, args.magnification_level, args.dataset_name,
        args.enablemask, args.trainsize, args.valsize, args.epochs, args.training, args.validation)

    elif args.mode == "detect":
        return Args_Detect(args.mode, args.input_wsi, args.outdir, args.model, args.gpuid,
         args.enablemask, args.white_ratio)
    else:
        parser.error("Mode not provided - please select between train and detect")





def main() -> None:
    """the Juicy part"""

    ########## Parse args
    args = get_args()

    #create outdir if it does not exist

    Path(f"{args.outdir}/").mkdir(parents=True, exist_ok=True)

    #Instantiate simple logger

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname) - 8s %(message)s',
     datefmt='%a, %d %b %Y %H:%M:%S', filename=f'{args.outdir}/histoblur.log', filemode='w')
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)


    
    files = glob.glob(args.input_wsi) # WSI files to use
    if len(files) == 0:
        logger.error("ERROR: No files matching the glob pattern")
        sys.exit(1)


    logger.info(f"{len(files)} files matched glob pattern")

    phases = ["train","val"] #how many phases did we create databases for?
    validation_phases= ["val"]
    
    ############## TRAINING 
        
    if args.mode == "train" and args.input_wsi == "" and args.training != "" and args.validation != "":
        logger.info("Training mode")



        print("Skipping dataset creation, using provided pytables files for training")

        dataset_train_val = [["train", args.training], ["val", args.validation]]
        model_path = train_model(path_to_pytables_list=dataset_train_val, dataname=args.dataset_name, gpuid=args.gpuid, batch_size=args.batchsize, 
        patch_size=args.patchsize,phases=phases, num_epochs=args.epochs, output_dir=args.outdir, validation_phases=validation_phases)

        logger.info(f"Training complete, model can be found at '{model_path}' and can be used to detect blurry regions")
        
    elif args.mode == "train" and args.input_wsi != "":
        logger.info("Training mode")

        Path(f"{args.outdir}/tissue_masks").mkdir(parents=True, exist_ok=True)
        print("WSI was provided, creating dataset for training and validation")
        
        dataset_train_val = create_pytables(files=files, phases=phases, dataname=args.dataset_name,patch_size=args.patchsize, trainsize=args.trainsize,
        valsize=args.valsize, magnification_level=args.magnification, output_dir=args.outdir, mask_bool=args.enablemask)
        print("Dataset preparation complete")
        print("Beginning model training")
        model_path = train_model(path_to_pytables_list=dataset_train_val, dataname=args.dataset_name, gpuid=args.gpuid, batch_size=args.batchsize,
         patch_size=args.patchsize, phases=phases, num_epochs=args.epochs, output_dir=args.outdir, validation_phases=validation_phases,magnification_level=args.magnification)
        logger.info(f"Training complete, model can be found at '{model_path}' and can be used to detect blurry regions")

    ############## GENERATE OUTPUT

    if args.mode == "detect":
        logger.info("Detect mode")
        Path(f"{args.outdir}/tissue_masks").mkdir(parents=True, exist_ok=True)
        results_df = generate_output(images=files, gpuid=args.gpuid, model=args.model, outdir=args.outdir, enablemask=args.enablemask, perc_white=args.white_ratio)
        results_df.to_csv(f"{args.outdir}/results_overview.csv", sep=",")
        print("Analysis complete")

if __name__ == "__main__":
    
    main()
