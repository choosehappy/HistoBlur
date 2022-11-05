import os
import shutil
import textwrap
from conftest import *
import pytest
import openslide
import numpy as np
from subprocess import getstatusoutput, getoutput
from histoblur.generate_output import generate_mask_loose
from histoblur.dataset_creation_train import generate_mask_stringent

PRG = 'HistoBlur'
MODEL_PATH = 'pretrained_model/blur_detection_densenet_best_model.pth'
ABS_PATH_MODEL = os.path.abspath(MODEL_PATH)

# --- small helpers ---
def _filenames_in(pth): return set(x.name for x in pth.glob('*'))

@pytest.fixture(scope='module')
def single_svs_dir(tmp_path_factory, svs_small):
    pth = tmp_path_factory.mktemp('histoqc_test_single')
    shutil.copy(svs_small, pth)
    yield pth

def test_mask_generation_stringent(single_svs_dir):
    osh  = openslide.OpenSlide(f"{single_svs_dir}/CMU-1-JP2K-33005.svs")
    image = osh.read_region((0, 0), 2, osh.level_dimensions[2])
    image = np.asarray(image)[:, :, 0:3]
    mask = generate_mask_stringent(image) #generate mask
    total_pixels = int(sum(sum(mask)))
    assert 717796 == total_pixels #make sure that the exact same number of pixels as expected are within mask

def test_mask_generation_loose(single_svs_dir):
    osh  = openslide.OpenSlide(f"{single_svs_dir}/CMU-1-JP2K-33005.svs")
    image = osh.read_region((0, 0), 2, osh.level_dimensions[2])
    image = np.asarray(image)[:, :, 0:3]
    mask = generate_mask_loose(image) #generate mask
    total_pixels = int(sum(sum(mask)))
    assert 1247254 == total_pixels #make sure that the exact same number of pixels as expected are within mask


def test_training(single_svs_dir, tmp_path):
    rv, out = getstatusoutput(f"{PRG} train -f '{os.fspath(single_svs_dir)}/*.svs' -t 10 -v 5 -e 1 -l 0 -s 32 -o {tmp_path}")
    assert rv == 0
    assert _filenames_in(tmp_path) == _filenames_in(tmp_path).union(["blur_detection_val.pytable", "blur_detection_train.pytable", "blur_detection_densenet_best_model.pth",
     "tissue_masks", "logs"])

def test_output_gen(single_svs_dir, tmp_path):
    rv, out = getstatusoutput(f"{PRG} detect -f '{os.fspath(single_svs_dir)}/*.svs' -m {ABS_PATH_MODEL} -o {tmp_path}")
    assert rv == 0
    assert _filenames_in(tmp_path) == _filenames_in(tmp_path).union(["output_CMU-1-JP2K-33005.tif", "results_overview.csv", "tissue_masks"])
