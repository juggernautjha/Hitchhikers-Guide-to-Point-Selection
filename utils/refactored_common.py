import numpy as np
import librosa
import pydub
from pydub import AudioSegment
from pydub import effects
from typing import List, Tuple, Dict
import random
import glob
import os
import soundfile as sf
from scipy import signal
from tqdm.notebook import tqdm
import yaml
import shutil

def yaml_load(filename = "baseline.yaml"):
    with open(filename) as stream:
        param = yaml.safe_load(stream)
    return param

def split_dataset(base_dir: str, out_dir: str, train_ratio: float, val_ratio: float, IGNORE: List[str] = []):
    """Splits the base_dir into the correct format
    """
    classes = []
    for c in os.listdir(base_dir):
        if os.path.isdir(f"{base_dir}/{c}"):
            classes.append(c)
    print(classes)
    paths = {
        cls: glob.glob(f"{base_dir}/{cls}/*.*") for cls in classes
    }
    train_paths = []
    validation_paths = []
    test_paths = []
    holdout_paths = []

    for cls in classes:
        if cls in IGNORE:
            continue
        cls_paths = paths[cls]
        total_paths = len(cls_paths)

        train_len = int(total_paths * train_ratio)
        validation_len = int(total_paths * (train_ratio + val_ratio))

        # print(train_len)
        # print(validation_len)
        # print(total_paths)

        train_paths = cls_paths[:train_len]
        validation_paths = cls_paths[train_len:validation_len]
        test_paths = cls_paths[validation_len:]

        # print(train_paths)
        # print(validation_paths)
        # print(holdout_paths)


        # print(len(tr))

        os.makedirs(f"{out_dir}/train/{cls}")
        os.makedirs(f"{out_dir}/test/{cls}")
        os.makedirs(f"{out_dir}/val/{cls}")

        for path in train_paths:
            filename = os.path.basename(path)
            shutil.copy(path, f"{out_dir}/train/{cls}/{filename}")

        for path in validation_paths:
            filename = os.path.basename(path)
            shutil.copy(path, f"{out_dir}/val/{cls}/{filename}")

        for path in test_paths:
            filename = os.path.basename(path)
            shutil.copy(path, f"{out_dir}/test/{cls}/{filename}")

    return train_paths, validation_paths, test_paths, holdout_paths        
        
        



    
    