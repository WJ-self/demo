import argparse
import torch
import numpy as np
from os.path import join
import os
import cv2
from tqdm import tqdm

from utils.util import ensure_dir, flow2bgr_np
from model import model as model_arch
from data_loader.data_loaders import InferenceDataLoader
from model.model import ColorNet
from utils.util import CropParameters, get_height_width, torch2cv2, \
                       append_timestamp, setup_output_folder
from utils.timers import CudaTimer
from utils.henri_compatible import make_henri_compatible

from parse_config import ConfigParser

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

