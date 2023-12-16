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

def load_model(checkpoint):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    if args.color:
        model = ColorNet(model)
    for param in model.parameters():
        param.requires_grad = False

    return model

def main(args, model):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 5,
                      'filter_hot_events': args.filter_hot_events,
                      'voxel_method': {'method': args.voxel_method,
                                       'k': args.k,
                                       't': args.t,
                                       'sliding_window_w': args.sliding_window_w,
                                       'sliding_window_t': args.sliding_window_t}
                      }
    if args.update:
        print("Updated style model")
        dataset_kwargs['combined_voxel_channels'] = False

    if args.legacy_norm:
        print('Using legacy voxel normalization')
        dataset_kwargs['transforms'] = {'LegacyNorm': {}}

    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)

    height, width = get_height_width(data_loader)

    model_info['input_shape'] = height, width
    crop = CropParameters(width, height, model.num_encoders)

    ts_fname = setup_output_folder(args.output_folder)
    
    model.reset_states()
    for i, item in enumerate(tqdm(data_loader, desc="[infer]")):
        voxel = item['events'].to(device)
        if not args.color:
            voxel = crop.pad(voxel)
        with CudaTimer('Inference'):
            output = model(voxel)
        # save sample images, or do something with output here
        if args.is_flow:
            flow_t = torch.squeeze(crop.crop(output['flow']))
            # Convert displacement to flow
            if item['dt'] == 0:
                flow = flow_t.cpu().numpy()
            else:
                flow = flow_t.cpu().numpy() / item['dt'].numpy()
            ts = item['timestamp'].cpu().numpy()
            flow_dict = flow
            fname = 'flow_{:010d}.npy'.format(i)
            np.save(os.path.join(args.output_folder, fname), flow_dict)
            with open(os.path.join(args.output_folder, fname), "a") as myfile:
                myfile.write("\n")
                myfile.write("timestamp: {:.10f}".format(ts[0]))
            flow_img = flow2bgr_np(flow[0, :, :], flow[1, :, :])
            fname = 'flow_{:010d}.png'.format(i)
            cv2.imwrite(os.path.join(args.output_folder, fname), flow_img)
        else:
            if args.color:
                image = output['image']
            else:
                image = crop.crop(output['image'])
                image = torch2cv2(image)
            fname = 'frame_{:010d}.png'.format(i)
            cv2.imwrite(join(args.output_folder, fname), image)
        append_timestamp(ts_fname, fname, item['timestamp'].item())