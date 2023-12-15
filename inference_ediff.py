import argparse
import torch
import numpy as np
from os.path import join
import os
import cv2
from tqdm import tqdm

from utils.util import ensure_dir, flow2bgr_np
from utils.training_utils import make_recon_video
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
# pathlib.PosixPath = pathlib.WindowsPath
from torch.utils.tensorboard import SummaryWriter

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    try:
        # model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
        model_info['num_bins'] = 5
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
    dataset_kwargs = {'transforms': {"RandomCrop": {
                            "size": 112
                        },},
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
    writer = SummaryWriter(args.log_path)

    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)

    height, width = get_height_width(data_loader)

    model_info['input_shape'] = height, width
    # crop = CropParameters(width, height, model.num_encoders)
    crop = CropParameters(width, height, 1)

    ts_fname = setup_output_folder(args.output_folder)
    
    model.reset_states()
    imgs, pred_imgs = [], []
    for i, item in enumerate(tqdm(data_loader, desc="[infer.]")):
        voxel = item['events'].to(device)
        image = item['frame'].to(device)
        # writer.add_images('infer Images. GT', image, i)
        imgs.append(image)
        if not args.color:
            voxel = crop.pad(voxel)
        with CudaTimer('Inference'):
            img_XT = torch.randn_like(image).to(device)
            output = model.sample(img_XT, voxel)
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
            # if args.color:
            #     image = output['image']
            # else:
            #! modified
            pred_img = output['image']
            pred_imgs.append(pred_img)
            writer.add_images('infer Images. Sample', pred_img, i)

            pred_img = crop.crop(output['image'])
            pred_img = torch2cv2(pred_img)
            fname = 'frame_{:010d}.png'.format(i)
            cv2.imwrite(join(args.output_folder, fname), pred_img)
        append_timestamp(ts_fname, fname, item['timestamp'].item())
        if (i==10):
            break
    pred_video_tensor, gt_video_tensor = make_recon_video(pred_imgs, imgs)
    # 将视频数据添加到 TensorBoard
    # 注意：TensorBoard期望视频数据的形状为 (B, C, T, H, W)，其中 T 是帧数  
    # 预测视频
    writer.add_video('Predicted Video', pred_video_tensor,global_step=0,fps=20)
    # 真实视频
    writer.add_video('Ground Truth Video', gt_video_tensor,global_step=0,fps=20)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--log_path', required=True, type=str,
                        help='path to log')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--is_flow', action='store_true',
                        help='If true, save output to flow npy file')
    parser.add_argument('--update', action='store_true',
                        help='Set this if using updated models')
    parser.add_argument('--color', action='store_true', default=False,
                      help='Perform color reconstruction')
    parser.add_argument('--voxel_method', default='between_frames', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--filter_hot_events', action='store_true',
                        help='If true, auto-detect and remove hot pixels')
    parser.add_argument('--legacy_norm', action='store_true', default=False,
                        help='Normalize nonzero entries in voxel to have mean=0, std=1 according to Rebecq20PAMI and Scheerlinck20WACV.'
                        'If --e2vid or --firenet_legacy are set, --legacy_norm will be set to True (default False).')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI')
    parser.add_argument('--firenet_legacy', action='store_true', default=False,
                        help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')

    args = parser.parse_args()
    
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    # args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint)
    main(args, model)

# python inference_ediff.py --checkpoint_path pretrained/demo_best_231214.pth --log_path /root/autodl-tmp/save_eventcnn/infer_log --device 0 --events_file_path /root/autodl-tmp/data_eventcnn/ECD_H5/val/shapes_6dof.h5 --output_folder /root/autodl-tmp/save_eventcnn/infer