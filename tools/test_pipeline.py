# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.evaluate import accuracy
from core.inference import get_final_preds
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import transform_preds


import dataset
import models

import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    # model.load_state_dict(torch.load(os.path.join('../output/coco/pose_resnet/res50_256x192_d256x3_adam_lr1e-3','model_best.pth')), strict=False)
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))

        # for unseen test
        pretrained_state_dict = torch.load(cfg.TEST.MODEL_FILE)
        existing_state_dict = {}
        for name, m in pretrained_state_dict.items():
            if True:
                existing_state_dict[name] = m
                print("load layer param:{}".format(name))
        model.load_state_dict(existing_state_dict, strict=False)   

        # # # for normal test
        # model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # # define loss function (criterion) and optimizer
    # criterion = JointsMSELoss(
    #     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    # ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    model.eval()
    mask_boxes_lists = {}
    use_cls = cfg.USE_CLS
    use_det_reg = cfg.USE_DET_REG
    extra_point = 0
    if use_det_reg:
        extra_point = 2
        
    start = time.time()
    for process_idx in range(1,4):
        
        process_name = "preds_" + str(process_idx)
        # print("--------------" + process_name)
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            mask_boxes = mask_boxes_lists if process_idx != 1  else None
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True
        )
        idx = 0

        # num_samples = len(valid_dataset)
        # all_preds = np.zeros(
        #     (num_samples, cfg.MODEL.NUM_JOINTS+extra_point, 3),
        #     dtype=np.float32
        # )
        # new_targets = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS+extra_point, 2),
        #                 dtype=np.float32)
        # all_boxes = np.zeros((num_samples, 6))


        with torch.no_grad():
            for i, (input, target, target_weight, meta) in enumerate(valid_loader):

                # compute output
                coords_tensor = meta['joints'][:,:,:].cuda(non_blocking=True)

                if use_cls:
                    outputs,predict_cls = model(input)
                    final_predict_cls = torch.argmax(predict_cls,dim=1).clone().cpu().numpy()
                elif use_det_reg:
                    outputs,predict_bbox = model(input)
                else:
                    outputs = model(input)

                num_images = input.size(0)
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # score = meta['score'].numpy()

                preds, maxvals = get_final_preds(cfg, outputs.clone().cpu().numpy(), c, s)
                    #coords_tensor都需要映射回去
                # coords_tensor = coords_tensor.clone().cpu().numpy()

                # for i_ in range(coords_tensor.shape[0]):
                #     coords_tensor[i_] = transform_preds(coords_tensor[i_], c[i_], s[i_],cfg.MODEL.IMAGE_SIZE)
                if use_det_reg:
                    predict_bbox = predict_bbox.unsqueeze(1).clone().cpu().numpy()*cfg.MODEL.IMAGE_SIZE[0]
                    # print('pred',predict_bbox)
                    for i_ in range(coords_tensor.shape[0]):
                        predict_bbox[i_,:,:2] = transform_preds(predict_bbox[i_,:,:2], c[i_], s[i_],cfg.MODEL.IMAGE_SIZE) 
                        predict_bbox[i_,:,2:] = transform_preds(predict_bbox[i_,:,2:], c[i_], s[i_],cfg.MODEL.IMAGE_SIZE) 

                # print("coords_tensor:",coords_tensor)
                if use_det_reg:
                    preds = np.concatenate([predict_bbox[:,:,:2],predict_bbox[:,:,2:],preds],axis=1)
                    bbox_score = np.ones((num_images,2,1))
                    maxvals = np.concatenate((bbox_score,maxvals),axis=1)
                for n in range(num_images):
                    image_id = meta['image'][n]
                    if process_idx == 1:
                        if use_cls:
                            mask_boxes_lists[image_id] = {process_name:[np.concatenate((preds[n,:,:], maxvals[n]), axis=1),final_predict_cls[n]]}
                        else:
                            mask_boxes_lists[image_id] = {process_name:[np.concatenate((preds[n,:,:], maxvals[n]), axis=1)]}
                    else:
                        if use_cls:
                            mask_boxes_lists[image_id][process_name] = [np.concatenate((preds[n,:,:], maxvals[n]), axis=1),final_predict_cls[n]]
                        else:
                            mask_boxes_lists[image_id][process_name] = [np.concatenate((preds[n,:,:], maxvals[n]), axis=1)]

                # new_targets[idx:idx + num_images, :,:] = coords_tensor[:,:,:2]
                # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                # all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # # double check this all_boxes parts
                # all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                # all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                # all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                # all_boxes[idx:idx + num_images, 5] = score

                idx += num_images
    end = time.time()
    spend_time = end-start
    aver_time = spend_time/1029
    FPS = 1/aver_time
    print('spend_time:',spend_time)
    print('aver_time:',aver_time)
    print('FPS:',FPS)
    np.save(os.path.join(final_output_dir,'second_process.npy'),[mask_boxes_lists])


if __name__ == '__main__':
    main()


