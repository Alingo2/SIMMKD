# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F

from utils.transforms import transform_preds

import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)
coord_loss = nn.L1Loss(reduction='none')
score_bce = torch.nn.BCELoss(reduction='none')

def calculate_l2_distance(predictions,targets):
    predictions = (predictions.astype(np.float)).reshape(-1,2)
    targets = targets.astype(np.float).reshape(-1,2)
    # targets = targets[:, :]
    # predictions = predictions[:, :] 

    dist = np.power((predictions - targets), 2)
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    dist_mean = np.mean(dist)
    dist_std = np.std(dist)
    print("dist_mean:",dist_mean,"  dist_std",dist_std) #与最终的不同是因为旋转缩放了
    return dist_mean,dist_std

def cal_match_loss(t,p):     #predict[32,9,3]
    target = t.clone().detach()
    predict = p.clone().detach()
    num_box = 3
    target = target.reshape(-1,3,3,3)   #转成[32,3,3,3]
    predict = predict.reshape(-1,3,3,3)


    predict_matrix = predict.repeat(1,3,1,1)      #[32,9,3,3]                             内部[a,b,c,a,b,c,a,b,c]
    target_matrix = (target.repeat(1,1,3,1)).view(-1,9,3,3) #[32,3,9,3] ->[32,9,3,3] 内部 [A,A,A,B,B,B,C,C,C]

    predict_coords_matrix = predict_matrix[:,:,:,:2]
    predict_score_matrix = predict_matrix[:,:,:,2]
    target_coords_matrix = target_matrix[:,:,:,:2]
    target_score_matrix = target_matrix[:,:,:,2]

    target_flag = target_score_matrix[:,:,0]    #取第一个值代表这个bbox即可

    coords_loss = torch.mean(coord_loss(predict_coords_matrix,target_coords_matrix),3) #[32,9,3] 其中9代表框的矩阵损失, 3是框内每一个点的损失
    box_loss = torch.mean(coords_loss,2)    #[32,9]9:分别是[aA,bA,cA,aB,bB,cB,aC,bC,cC]损失  每张图片应该找到
    # coord_score_loss = score_bce(predict_score_matrix,target_score_matrix)  #[32,9,3]  #每个框 每个点的置信度损失
    # box_score_loss = torch.mean(coord_score_loss,2) #[32,9] 每个框的平均score损失(由3个点的置信度加权)

    batch_idx = range(box_loss.shape[0])
    # print("box_loss",box_loss)
    # print("box_score_loss",box_score_loss)
    # print("target_flag",target_flag)
    back_coords_loss = torch.zeros(box_loss.shape[0]).cuda()
    back_score_loss = torch.zeros(box_loss.shape[0]).cuda()

    match_seq = torch.zeros(3,box_loss.shape[0])
    # 是否需要加入设置梯度回传
    for i in range(num_box):
        # print(i)
        matched_box_loss,precise_idx = torch.min(box_loss,1)

        match_seq[i,:] = precise_idx
        matched_score_loss = torch.zeros(box_loss.shape[0]).cuda()
        # precise_idx = torch.argmin(box_loss,1)
        # print("matched_box_loss",matched_box_loss.shape)
        # print("precise_idx",precise_idx.shape)

        idx = precise_idx//3
        
        # print("before",matched_box_loss)
        for j in batch_idx:
            # matched_box_loss[j] = matched_box_loss[j] * target_flag[j][precise_idx[j]]
            # matched_score_loss[j] = box_score_loss[j,precise_idx[j]]
            box_loss[j,idx[j]*3] = 100
            box_loss[j,idx[j]*3+1] = 100
            box_loss[j,idx[j]*3+2] = 100
        # print("after",matched_box_loss)
    
        # back_coords_loss = back_coords_loss + matched_box_loss
        # back_score_loss = back_score_loss + matched_score_loss
    return match_seq
    # return back_coords_loss.mean()+back_score_loss.mean()*0.5

def Diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:#
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1] 
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    
    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2 
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2 
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:]) 
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2]) 
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:]) 
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    use_regression = config.USE_REGRESSION
    use_cls = config.USE_CLS
    use_detr = config.USE_DETR
    use_det_reg = config.USE_DET_REG
    L1 = torch.nn.L1Loss()
    entropy = torch.nn.CrossEntropyLoss()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # compute output
        if use_cls:
            outputs,predict_cls = model(input)
            gt_cls = meta['speed_category'].cuda(non_blocking=True).long()            
            cls_loss = entropy(predict_cls,gt_cls)
        elif use_det_reg:
            left_upper = meta['left_upper']/config.MODEL.IMAGE_SIZE[0]
            right_lower = meta['right_lower']/config.MODEL.IMAGE_SIZE[0]
            mybbox = (torch.cat((left_upper,right_lower),1)).cuda(non_blocking=True)
            outputs,predict_bbox = model(input)
            det_loss = -1*torch.mean(Diou(predict_bbox,mybbox))            
            
        else:
            outputs = model(input)

        if use_regression:
            output = outputs[:,:,:,:2]
     

            coords_tensor = meta['joints'][:,:,:3].cuda(non_blocking=True)
            coords_tensor[:,:,0] = (coords_tensor[:,:,0]-1)/(config.MODEL.IMAGE_SIZE[0]-1)    #映射为[0,1]
            coords_tensor[:,:,1] = (coords_tensor[:,:,1]-1)/(config.MODEL.IMAGE_SIZE[1]-1)


            # print("outputs:",outputs*512)
            if use_detr:
                outputs = outputs.squeeze(1)
                match_seq = cal_match_loss(coords_tensor.to(torch.float32),outputs.to(torch.float32))

                predict_idx = match_seq % 3
                target_idx = match_seq // 3
                loss = torch.tensor(0.0, requires_grad=True)
                for box_idx in range(match_seq.shape[0]):
                    for batch_idx in range(match_seq.shape[1]):
                        matched_gt_box = coords_tensor[batch_idx,int(target_idx[box_idx,batch_idx]):int(target_idx[box_idx,batch_idx])+3,:]    #得到[3,3]
                        matched_predict_box = outputs[batch_idx,int(predict_idx[box_idx,batch_idx]):int(predict_idx[box_idx,batch_idx])+3,:]
                        temp_loss = bce(matched_predict_box[:,2].to(torch.float32),matched_gt_box[:,2].to(torch.float32))
                        if matched_gt_box[0,-1] == 1:
                            temp_loss = temp_loss + L1(matched_gt_box[:,:2].to(torch.float32),matched_predict_box[:,:2].to(torch.float32))
                        
                        loss = loss + temp_loss
                loss = loss/(match_seq.shape[0]*match_seq.shape[1])


            else:
                coords_tensor = coords_tensor[:,:,:2]                
                loss = L1(coords_tensor.to(torch.float32),output.to(torch.float32))     #*5
            # loss = L1(coords_tensor,outputs)

            # output = output.squeeze(1)    #在第一维减少一个keypoint的维度

        else:

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)
        if use_cls:
            # loss = loss*0.5
            loss = loss + cls_loss*0.5
        if use_det_reg:
            loss = loss + det_loss*0.5
            
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        # losses.update(loss.item(), input.size(0))
        losses.update(loss, input.size(0))  #多次累加 不能使用.item()
        
        if use_regression:
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                        target.detach().cpu().numpy(),'regression')
        else:
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            if not use_regression:
                save_debug_images(config, input, meta, target, pred*4, output,
                                prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    L1 = torch.nn.L1Loss()
    entropy = torch.nn.CrossEntropyLoss()
    use_cls = config.USE_CLS
    use_regression = config.USE_REGRESSION
    use_detr = config.USE_DETR
    use_det_reg = config.USE_DET_REG

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    new_targets = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 2),
                    dtype=np.float32)

    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            coords_tensor = meta['joints'][:,:,:].cuda(non_blocking=True)
            if use_cls:
                outputs,predict_cls = model(input)
                gt_cls = meta['speed_category'].cuda(non_blocking=True).long()            
                cls_loss = entropy(predict_cls,gt_cls)
                final_predict_cls = torch.argmax(predict_cls,dim=1).clone().cpu().numpy()
            elif use_det_reg:
                left_upper = meta['left_upper']/config.MODEL.IMAGE_SIZE[0]
                right_lower = meta['right_lower']/config.MODEL.IMAGE_SIZE[0]
                mybbox = (torch.cat((left_upper,right_lower),1)).cuda(non_blocking=True)
                outputs,predict_bbox = model(input)
                det_loss = -1*torch.mean(Diou(predict_bbox,mybbox))    
            else:
                outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            if use_regression:
                output = outputs[:,:,:,:2]
                maxvals = outputs[:,:,:,2].clone().squeeze(1).unsqueeze(2).cpu().numpy()

                coords_tensor[:,:,0] = (coords_tensor[:,:,0]-1)/(config.MODEL.IMAGE_SIZE[0]-1)
                coords_tensor[:,:,1] = (coords_tensor[:,:,1]-1)/(config.MODEL.IMAGE_SIZE[1]-1)
                # output = output.squeeze(1)    #在第一维减少一个keypoint的维度

                # loss = L1(target.to(torch.float32),output.to(torch.float32))
                if use_detr:
                    outputs = output.squeeze(1)
                    match_seq = cal_match_loss(coords_tensor.to(torch.float32),outputs.to(torch.float32))

                    predict_idx = match_seq % 3
                    target_idx = match_seq // 3
                    loss = torch.tensor(0.0, requires_grad=True)
                    for box_idx in range(match_seq.shape[0]):
                        for batch_idx in range(match_seq.shape[1]):
                            matched_gt_box = coords_tensor[batch_idx,int(target_idx[box_idx,batch_idx]):int(target_idx[box_idx,batch_idx])+3,:]    #得到[3,3]
                            matched_predict_box = outputs[batch_idx,int(predict_idx[box_idx,batch_idx]):int(predict_idx[box_idx,batch_idx])+3,:]
                            temp_loss = bce(matched_predict_box[:,2].to(torch.float32),matched_gt_box[:,2].to(torch.float32))
                            if matched_gt_box[0,-1] == 1:
                                temp_loss = temp_loss + L1(matched_gt_box[:,:2].to(torch.float32),matched_predict_box[:,:2].to(torch.float32))
                            
                            loss = loss + temp_loss
                    loss = loss/(match_seq.shape[0]*match_seq.shape[1])

                    maxvals = outputs[:,:,2].clone().unsqueeze(2).cpu().numpy()
                    output = output[:,:,:,:2]
                else:
                    coords_tensor = coords_tensor[:,:,:2]
                    loss = F.l1_loss(coords_tensor.to(torch.float32),output.to(torch.float32))*5
                if use_cls:
                    loss = loss
                    loss = loss + cls_loss*0.5

            else:
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                
                loss = criterion(output, target, target_weight)
                if use_cls:
                    loss = loss + cls_loss*0.5
                if use_det_reg:
                    loss = loss + det_loss*0.5

            num_images = input.size(0)
            # measure accuracy and record loss
            # losses.update(loss.item(), num_images)
            losses.update(loss, num_images)
            if use_regression:
                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy(),'regression')
            else:
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()


            if use_regression:
                preds = output.squeeze(1)
                preds = preds.clone().cpu().numpy()
                preds = preds[:,:,:2] * (config.MODEL.IMAGE_SIZE[0]-1) + 1

                coords_tensor = coords_tensor.clone().cpu().numpy()
                coords_tensor = coords_tensor* (config.MODEL.IMAGE_SIZE[0]-1) + 1
                if use_cls:
                    maxvals = predict_cls.unsqueeze(1).clone().cpu().numpy()
                # else:
                for i_ in range(coords_tensor.shape[0]):
                    preds[i_] = transform_preds(preds[i_], c[i_], s[i_],config.MODEL.IMAGE_SIZE)
            else:
                preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
                #coords_tensor都需要映射回去
                coords_tensor = coords_tensor.clone().cpu().numpy()


            for i_ in range(coords_tensor.shape[0]):
                coords_tensor[i_] = transform_preds(coords_tensor[i_], c[i_], s[i_],config.MODEL.IMAGE_SIZE)

                # print("coords_tensor:",coords_tensor)
            new_targets[idx:idx + num_images, :,:] = coords_tensor[:,:,:2]
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            # all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                if not use_regression:
                    save_debug_images(config, input, meta, target, pred*4, output,
                                    prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
        
        dist_mean,dist_std = calculate_l2_distance(all_preds[:,:,:2],new_targets)


        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
