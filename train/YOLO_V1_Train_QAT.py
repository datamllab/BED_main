import importlib

import os
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
#import distiller.apputils as apputils
import cv2

import sys
#sys.path.insert(0, 'yolo/')
#sys.path.insert(1, 'distiller/')
#sys.path.insert(2, '/data/detection/')

from YOLO_V1_DataSet import YoloV1DataSet
from YOLO_V1_LossFunction import  Yolov1_Loss

mod = importlib.import_module("yolov1_bn_model_noaffine")

import ai8x
#%matplotlib inline

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=400, help='Maximum training epoch.')
parser.add_argument('--qat_strt_epoch', type=int, default=150, help='Maximum training epoch.')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=16, help='Minibatch size.')
parser.add_argument('--img_train', type=str, default="100", help='Image number per class for training.')
parser.add_argument('--gpu', type=int, default=0, help='Use which gpu to train the model.')
parser.add_argument('--exp', type=str, default="QAT", help='Experiment name.')
parser.add_argument('--seed', type=int, default=7, help='Random seed.')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def log_init():

    import log_utils, time, glob, logging, sys

    fdir0 = os.path.join("log", args.exp + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    log_utils.create_exp_dir(fdir0, scripts_to_save=glob.glob('*.py'))
    args.output_dir = fdir0

    logger = log_utils.get_logger(tag=(args.exp), log_level=logging.INFO)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(fdir0, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info("hyperparam = %s", args)

    return logger


# Initialize the dataset and dataloader
def dataset_init(logger):

    # dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
    #                         annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
    #                         ClassesFile="../../VOC_remain_class.data",
    #                         train_root="../../../../../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main/",
    #                         img_per_class=eval(args.img_train),
    #                         ms_logger=logger)

    dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
                            annotations_dir="../YOLO_V1_GPU/VOC2007/Train/Annotations",
                            ClassesFile="../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_V2.data",#../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_2class.data"
                            train_root = "../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main/",
                            img_per_class = eval(args.img_train),
                            ms_logger=logger)

    dataLoader = DataLoader(dataSet, batch_size=args.batch_size, shuffle=True,num_workers=4)
    return dataSet, dataLoader


# Train
def train(logger):

    dataSet, dataLoader = dataset_init(logger)

    # Set ai8x device
    ai8x.set_device(device=85, simulate=False, round_avg=False)
    Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)
    Yolo = Yolo.to(args.device)
    logger.info("NUMBER OF PARAMETERS {}".format(sum(p.numel() for p in Yolo.parameters())))

    # Initialize the loss function
    loss_function = Yolov1_Loss().to(args.device)
    optimizer = optim.SGD(Yolo.parameters(),lr=args.lr,momentum=0.9,weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000],gamma=0.8)

    # Initialize the quantization policy
    num_epochs = args.max_epoch
    qat_policy = {'start_epoch':args.qat_strt_epoch,
                  'weight_bits':8,
                  'bias_bits':8,
                  'shift_quantile': 0.99}

    # Main training
    best_acc = 0
    best_qat_acc = 0
    for epoch in range(0, num_epochs):

        loss_sum = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        epoch_iou = 0
        epoch_object_num = 0
        scheduler.step()

        if epoch > 0 and epoch == qat_policy['start_epoch']:
            logger.info('QAT is starting!')
            # Fuse the BN parameters into conv layers before Quantization Aware Training (QAT)
            torch.save(Yolo.state_dict(), os.path.join(args.output_dir, "scaled224_noaffine_shift{}_maxim_yolo_beforeQAT_ep{}.pth".format(qat_policy["shift_quantile"], epoch)))
            ai8x.fuse_bn_layers(Yolo)

            # Switch model from unquantized to quantized for QAT
            ai8x.initiate_qat(Yolo, qat_policy)

            # Model is re-transferred to GPU in case parameters were added
            Yolo.to(args.device)


        for batch_index, batch_train in enumerate(dataLoader):

            optimizer.zero_grad()
            train_data = batch_train[0].float().to(args.device)
            train_data.requires_grad = True

            label_data = batch_train[1].float().to(args.device)
            # label_data[:, :, :, :, 5] = label_data[:, :, :, :, 5] / 224
            # label_data[:, :, :, :, 6] = label_data[:, :, :, :, 6] / 224
            # label_data[:, :, :, :, 7] = label_data[:, :, :, :, 7] / 224
            # label_data[:, :, :, :, 8] = label_data[:, :, :, :, 8] / 224
            # label_data[:, :, :, :, 9] = label_data[:, :, :, :, 9] / (224*224)
            bb_pred, _ = Yolo(train_data) #Z: bb_pred shape = 7x7x15
            loss = loss_function(bounding_boxes=bb_pred,ground_truth=label_data)
            batch_loss = loss[0]
            loss_coord = loss_coord + loss[1]
            loss_confidence = loss_confidence + loss[2]
            loss_classes = loss_classes + loss[3]
            epoch_iou = epoch_iou + loss[4]
            epoch_object_num = epoch_object_num + loss[5]
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()
            loss_sum = loss_sum + batch_loss

            #logger.info("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))


        if epoch % 50 == 0:
            if epoch >= qat_policy['start_epoch']:
                torch.save(Yolo.state_dict(), os.path.join(args.output_dir, "scaled224_noaffine_shift{}_maxim_yolo_qat_ep{}.pth".format(qat_policy["shift_quantile"], epoch)))
            else:
                torch.save(Yolo.state_dict(), os.path.join(args.output_dir, "scaled224_noaffine_shift{}_maxim_yolo_ep{}.pth".format(qat_policy["shift_quantile"], epoch)))


        batch_num = len(dataLoader)
        avg_loss= loss_sum/batch_num
        logger.info("epoch : {} ; loss : {} ; avg_loss: {}".format(epoch,{loss_sum},{avg_loss}))

        epoch = epoch + 1
    torch.save(Yolo.state_dict(), os.path.join(args.output_dir, "scaled224_noaffine_shift{}_maxim_yolo_qat_ep{}.pth".format(qat_policy["shift_quantile"], epoch)))


def main():
    # Set GPU
    # setup_seed(args.seed)
    logger = log_init()

    args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    logger.info('Running on device: {}'.format(args.device))
    train(logger)


if __name__ == "__main__":
    main()


# def count_params(model):
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     params = sum([np.prod(p.size()) for p in model_parameters])
#     return params
#
# class Args:
#     def __init__(self, act_mode_8bit):
#         self.act_mode_8bit = act_mode_8bit
#         self.truncate_testset = False
