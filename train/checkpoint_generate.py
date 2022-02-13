# 网络加载
import torch
import numpy as np
import torch
# from torchvision import transforms

# import matplotlib.pyplot as plt

import importlib
mod = importlib.import_module("yolov1_bn_model_noaffine")

import sys
sys.path.append("../../../") # go to the directory of ai8x
import ai8x
# from batchnormfuser import bn_fuser


ai8x.set_device(85, simulate=False, round_avg=False, verbose=True)

# from YOLO_V1_DataSet_small import YoloV1DataSet
# dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                         annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                         ClassesFile="../../VOC_remain_class.data",
#                         data_path='../../../../../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main')

# from YOLO_V1_DataSet import YoloV1DataSet
# dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                             annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                             ClassesFile="../../VOC_remain_class_V2.data",
#                             train_root="../../../../../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main/",
#                             img_per_class=100)

Yolo = mod.Yolov1_net(num_classes=5, bias=True)

qat_policy = {'start_epoch':150,
              'weight_bits':8,
              'bias_bits':8,
              'shift_quantile': 0.99}

ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)

# checkpoint_fname = './weight_20210711/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep0400.pth'        # Zaid
# checkpoint_fname = './log/QAT-20210711-064558/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # YOLO.train()
# checkpoint_fname = './yolo_models_test/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep0400.pth'       # Zaid 2
# checkpoint_fname = './log/QAT-20210712-003214/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep300.pth' # dataset v2
# checkpoint_fname = './log/QAT-20210711-203619/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep300.pth' # whole dataset
# checkpoint_fname = './log/QAT-20210712-024843/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep350.pth' # batch_size 64
# checkpoint_fname = './log/QAT-20210712-033721/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # batch_size 16 seed 7
# checkpoint_fname = './log/QAT-20210712-042211/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth'   # batch_size 16
# checkpoint_fname = './log/QAT-20210714-230119/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth'   # batch_size 16
# checkpoint_fname = './log/QAT-20210715-030212/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep500.pth'   # batch_size 16 full training set
# checkpoint_fname = './log/QAT-20210715-082952/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth'   # batch_size 16 full training set
#checkpoint_fname = './log/QAT-20210924-175040/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth'   # batch_size 16 full training set
checkpoint_fname = './log/QAT-20211116-163705/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' #5 class animal
Yolo.load_state_dict(torch.load(checkpoint_fname, map_location=lambda storage, loc: storage)) # batch_size 16


ck_fname = checkpoint_fname.split("/")[-1]
checkpoint_dir = checkpoint_fname.replace(ck_fname, "")
import distiller.apputils as apputils
apputils.save_checkpoint(checkpoint_dir, "ai85net5", Yolo,
                            optimizer=None, scheduler=None, extras=None,
                            is_best=False, name="Yolov1", dir="./",
                         )


