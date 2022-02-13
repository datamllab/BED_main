import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
from torchvision import transforms
from YOLO_V1_DataSet import YoloV1DataSet
from sklearn.metrics.pairwise import euclidean_distances
# import matplotlib.pyplot as plt

from nms import generate_q_sigmoid, sigmoid_lut, post_process, NMS_max, torch_post_process, torch_NMS_max
from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div

import importlib
mod = importlib.import_module("yolov1_bn_model_noaffine")

import sys, os
sys.path.append("../../../") # go to the directory of ai8x
import ai8x
# from batchnormfuser import bn_fuser
# import distiller.apputils as apputils

ai8x.set_device(85, simulate=True, round_avg=False, verbose=True)


# from YOLO_V1_DataSet import YoloV1DataSet
# dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                             annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                             ClassesFile="../../VOC_remain_class.data",
#                             train_root="../../../../../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main/",
#                             img_per_class=None)

# from YOLO_V1_DataSet import YoloV1DataSet
# dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                         annotations_dir="../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                         ClassesFile="../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_V2.data",
#                         train_root="../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main/",
#                         img_per_class=None)
# 
dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Test/JPEGImages",
                            annotations_dir="../YOLO_V1_GPU/VOC2007/Test/Annotations",
                            ClassesFile="../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_transport.data",
                            train_root="../YOLO_V1_GPU/VOC2007/Test/ImageSets/Main/",
                            img_per_class=None)


Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)

qat_policy = {'start_epoch':150,
              'weight_bits':8,
              'bias_bits':8,
              'shift_quantile': 0.99}

ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)

checkpoint = torch.load('Yolov1_checkpoint-q.pth.tar')
Yolo.load_state_dict(checkpoint['state_dict'])
output_dir = checkpoint['epoch']
print("Output directory: {}".format(output_dir))

def iou(box1, box2): # left up x, y, right down x, y

    CrossLX = max(box1[0], box2[0])
    CrossRX = min(box1[2], box2[2])
    CrossUY = max(box1[1], box2[1])
    CrossDY = min(box1[3], box2[3])

    predict_Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    ground_Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if CrossRX < CrossLX or CrossDY < CrossUY:  # 没有交集
        return 0

    interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)

    # print(interSection, predict_Area, ground_Area)

    return interSection / (predict_Area + ground_Area - interSection)

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.act_mode_8bit = True
normalizer = ai8x.normalize(args)
pred_bb_num = 1

pred_score = np.zeros((len(dataSet),))
pred_box_buf = []

for image_index in range(len(dataSet)):

    img_data, ground_truth = dataSet.__getitem__(image_index)

    train_data = normalizer(img_data)
    train_data = torch.unsqueeze(train_data, 0)

    bounding_boxes, fl_y = Yolo(train_data)
    fl_y = fl_y * 64 - 32
    feature_map = fl_y.permute(0, 2, 3, 1).detach().reshape(-1).numpy().astype(np.int)

    q_sigmoid, l, h, resolution = generate_q_sigmoid()
    x = sigmoid_lut(feature_map, q_sigmoid, l, h, resolution)
    softmax_x = post_process(x)
    boxes2, pred_boxes2 = NMS_max(softmax_x, img_size=224, classes=5, confidence_threshold=q17p14(0.), topk=pred_bb_num)
    pred_box_buf.append(boxes2)

    # print(ground_truth.shape)
    ground_truth_class_onehot = [x[10:] for x in ground_truth.reshape(49, 15) if x[9]>0]
    ground_truth_classindex = [int(np.where(np.array(x))[0]) for x in ground_truth_class_onehot]
    # print(ground_truth_classindex)
    ground_boundingbox = ground_truth.reshape(49, 15)[:, 5:10]
    ground_truth_std = np.array([[x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in ground_boundingbox if x[4] > 0])
    ground_truth_valid_flag = np.array([True for x in ground_boundingbox if x[4] > 0])
    ground_boundingbox_num = len(ground_truth_std)

    pred_bb_score = np.zeros((pred_bb_num,))
    pred_bb_max_index = np.zeros((pred_bb_num,))

    for box_index, box in enumerate(boxes2):
        pred_class = box[6]
        pred_boundingbox = torch.tensor(box[0:4]).numpy()
        pred_boundingbox_std = [pred_boundingbox[0], pred_boundingbox[1],
                                pred_boundingbox[0] + pred_boundingbox[2],
                                pred_boundingbox[1] + pred_boundingbox[3]]

        iou_buf = np.zeros((ground_boundingbox_num,))
        for bb_box_index, gt in enumerate(ground_truth_std):
            if ground_truth_valid_flag[bb_box_index] == False:
                continue
            
            # if ground_truth_classindex[bb_box_index] != 3:
            #     continue
            
            # print(pred_boundingbox_std, gt, pred_class, ground_truth_classindex[index])
            correct_class = int(pred_class == ground_truth_classindex[bb_box_index])
            iou_buf[bb_box_index] = iou(pred_boundingbox_std, gt) * correct_class

            # if image_index == 1:
            #     print(iou(pred_boundingbox_std, gt), correct_class)

        # print(dis_buf.shape)
        pred_bb_score[box_index] = iou_buf.max()
        pred_bb_max_index[box_index] = iou_buf.argmax()
        ground_truth_valid_flag[iou_buf.argmax()] = False

    pred_score[image_index] = pred_bb_score.mean()

    # if pred_bb_score.min() > 0:
    #     print(pred_bb_max_index, ground_boundingbox_num, ground_truth_valid_flag)
        # hegsns

    if image_index % 100 == 99:
        print("Check {} images".format(image_index), end="\n")
#         break

# print(pred_score)

average_score = np.array(pred_score).mean()
print("Average_score: {}".format(average_score))

topk = 100
best_index_buf = (-pred_score).argsort()[0:topk]
print(best_index_buf, pred_score[best_index_buf])

output_dir_pred = os.path.join(output_dir, str(pred_bb_num) + "bb_pred")
if not os.path.exists(output_dir_pred):
    print(output_dir_pred)
    os.mkdir(output_dir_pred)

output_dir_image = os.path.join(output_dir, str(pred_bb_num) + "bb_image")
if not os.path.exists(output_dir_image):
    print(output_dir_image)
    os.mkdir(output_dir_image)

# best_index_buf = [0]

for rank_idx, index in enumerate(best_index_buf):
    img_data, _ = dataSet.__getitem__(index)

    img_std = (img_data.permute(1,2,0).numpy() + 1)/2.
    # print(img_data.shape, img_std.shape)
    img_cv2 = cv2.cvtColor(img_std[:, :, [2,1,0]], cv2.COLOR_RGB2BGR)

    boxes2 = pred_box_buf[index]
    # boxes2 = [[0,0,0,0,0,0,0]]
    for box in boxes2:
        # print("HERE", box[0], box[1], box[2], box[3], box[4], box[6])
        confidence = box[5]
        class_index = box[6]
        box = np.array(box[0:4]).astype(np.int)

        img_cv2 = cv2.rectangle(img_cv2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=1)
        img_cv2 = cv2.putText(img_cv2, "{} {}".format(dataSet.IntToClassName[class_index], confidence), (box[0], box[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=1)

    # plt.figure()
    # plt.imshow(img_cv2)
    cv2.imwrite(os.path.join(output_dir_pred, str(rank_idx) + ".jpg"), img_cv2*255)

    import shutil
    print(dataSet.img_path[index])
    shutil.copyfile(dataSet.img_path[index], os.path.join(output_dir_image, str(rank_idx) + ".jpg"))
    # plt.show()


