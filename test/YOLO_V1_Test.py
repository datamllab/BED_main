# 网络加载
import torch
import numpy as np
import cv2
import torch
from torchvision import transforms

import matplotlib.pyplot as plt

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

from YOLO_V1_DataSet import YoloV1DataSet
dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
                            annotations_dir="../YOLO_V1_GPU/VOC2007/Train/Annotations",
                            ClassesFile="../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_V2.data",
                            train_root="../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main/",
                            img_per_class=100)

Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)

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
# checkpoint_fname = './log/QAT-20210712-042211/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # batch_size 16
# checkpoint_fname = './log/QAT-20210714-230119/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # batch_size 16
# checkpoint_fname = './log/QAT-20210715-030212/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep500.pth' # batch_size 16
# checkpoint_fname = './log/QAT-20210715-082952/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # batch_size 16

# checkpoint_fname = './log/QAT-20210924-175040/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # class V2
checkpoint_fname = './log/QAT-20211115-173856/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # transport
# checkpoint_fname = './log/QAT-20211116-163705/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth' # animals

Yolo.load_state_dict(torch.load(checkpoint_fname, map_location=lambda storage, loc: storage)) # batch_size 16


ck_fname = checkpoint_fname.split("/")[-1]
checkpoint_dir = checkpoint_fname.replace(ck_fname, "")
print("Checkpoint path:", checkpoint_dir)
import distiller.apputils as apputils
apputils.save_checkpoint(checkpoint_dir, "ai85net5", Yolo,
                            optimizer=None, scheduler=None, extras=None,
                            is_best=False, name="Yolov1", dir="./",
                         )


# class to index
IndexToClassName = {}
with open("../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_V2.data","r") as f:
    index = 0
    for line in f:
        IndexToClassName[index] = line
        index = index + 1

def iou(box_one, box_two):
    LX = max(box_one[0], box_two[0])
    LY = max(box_one[1], box_two[1])
    RX = min(box_one[2], box_two[2])
    RY = min(box_one[3], box_two[3])
    if LX >= RX or LY >= RY:
        return 0
    return (RX - LX) * (RY - LY) / ((box_one[2]-box_one[0]) * (box_one[3] - box_one[1]) + (box_two[2]-box_two[0]) * (box_two[3] - box_two[1]))

def NMS(bounding_boxes,S=7,B=2,img_size=448,confidence_threshold=0.55,iou_threshold=0.2):
    bounding_boxes = bounding_boxes.cpu().detach().numpy().tolist()
    predict_boxes = []
    nms_boxes = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                gridX = grid_size * i
                gridY = grid_size * j
                if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                    bounding_box = bounding_boxes[batch][i][j][5:10]
                else:
                    bounding_box = bounding_boxes[batch][i][j][0:5]
                bounding_box.extend(bounding_boxes[batch][i][j][10:])
                if bounding_box[4] >= confidence_threshold:
                    predict_boxes.append(bounding_box)

                centerX = (int)(gridX + bounding_box[0] * grid_size)
                centerY = (int)(gridY + bounding_box[1] * grid_size)
                width = (int)(bounding_box[2] * img_size)
                height = (int)(bounding_box[3] * img_size)
                bounding_box[0] = max(0, (int)(centerX - width / 2))
                bounding_box[1] = max(0, (int)(centerY - height / 2))
                bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))

                # print(centerX, centerY, width, height)

        while len(predict_boxes) != 0:
            predict_boxes.sort(key=lambda box:box[4])
            assured_box = predict_boxes[0]
            temp = []
            classIndex = np.argmax(assured_box[5:])
            print("Class index:{}".format(classIndex), "Confidence:", assured_box)
            assured_box[4] = assured_box[4] * assured_box[5 + classIndex] #修正置信度为 物体分类准确度 × 含有物体的置信度
            assured_box[5] = classIndex
            nms_boxes.append(assured_box)
            i = 1
            while i < len(predict_boxes):
                if iou(assured_box,predict_boxes[i]) <= iou_threshold:
                    temp.append(predict_boxes[i])
                i = i + 1
            predict_boxes = temp

        return nms_boxes


# 读取测试数据
transfrom = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(), # hui zi dong bian huan tong dao
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])

test_dir = "../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000012.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000138.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000047.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000060.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000083.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000282.jpg"

# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000012.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000138.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000050.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000089.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000540.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000860.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000133.jpg"

img_data = cv2.imread(test_dir)
img_data_resize = cv2.resize(img_data,(224,224),interpolation=cv2.INTER_AREA)

train_data = transfrom(img_data).float()
# print(train_data.shape)
train_data = torch.unsqueeze(train_data, 0)
# print(train_data.shape)
bounding_boxes, fl_y = Yolo(train_data)
# print(bounding_boxes.shape)

class_prob = bounding_boxes[0, :, :, [4,9]]
# print(class_prob)
# print(fl_y)

NMS_boxes = NMS(bounding_boxes, img_size=224, confidence_threshold=0.5) # , confidence_threshold=1e-10,iou_threshold=0.)
for box in NMS_boxes:
    print("HERE", box[0],box[1],box[2],box[3],box[4])
    img_data_resize = cv2.rectangle(img_data_resize, (box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    img_data_resize = cv2.putText(img_data_resize, "class:{} confidence:{}".format(IndexToClassName[box[5]],box[4]),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    # break

print(img_data_resize.shape)
plt.imshow(img_data_resize)
plt.show()

# cv2.imwrite('img.png', img_data)
# cv2.waitKey()
