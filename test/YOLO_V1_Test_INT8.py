import torch
from torch.utils.data import DataLoader
import time
import cv2
import numpy as np
from torchvision import transforms
from YOLO_V1_DataSet import YoloV1DataSet

# import matplotlib.pyplot as plt

from nms import generate_q_sigmoid, sigmoid_lut, post_process, NMS_max, torch_post_process, torch_NMS_max
from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div

import importlib
mod = importlib.import_module("yolov1_bn_model_noaffine")

import sys
sys.path.append("../../../") # go to the directory of ai8x
import ai8x
# from batchnormfuser import bn_fuser
# import distiller.apputils as apputils

ai8x.set_device(85, simulate=True, round_avg=False, verbose=True)


# dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                         annotations_dir="../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                         ClassesFile="../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class.data",
#                         )

Yolo = mod.Yolov1_net(num_classes=5, bias=True)

qat_policy = {'start_epoch':150,
              'weight_bits':8,
              'bias_bits':8,
              'shift_quantile': 0.99}

ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)

checkpoint = torch.load('Yolov1_checkpoint-q.pth.tar')
Yolo.load_state_dict(checkpoint['state_dict'])


# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000012.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000138.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000050.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000089.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000540.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000860.jpg"
# test_dir = "log/QAT-20210924-175040/1bb_image/61.jpg" 
# test_dir = "log/QAT-20210924-175040/1bb_image/19.jpg"
test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Test/JPEGImages/000010.jpg" 

# test_dir = "car1.jpeg"

img_data = cv2.imread(test_dir)
# print(img_data.shape)
print("Origanal Image: {}".format(img_data))

transfrom = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((224, 224)),
       transforms.ToTensor(),  # hui zi dong bian huan tong dao
       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.act_mode_8bit = True
normalizer = ai8x.normalize(args)
train_data = cv2.resize(img_data, (224, 224), interpolation=cv2.INTER_AREA)

print("Resize Image: {}".format(train_data))
train_data = transfrom(train_data) # .float()

print("Float Image: {}".format(train_data.permute(2,1,0)))
print(train_data.shape)

train_data = normalizer(train_data)
# train_data = torch.unsqueeze(train_data, 0)
 
print("Int Image: {}".format(train_data.permute(2,1,0)))
print("Max: {}, Min: {}".format(train_data.max(), train_data.min()))
print(train_data.shape)

data_cvt = cv2.cvtColor(train_data.permute(1,2,0).numpy() + 128, cv2.COLOR_BGR2RGB)
print("CVT Image: {}".format(data_cvt))
 
train_data = torch.unsqueeze(train_data, dim=0)
 
# pixel_value = "121110" # 10 # -1 FF # FF => -1: int(FF, 16) = 255, 255 - 256 = -1
# 
# r_value = int(pixel_value[0:2], 16)
# g_value = int(pixel_value[2:4], 16)
# b_value = int(pixel_value[4:6], 16)
# 
# r_value_dec = r_value if r_value < 128 else r_value - 256
# g_value_dec = g_value if g_value < 128 else g_value - 256
# b_value_dec = b_value if b_value < 128 else b_value - 256
# 
# # print(b_value_dec, g_value_dec, r_value_dec)
# rgb_array = torch.cat((r_value_dec*torch.ones((1, 224,224)),
#                        g_value_dec*torch.ones((1, 224,224)),
#                        b_value_dec*torch.ones((1, 224,224))), dim=0).unsqueeze(dim=0)
# 
# # print(rgb_array.shape)
# 
# train_data = rgb_array

with torch.no_grad():
    t0 = time.time()
    bounding_boxes, fl_y = Yolo(train_data)
    t1 = time.time()
print("Inference time: {}".format(t1 - t0))

print(fl_y)
# print(fl_y.permute(0,3,2,1).reshape(1, 7*7*15))
# print(fl_y.permute(0,2,3,1).reshape(1, 7*7*15))
# print(fl_y.shape)

# hegsns

fl_y = fl_y * 64 - 32
feature_map = fl_y.permute(0, 2, 3, 1).detach().reshape(-1).numpy().astype(np.int)

# print("Final Layer output:", fl_y.permute(0, 2, 3, 1)[0,0,0,:].detach().numpy())
# print(feature_map[0:15])

# print(fl_y.permute(0, 2, 3, 1))

print("Model Output:", feature_map)

q_sigmoid, l, h, resolution = generate_q_sigmoid()
x = sigmoid_lut(feature_map, q_sigmoid, l, h, resolution)
# print(x)

# torch_softmax = torch_post_process(x)
# # print(torch_softmax.numpy().tolist()[0][0][0])
# boxes1, pred_boxes1 = torch_NMS_max(torch_softmax, img_size=224, classes=5, confidence_threshold=0.1)
# print('torch', boxes1)  # [0, 20, 24, 54, 0.82427978515625, 0.0579104907810688, 19]

softmax_x = post_process(x)
# print(len(softmax_x))

boxes2, pred_boxes2 = NMS_max(softmax_x, img_size=224, classes=5, confidence_threshold=q17p14(0.4))
# print('approx', boxes2)
print(boxes2)

IndexToClassName = {}
with open("../YOLO_V1_GPU/VOC2007/Train/VOC_remain_class_V2.data","r") as f:
    index = 0
    for line in f:
        IndexToClassName[index] = line
        index = index + 1

img_data_copy = cv2.imread(test_dir)
# img_data = cv2.resize(img_data,(448,448),interpolation=cv2.INTER_AREA)
img_data_resize = cv2.resize(img_data_copy, (224,224), interpolation=cv2.INTER_AREA)

print(boxes2)
for box in boxes2:
    print("HERE", box[0],box[1],box[2],box[3],box[4])
    confidence = box[5]
    class_index = box[6]
    box = np.array(box[0:4]).astype(np.int)
    # img_data = cv2.rectangle(img_data, (box[0] * 2,box[1] * 2),(box[2] * 2,box[3] * 2),(0,255,0),1)
    # img_data = cv2.putText(img_data, "class:{} confidence:{}".format(IndexToClassName[box[5]],box[4]),(box[0] * 2,box[1] * 2),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
    img_data_resize = cv2.rectangle(img_data_resize, (box[0],box[1]), (box[2],box[3]), (0,255,0), thickness=1)
    img_data_resize = cv2.putText(img_data_resize, "class:{} ".format(IndexToClassName[class_index]),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),thickness=1)
    cv2.imwrite(test_dir[0:-4] + "_pred.jpg", img_data_resize*255)

# print(img_data_resize.shape)
# # img_data = np.transpose(img_data, (2, 1, 0))
# plt.figure()
# plt.imshow(img_data_resize)
# plt.show()

