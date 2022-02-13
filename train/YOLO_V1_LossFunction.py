import torch.nn as nn
import math
import torch
import sys

class Yolov1_Loss(nn.Module):

    def __init__(self, S=7, B=2, Classes=20, l_coord=5, l_noobj=0.5):
        # 有物体的box损失权重设为l_coord,没有物体的box损失权重设置为l_noobj
        super(Yolov1_Loss, self).__init__()
        self.S = S
        self.B = B
        self.Classes = Classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def iou(self, bounding_box, ground_box, gridX, gridY, img_size=448, grid_size=64):  # 计算两个box的IoU值
        # predict_box: [centerX, centerY, width, height]
        # ground_box : [centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)
        # 1.  预处理 predict_box  变为  左上X,Y  右下X,Y  两个边界点的坐标 避免浮点误差 先还原成整数
        # 不要共用引用
        predict_box = list([0,0,0,0])
        predict_box[0] = (int)(gridX + bounding_box[0] * grid_size)
        predict_box[1] = (int)(gridY + bounding_box[1] * grid_size)
        predict_box[2] = (int)(bounding_box[2] * img_size)
        predict_box[3] = (int)(bounding_box[3] * img_size)

        # [xmin,ymin,xmax,ymax]
        predict_coord = list([max(0, predict_box[0] - predict_box[2] / 2), max(0, predict_box[1] - predict_box[3] / 2),min(img_size - 1, predict_box[0] + predict_box[2] / 2), min(img_size - 1, predict_box[1] + predict_box[3] / 2)])
        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])

        ground_coord = list([ground_box[5],ground_box[6],ground_box[7],ground_box[8]])
        ground_Area = (ground_coord[2] - ground_coord[0]) * (ground_coord[3] - ground_coord[1])

        # 存储格式 xmin ymin xmax ymax

        # 2.计算交集的面积 左边的大者 右边的小者 上边的大者 下边的小者
        CrossLX = max(predict_coord[0], ground_coord[0])
        CrossRX = min(predict_coord[2], ground_coord[2])
        CrossUY = max(predict_coord[1], ground_coord[1])
        CrossDY = min(predict_coord[3], ground_coord[3])

        if CrossRX < CrossLX or CrossDY < CrossUY: # 没有交集
            return 0

        interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)

        if interSection > (predict_Area + ground_Area - interSection):
            print("interSection:{} predict_Area:{} ground_Area:{}".format(interSection, predict_Area, ground_Area))
            print("predictLX:{} predictLY:{} predictRX:{} predictRY:{}".format(predict_coord[0], predict_coord[1], predict_coord[2], predict_coord[3]))
            print("groundLX:{} groundLY:{} groundRX:{} groundRY:{}".format(ground_coord[0], ground_coord[1], ground_coord[2], ground_coord[3]))
            print("interLX:{} interLY:{} interRX:{} interRY:{}".format(CrossLX, CrossUY, CrossRX, CrossDY))
            sys.exit()
            
        return interSection / (predict_Area + ground_Area - interSection)

    def forward(self, bounding_boxes, ground_truth, batch_size=32,grid_size=64, img_size=448):  # 输入是 S * S * ( 2 * B + Classes)
        # 定义三个计算损失的变量 正样本定位损失 样本置信度损失 样本类别损失
        loss = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        iou_sum = 0
        object_num = 0
        mseLoss = nn.MSELoss()
        for batch in range(len(bounding_boxes)):
            for i in range(self.S):  # 先行 - Y
                for j in range(self.S):  # 后列 - X
                    # 取bounding box中置信度更大的框
                    if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                        predict_box = bounding_boxes[batch][i][j][5:]
                        # 另一个框是负样本
                        loss = loss + self.l_noobj * torch.pow(bounding_boxes[batch][i][j][4], 2)
                        loss_confidence += self.l_noobj * math.pow(bounding_boxes[batch][i][j][4], 2)
                    else:
                        predict_box = bounding_boxes[batch][i][j][0:5]
                        predict_box = torch.cat((predict_box, bounding_boxes[batch][i][j][10:]), dim=0)
                        # 另一个框是负样本
                        loss = loss + self.l_noobj * torch.pow(bounding_boxes[batch][i][j][9], 2)
                        loss_confidence += self.l_noobj * math.pow(bounding_boxes[batch][i][j][9], 2)
                    # 为拥有最大置信度的bounding_box找到最大iou的groundtruth_box
                    if ground_truth[batch][i][j][0][9] == 0:  # 面积为0的grount_truth 为了形状相同强行拼接的无用的0-box negative-sample
                        #A grount_truth with an area of 0 is a useless 0-box negative-sample that is forcibly spliced for the same shape
                        loss = loss + self.l_noobj * torch.pow(predict_box[4], 2)
                        loss_confidence += self.l_noobj * math.pow(predict_box[4].item(), 2)
                    else:
                        #object detected in that slice
                        object_num = object_num + 1
                        #find iou of the object detected in that slice
                        iou = self.iou(predict_box, ground_truth[batch][i][j][0], j * 64, i * 64)
                        iou_sum = iou_sum + iou
                        #gound truth shape = 16x7x7x1x15, to get rid of the 1 we do [0]
                        ground_box = ground_truth[batch][i][j][0]
                        loss = loss + self.l_coord * (torch.pow((ground_box[0] - predict_box[0]), 2) + torch.pow((ground_box[1] - predict_box[1]), 2) + torch.pow(torch.sqrt(ground_box[2] + 1e-8) - torch.sqrt(predict_box[2] + 1e-8), 2) + torch.pow(torch.sqrt(ground_box[3] + 1e-8) - torch.sqrt(predict_box[3] + 1e-8), 2))
                        loss_coord += self.l_coord * (math.pow((ground_box[0] - predict_box[0]), 2) + math.pow((ground_box[1] - predict_box[1]), 2) + math.pow(math.sqrt(ground_box[2] + 1e-8) - math.sqrt(predict_box[2] + 1e-8), 2) + math.pow(math.sqrt(ground_box[3] + 1e-8) - math.sqrt(predict_box[3] + 1e-8), 2))
                        loss = loss + torch.pow(ground_box[4] - predict_box[4], 2)
                        loss_confidence += math.pow(ground_box[4] - predict_box[4], 2)
                        #Z: This is to find MSE classification loss, will need to modify this for 1 class, before this only BB no change needed
                        ground_class = ground_box[10:]
                        predict_class = bounding_boxes[batch][i][j][self.B * 5:]
                        loss = loss + mseLoss(ground_class,predict_class) * self.Classes
                        loss_classes += mseLoss(ground_class,predict_class).item() * self.Classes
        #print("{} :{} iou_sum:{} object_num:{} iou:{}".format(loss_coord, loss_confidence, loss_classes, iou_sum, object_num, "nan" if object_num == 0 else (iou_sum / object_num)))
        return loss, loss_coord, loss_confidence, loss_classes, iou_sum, object_num
