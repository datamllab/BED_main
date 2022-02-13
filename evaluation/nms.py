from copy import deepcopy

import numpy as np
import torch
from torch import nn

from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div


def torch_NMS_max(bounding_boxes, S=7, B=2, img_size=56, confidence_threshold=0.55, classes=20):
    bounding_boxes = bounding_boxes.cpu().detach().numpy().tolist()
    nms_boxes = []
    predict_boxes = []
    grid_size = img_size / S
    cls_start = B * 5
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                grid = bounding_boxes[batch][i][j]
                cls_score = 0
                cls_idx = 0
                max_score = 0
                for b in range(B):
                    score = grid[5 * b + 4]
                    if score > max_score:
                        max_score = score
                        bounding_box = grid[b * 5:(b + 1) * 5]
                if max_score < confidence_threshold:
                    continue
                for k in range(cls_start, cls_start + classes):
                    if grid[k] > cls_score:
                        cls_score = grid[k]
                        cls_idx = k - cls_start
                #First 5 contain the most confident BB and next 2 are the conf score and index of the classification result
                bounding_box.append(cls_score)
                bounding_box.append(cls_idx)
                # bounding_box[4] = max_score * cls_score
                predict_boxes.append(bounding_box)

                gridX = grid_size * i
                gridY = grid_size * j

                centerX = (int)(gridX + bounding_box[0] * grid_size)
                centerY = (int)(gridY + bounding_box[1] * grid_size)
                width = (int)(bounding_box[2] * img_size)
                height = (int)(bounding_box[3] * img_size)

                bounding_box[0] = max(0, (int)(centerX - width / 2))
                bounding_box[1] = max(0, (int)(centerY - height / 2))
                bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))

                if len(nms_boxes) == 0:
                    nms_boxes.append(bounding_box)
                    continue
                if bounding_box[4] > nms_boxes[-1][4]:
                    nms_boxes[-1] = bounding_box
        return nms_boxes, predict_boxes


def NMS_max(bounding_boxes, S=7, B=2, img_size=56, confidence_threshold=q17p14(0.55), classes=20, topk=1):
    nms_boxes = []
    predict_boxes = []
    grid_size = img_size / S
    cls_start = B * 5
    channels = cls_start + classes
    for start in range(0, S * S * channels, channels):
        grid = bounding_boxes[start:start + channels]
        # print(grid)
        i = start // (S * channels)
        j = start // channels % S
        cls_score = 0
        cls_idx = 0
        max_score = 0
        for b in range(B):
            score = grid[5 * b + 4]
            if score > max_score:
                max_score = score
                bounding_box = grid[b * 5:(b + 1) * 5]
                # bounding_box[0] = i
                # bounding_box[1] = j
                # bounding_box[2] = b

        # if max_score < confidence_threshold:
            # continue
        for k in range(cls_start, cls_start + classes):
            if grid[k] > cls_score:
                cls_score = grid[k]
                cls_idx = k - cls_start
        bounding_box.append(cls_score)
        bounding_box.append(cls_idx)
#         bounding_box.extend(grid[cls_start:cls_start+classes])

        # bounding_box[4] = q_mul(max_score, cls_score)
        predict_boxes.append(bounding_box)
        # print(bounding_box)

        gridX = grid_size * i
        gridY = grid_size * j

        centerX = (gridX + q_mul(bounding_box[0], grid_size))
        centerY = (gridY + q_mul(bounding_box[1], grid_size))
        width = q_mul(bounding_box[2], img_size)
        height = q_mul(bounding_box[3], img_size)

        bounding_box[0] = max(0, (centerX - (width >> 1)))
        bounding_box[1] = max(0, (centerY - (height >> 1)))
        bounding_box[2] = min(img_size - 1, (centerX + (width >> 1)))
        bounding_box[3] = min(img_size - 1, (centerY + (height >> 1)))

        # if len(nms_boxes) == 0:
        #     nms_boxes.append(bounding_box)
        #     continue
        # if bounding_box[4] > nms_boxes[-1][4]:
        #     nms_boxes[-1] = bounding_box

        insert_index = len(nms_boxes)
        for index, box in enumerate(nms_boxes):
            # print(bounding_box[4], box[4], (bounding_box[4] > box[4]))
            if bounding_box[4] > box[4]:
                insert_index = index
                break
        nms_boxes.insert(insert_index, bounding_box)
        if len(nms_boxes) > topk:
            nms_boxes.pop()

        # print([x[4] for x in nms_boxes], insert_index)

    return nms_boxes, predict_boxes


def approx_softmax(arr, start, end):
    nominator = []
    for i in range(start, end):
        e = arr[i]
        nominator.append(pow(2, e / 16384.0))
        # nominator.append(pow(2, e >> 14))
    denominator = sum(nominator)
    for i in range(start, end):
        arr[i] = q17p14(nominator[i - start] / denominator)
        # arr[i] = nominator[i - start] / denominator


def torch_post_process(x):
    x = torch.FloatTensor(x).reshape(7, 7, 15).unsqueeze(dim=0) / (2 ** 14)
    bounding_box = x[..., :10]
    # Q17.14: 1 signed bit, 17-bit integer part, 14-bit fractional part
    class_possible = nn.Softmax(dim=3)(x[:, :, :, 10:])
    x = torch.cat((bounding_box, class_possible), dim=3)
    return x


def post_process(x):
    cls_start = 2 * 5
    cls = 5 # 20
    S = 7
    channels = cls_start + cls
    for i in range(0, S * S * channels, channels):
        start = i + cls_start
        end = start + cls
        approx_softmax(x, start, end)
    return x


if __name__ == '__main__':
    path = 'feature_map.npy'
    feature_map = np.load(path)
    q_sigmoid, l, h, resolution = generate_q_sigmoid()
    x = sigmoid_lut(feature_map, q_sigmoid, l, h, resolution)
    print(x)

    torch_softmax = torch_post_process(x)
    # print(torch_softmax.numpy().tolist()[0][0][0])
    boxes1, pred_boxes1 = torch_NMS_max(torch_softmax)
    print('torch', boxes1[-1])  # [0, 20, 24, 54, 0.82427978515625, 0.0579104907810688, 19]

    print(feature_map.shape)

    softmax_x = post_process(x)
    print(softmax_x)
    boxes2, pred_boxes2 = NMS_max(softmax_x)
    print('approx', boxes2[-1])  # [0, 20.0, 25.0, 54.0, 13505, 908, 19]  or [0, 20.0, 25.0, 54.0, 13505, 819, 19]
