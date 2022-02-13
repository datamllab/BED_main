import numpy as np

def calculate_map_main(gt_results, pred_results, iou_gt_thr=0.5, class_num=5):
    '''
    说明: 此函数用于计算目标检测中的mAP
    输入:
          gt_results:  list, 每一个元素对应一个样本中的所有目标的真实值
          如gt_gt_result[0] = [
                                [xmin, ymin, xmax, ymax, label],
                                [xmin, ymin, xamx, yamx, label],
                                ...
                              ]
          pred_results: list, 每一个元素对应一个样本中的所有目标的预测值
          如pred_results[0] = [
                                [xmin, ymin, xmax, ymax, label, score],
                                [xmin, ymin, xamx, yamx, label, score],
                                ...
                              ]
          iou_gt_thr:   float, 用于判定正负样本, 如iou_gt_thr=0.5, 计算出的就是mAP 0.5
          class_num:    int, 类别数
    输出：
          calss_ap:     array, 各类别的AP
          mean_ap :     float, 各类别平均得到mAP
    '''

    all_tp = [ [] for i in range(class_num) ]    # 用于存放各类所有的tp
    all_fp = [ [] for i in range(class_num) ]    # 用于存放各类所有的fp
    all_score = [ [] for i in range(class_num) ] # 用于存放bbox对应的scores
    all_gt_num = np.zeros([class_num])           # 用于存放各类的真实目标数, 之后计算recall会用到
    data_num = len(gt_results)                        # 样本总数

    # 对于每一个样本, 计算tp, fp, 并且统计预测bbox的score以及真实目标的个数
    for i in range(data_num):
        gt_result = gt_results[i]
        pred_result = pred_results[i]
        tp, fp, score, gt_num = calculate_tpfp_single(gt_result, pred_result, iou_gt_thr, class_num)
        # 按类别更新到总数中
        for n in range(class_num):
            all_tp[n].extend(tp[n])
            all_fp[n].extend(fp[n])
            all_score[n].extend(score[n])
            all_gt_num[n] += gt_num[n]

    # 计算出各类的AP，进而得到mAP
    all_map = calculate_map(all_tp, all_fp, all_score, all_gt_num, class_num)
    mean_ap = np.mean(all_map)
    print('mAP', mean_ap)
    return all_map, mean_ap


def calculate_tpfp_single(gt_result, pred_result, iou_gt_thr, class_num):
    '''
    说明: 此函数用于计算单个样本的tp和fp, 并且存放对应的score, 统计真实目标的个数
    输入:
          gt_result:  list, 一个样本中所有的目标的真实值
          如gt_result=[
                        [xmin, ymin, xmax, ymax, label],
                        [xmin, ymin, xmax, ymax, label],
                        ...
                      ]
          pred_result: list, 一个样本中所有的目标的预测值
          如pred_result=[
                          [xmin, ymin, xmax, ymax, label, score],
                          [xmin, ymin, xmax, ymax, label, score],
                          ...
                        ]
          iou_gt_thr:  float, 用于判断正负样本的iou阈值
          class_num:   类别数
    输出:
          all_tp:      list, 每一个元素对应该样本计算出的对应类别的tp
          all_fp:      list, 每一个元素对应该样本计算出的对应类别的fp
          all_score:   list, 每一个元素对应该样本bbox对应类别的score
          gt_num:      list, 每一个元素对应该样本对应类别的真实目标数
    '''

    all_tp = [[] for i in range(class_num)]
    all_fp = [[] for i in range(class_num)]
    all_score = [[] for i in range(class_num)]
    gt_num = np.zeros([class_num])

    # 逐个类别提取真实bbox和预测bbox
    for i in range(class_num):
        tp = []
        fp = []
        score = []

        match_gt_bbox = [obj[0:4] for obj in gt_result if int(obj[4] - 1) == i]
        match_pred_bbox = [obj[0:4] for obj in pred_result if int(obj[4]) == i]
        match_pred_score = [obj[5] for obj in pred_result if int(obj[4]) == i]

        len_gt = len(match_gt_bbox)
        len_pred = len(match_pred_bbox)

        if len_gt == 0 and len_pred != 0:
            # 说明不存在该类目标，但是预测出来了，属于误检
            score.extend(match_pred_score)
            for k in range(len_pred):
                tp.extend([0])
                fp.extend([1])

        if len_gt != 0 and len_pred != 0:
            # 说明存在该目标，并且检测出来了,那么计算若干gt与若干pred的iou
            score.extend(match_pred_score)
            ious = calculate_iou(match_gt_bbox, match_pred_bbox)
            max_iou = np.max(ious, axis=0)  # [x,x,x...] 每一个预测框与某个gt最大的iou

            # 使用iou_gt_thr来进行正负样本的判定，若满足条件，则为tp，否则为fp
            for k in range(len_pred):
                if max_iou[k] >= iou_gt_thr:
                    tp.extend([1])
                    fp.extend([0])
                if max_iou[k] < iou_gt_thr:
                    tp.extend([0])
                    fp.extend([1])

        all_tp[i].extend(tp)
        all_fp[i].extend(fp)
        all_score[i].extend(score)
        gt_num[i] += len_gt


    return all_tp, all_fp, all_score, gt_num


def calculate_area(bbox):
    # 计算一个bbox的面积
    w = max(bbox[2] - bbox[0], 0)
    h = max(bbox[3] - bbox[1], 0)
    w = max(0, w)
    h = max(0, h)
    return w * h

def calculate_inter(bbox1, bbox2):
    # 计算两个bbox的交集面积
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    return calculate_area([xmin, ymin, xmax, ymax])

def calculate_union(bbox1, bbox2):
    # 计算两个bbox的并集面积
    area1 = calculate_area(bbox1)
    area2 = calculate_area(bbox2)
    inter = calculate_inter(bbox1, bbox2)
    union = area1 + area2 - inter
    return union

def IOU(bbox1, bbox2):
    # 计算两个bbox的iou
    inter = calculate_inter(bbox1, bbox2)
    union = calculate_union(bbox1, bbox2)
    iou = inter / union
    return iou

def calculate_iou(bbox1, bbox2):
    '''
    说明: 此函数用于计算M个bbox与N个bbox的iou
    输入:
          bbox1: list, 每一个元素是一个bbox, 如bbox1=[
                                                     [xmin, ymin, xamx, ymax],
                                                     [xmin, ymin, xmax, ymax],
                                                     ...
                                                    ]
          bbox2: list, 每一个元素是一个bbox, 如bbox2=[
                                                     [xmin, ymin, xamx, ymax],
                                                     [xmin, ymin, xmax, ymax],
                                                     ...
                                                    ]
    输出:
          ans:   array, size=[M, N], 计算出的iou矩阵
    '''

    len_1 = len(bbox1)
    len_2 = len(bbox2)
    ans = np.zeros([len_1, len_2])
    for i in range(len_1):
        for j in range(len_2):
            # 计算bbox1[i]和bbox2[j]的iou
            ans[i, j] = IOU(bbox1[i], bbox2[j])
    return ans


def calculate_map(all_tp, all_fp, all_score, all_gt_num, class_num):
    '''
    说明: 此函数的输入为所有类别的tp, fp, score和真实目标数, 计算每一个类别的AP
    输入:
          all_tp:     list,  每个元素是该类别下的tp
          all_fp:     list,  每个元素是该类别下的fp
          all_score:  list,  每个元素是该类别下预测bbox对应的score
          all_gt_num: list,  每个元素是该类下真实母目标的个数
          class_num:  int,   类别数
    输出:
          all_map:    array, 每个元素是该类的AP
    '''

    all_map = np.zeros([class_num])
    for i in range(class_num):
    # 首先提取出每一类的信息
        class_tp = all_tp[i]
        class_fp = all_fp[i]
        class_score = all_score[i]
        class_gt_num = all_gt_num[i]
        # 计算每一类的PR曲线
        class_P, class_R = calculate_PR(class_tp, class_fp, class_score, class_gt_num)
        # 计算PR曲线的面积，即AP
        class_map = calculate_map_single(class_P, class_R)
        # 写入该类别下
        all_map[i] = class_map
    return all_map


def calculate_PR(class_tp, class_fp, class_score, class_gt_num):
    '''
    说明: 此函数用于计算某一类的PR曲线
    输入:
          class_tp:     list, 该类下的tp, 每个元素为0或1, 代表当前样本是否为正样本
          class_fp:     list, 该类下的fp, 每个元素为0或1, 代表当前样本是否为负样本
          class_score:  list, 该类下预测bbox对应的score
          class_gt_num: int,  类别数
    输出:
          P: list, 该类下的查准率曲线
          R: list, 该类下的查全率曲线
    '''

    # 按照score排序
    sort_inds = np.argsort(class_score)[::-1].tolist()
    tp = [class_tp[i] for i in sort_inds]
    fp = [class_fp[i] for i in sort_inds]
    # 累加
    tp = np.cumsum(tp).tolist()
    fp = np.cumsum(fp).tolist()
    # 计算PR
    P = [tp[i] / (tp[i] + fp[i]) for i in range(len(tp))]
    R = [tp[i] / class_gt_num for i in range(len(tp))]
    return P, R


def calculate_map_single(P, R):
    '''
    说明: 此函数用于计算PR曲线的面积, 即AP
    输入:
          P: list, 查准率曲线
          R: list, 查全率曲线
    输出:
          single_map: float, 曲线面积, 即AP
    '''
    mpre = np.concatenate(([0.], P, [0.]))
    mrec = np.concatenate(([0.], R, [1.]))
    for i in range(np.size(mpre) - 1, 0, -1):
        # mpre的平整化
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # 寻找mrec变化的坐标
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # 计算面积
    single_map = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return single_map



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
    nms_boxes_buf = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        predict_boxes = []
        nms_boxes = []
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
            # print("Class index:{}".format(classIndex), "Confidence:", assured_box)
            assured_box[4] = assured_box[4] * assured_box[5 + classIndex] #修正置信度为 物体分类准确度 × 含有物体的置信度
            assured_box[5] = classIndex
            nms_boxes.append(assured_box)
            i = 1
            while i < len(predict_boxes):
                if iou(assured_box,predict_boxes[i]) <= iou_threshold:
                    temp.append(predict_boxes[i])
                i = i + 1
            predict_boxes = temp

        nms_boxes_buf.append(nms_boxes)

    return nms_boxes_buf



def gt_std(gt_results, S=7, B=2, img_size=224):

    gt_results_all = []
    grid_size = img_size / S
    for instance_index in range(gt_results.shape[0]): # N
        gt_results_instance = []
        for index_i in range(gt_results.shape[1]): # 7
            for index_j in range(gt_results.shape[2]): # 7
                gridX = grid_size * index_i
                gridY = grid_size * index_j
                area = gt_results[instance_index, index_i, index_j, 9]
                if area > 0:
                    gt_results_patch = gt_results[instance_index, index_i, index_j].tolist()
                    centerX = (int)(gridX + gt_results_patch[0] * grid_size)
                    centerY = (int)(gridY + gt_results_patch[1] * grid_size)
                    width = (int)(gt_results_patch[2] * img_size)
                    height = (int)(gt_results_patch[3] * img_size)
                    class_idx = int(gt_results[instance_index, index_i, index_j, 10:].argmax())
                    gt_results_patch[0] = max(0, (int)(centerX - width / 2))
                    gt_results_patch[1] = max(0, (int)(centerY - height / 2))
                    gt_results_patch[2] = min(img_size - 1, (int)(centerX + width / 2))
                    gt_results_patch[3] = min(img_size - 1, (int)(centerY + height / 2))
                    gt_results_patch[4] = class_idx + 1
                    gt_results_instance.append(gt_results_patch[0:5])

        gt_results_all.append(gt_results_instance)

    return gt_results_all





