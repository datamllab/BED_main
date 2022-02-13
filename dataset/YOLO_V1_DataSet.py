from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
import augment
from augment import PhotometricDistort


'''Data Augmentation
TODO: Move to separate file
'''
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




class YoloV1DataSet(Dataset):

    def __init__(self, imgs_dir="./VOC2007/Train/JPEGImages",
                 annotations_dir="./VOC2007/Train/Annotations", img_size=224, S=7, B=2,
                 ClassesFile="./VOC2007/Train/VOC_remain_class.data", img_per_class=None,
                 train_root="./VOC2007/Train/ImageSets/Main/", ms_logger=None): # 图片路径、注解文件路径、图片尺寸、每个grid cell预测的box数量、类别文件
        
        
        
        
        self.transfrom = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            #PhotometricDistort(),
            #transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            transforms.ToTensor(), # height * width * channel -> channel * height * width
	    #AddGaussianNoise(0.1, 0.08),
            #transforms.RandomErasing(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])

        self.img_size = img_size
        self.S = S
        self.B = B
        self.grid_cell_size = self.img_size / self.S
        self.img_per_class = img_per_class

        self.img_selection(ClassesFile, train_root)
        self.generate_img_path(self.five, imgs_dir)
        self.generate_annotation_path(self.annot, annotations_dir)
        self.generate_ClassNameToInt(ClassesFile)
        self.getGroundTruth()

        ## loggout the dataset
        if ms_logger is None:
            print("Number of images: {}, {} classes".format(len(self.img_path), self.Classes))
            print("Number of annotation: {}".format(len(self.annotation_path)))
            print("Class name to Int: {}".format(self.ClassNameToInt))
            print("Int to Class name: {}".format(self.IntToClassName))
            print("Number of images per class: {}".format(self.Class_img_num))
        else:
            ms_logger.info("Number of images: %s, %s classes", str(len(self.img_path)), str(self.Classes))
            ms_logger.info("Number of annotation: %s", str(len(self.annotation_path)))
            ms_logger.info("Class name to Int: %s", str(self.ClassNameToInt))
            ms_logger.info("Int to Class name: %s", str(self.IntToClassName))
            ms_logger.info("Number of images per class: %s", str(self.Class_img_num))


    def generate_img_path(self, img_names, imgs_dir):
        img_names.sort() # 图片和文件排序后可以按照相同索引对应
        self.img_path = []
        for img_name in img_names:
            self.img_path.append(os.path.join(imgs_dir, img_name))
   

    def generate_annotation_path(self, annotation_names, annotations_dir):
        annotation_names.sort()  # 图片和文件排序后可以按照相同索引对应
        self.annotation_path = []
        for annotation_name in annotation_names:
            self.annotation_path.append(os.path.join(annotations_dir, annotation_name))

    def generate_ClassNameToInt(self, ClassesFile):
        self.ClassNameToInt = {}
        # self.instance_counting = {}
        self.IntToClassName = {}
        classIndex = 0
        with open(ClassesFile, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                self.ClassNameToInt[line] = classIndex  # 根据类别名制作索引
                self.IntToClassName[classIndex] = line
                # self.instance_counting[line] = 0
                classIndex = classIndex + 1
        self.Classes = classIndex  # 一共的类别个数

    def img_selection(self, ClassesFile, train_root):
        #Generate image paths for classes in ClassesFile with img_per_class number of images per class

        def generate_class_index_fname(train_root, class_name):
            class_index_fname_buf = os.listdir(train_root)

            if class_name + "_test.txt" in class_index_fname_buf:
                return os.path.join(train_root, class_name + "_test.txt") # testing set
            else:
                return os.path.join(train_root, class_name + "_trainval.txt") # training set
                # return os.path.join(train_root, class_name + "_train.txt") # training set
                # return os.path.join(train_root, class_name + "_val.txt") # training set

        def one_class_img(img_index_fname):
            img_num = 0
            with open(img_index_fname, 'r') as f:
                for line in f:

                    img_valid_flag = line.strip().split(" ")[-1]
                    if img_valid_flag == "1":
                        img_No = line.strip().split(" ")[0]
                        img_num += 1
                        if img_No + ".jpg" not in self.five:
                            self.five.append(img_No + ".jpg")
                            self.annot.append(img_No + ".xml")
                            if self.img_per_class is not None and img_num == self.img_per_class:
                                return img_num
            return img_num

        self.img_index_fpaths = {}
        f =  open(ClassesFile, 'r')
        lines = f.readlines()
        for l in lines:
            class_name = l.strip()
            self.img_index_fpaths[class_name] = generate_class_index_fname(train_root, class_name)

        self.five = []
        self.annot = []
        self.Class_img_num = {}
        for class_name, img_index_fname in self.img_index_fpaths.items():
            # print(class_name, img_index_fname)
            img_num_class = one_class_img(img_index_fname)
            self.Class_img_num[class_name] = img_num_class


    # PyTorch 无法将长短不一的list合并为一个Tensor
    def getGroundTruth(self):
        self.ground_truth = [[[list() for i in range(self.S)] for j in range(self.S)] for k in
                             range(len(self.img_path))]  # 根据标注文件生成ground_truth
        ground_truth_index = 0
        for annotation_file in self.annotation_path:
            ground_truth = [[list() for i in range(self.S)] for j in range(self.S)]
            # 解析xml文件--标注文件
            tree = ET.parse(annotation_file)
            annotation_xml = tree.getroot()
            # 计算 目标尺寸 -> 原图尺寸 self.img_size * self.img_size , x的变化比例
            width = (int)(annotation_xml.find("size").find("width").text)
            scaleX = self.img_size / width
            # 计算 目标尺寸 -> 原图尺寸 self.img_size * self.img_size , y的变化比例
            height = (int)(annotation_xml.find("size").find("height").text)
            scaleY = self.img_size / height
            # 因为两次除法的误差可能比较大 这边采用除一次乘一次的方式
            # 一个注解文件可能有多个object标签，一个object标签内部包含一个bnd标签
            objects_xml = annotation_xml.findall("object")
            for object_xml in objects_xml:
                # 获取目标的名字
                class_name = object_xml.find("name").text
                if class_name not in self.ClassNameToInt: # 不属于我们规定的类
                    continue
                bnd_xml = object_xml.find("bndbox")
                # 目标尺度放缩
                xmin = (int)((int)(bnd_xml.find("xmin").text) * scaleX)
                ymin = (int)((int)(bnd_xml.find("ymin").text) * scaleY)
                xmax = (int)((int)(bnd_xml.find("xmax").text) * scaleX)
                ymax = (int)((int)(bnd_xml.find("ymax").text) * scaleY)
                # 目标中心点
                centerX = (xmin + xmax) / 2
                centerY = (ymin + ymax) / 2
                # 当前物体的中心点落于 第indexI行 第indexJ列的 grid cell内
                indexI = (int)(centerY / self.grid_cell_size)
                indexJ = (int)(centerX / self.grid_cell_size)
                # 真实物体的list
                #Z: Change here when class name sam as our class, keep BB 
                ClassIndex = self.ClassNameToInt[class_name]
                ClassList = [0 for i in range(self.Classes)]
                ClassList[ClassIndex] = 1
                #Z: gt [normalised 4 coord, confidence, unnormalise coord, area] how to modify for negative sample?
                ground_box = list([centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)])
                #增加上类别
                ground_box.extend(ClassList)
                ground_truth[indexI][indexJ].append(ground_box)

            #同一个grid cell内的多个groudn_truth，选取面积最大的两个
            for i in range(self.S):
                for j in range(self.S):
                    if len(ground_truth[i][j]) == 0:
                        #print(self.Classes,"class")
                        self.ground_truth[ground_truth_index][i][j].append([0 for i in range(10 + self.Classes)])
                    else:
                        ground_truth[i][j].sort(key = lambda box: box[9], reverse=True)
                        #Z: For each image(gt_index) store 7x7 BB annotations
                        self.ground_truth[ground_truth_index][i][j].append(ground_truth[i][j][0])

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth).float()

    def __getitem__(self, item):
        # height * width * channel
        #print(len(self.img_path),"KKKKKKK")
        img_data = cv2.imread(self.img_path[item])
        img_data = cv2.resize(img_data, (224, 224), interpolation=cv2.INTER_AREA)
        
        img_data = self.transfrom(img_data)
        return img_data,self.ground_truth[item]

    def __len__(self):
        return len(self.img_path)
