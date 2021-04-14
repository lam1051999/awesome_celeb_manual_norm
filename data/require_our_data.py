# coding: utf8
import pickle
import torch.utils.data as Data
from PIL import ImageFilter
import random
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import json
import os
import cv2
from config import opt
from EDABK_utils import check_path_exist

class ourData(torch.utils.data.Dataset):

    def __init__(self, label, image_size=opt.image_size, transform=None, test=False, data_source=None, type_train="train", our_data_path=""):
        self.transform = transform
        self.test = test
        self.img_label = []
        self.image_size = image_size
        self.type_train = type_train
        print('myData, test=', self.test)

        if self.test == False:
            root_split = our_data_path.split("/")
            for i in range(len(root_split) - 1, -1, -1):
                if len(root_split[i]) != 0:
                    break

            root_temp = our_data_path[:len(our_data_path) - (len(root_split[i]) + (len(root_split) - i - 1))]
            root_temp = os.path.join(root_temp, "our_crop_data")
            dir_type = os.path.join(our_data_path, label)

            if type_train == "test":
                for class_ in os.listdir(dir_type):
                    class_type = os.path.join(dir_type, class_)
                    images = os.listdir(class_type)
                    for image_ in images: 
                        self.img_label.append({'path': os.path.join(class_type, image_), 'class': 1 if class_ == "spoof" else 0})
            else:
                raise Exception("No available data type")
            # write logic for the data if it is test data

    def __getitem__(self, index):  # 第二步装载数据，返回[img,label]
        if self.test == False:
            image_path = self.img_label[index]['path']
            label = self.img_label[index]['class']
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, (self.image_size, self.image_size))
                # img = img/255.0
                if self.transform is not None:
                    # print(self.transform)
                    img = self.transform(img)
                return np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)), int(label)

            except Exception as e:
                # get broken images
                check_path_exist(opt.rqds_crop_our_broken_images_train)
                if self.type_train == "train" or self.type_train == "val":
                    with open(os.path.join(opt.rqds_crop_our_broken_images_train, "broken.txt"), "a") as file_object:
                        file_object.write(image_path + "\n")
                else:
                    with open(os.path.join(opt.rqds_crop_our_broken_images_train, "broken_test.txt"), "a") as file_object:
                        file_object.write(image_path + "\n")
                if "train" in image_path:
                    if "spoof" in image_path:
                        temp = cv2.imread(os.path.join(opt.our_train_temp_images, "spoof.jpg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        # temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 1
                    else:
                        temp = cv2.imread(os.path.join(opt.our_train_temp_images, "live.jpeg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        # temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 0
                else:
                    if "spoof" in image_path:
                        temp = cv2.imread(os.path.join(opt.our_test_temp_images, "spoof.jpg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        # temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 1
                    else:
                        temp = cv2.imread(os.path.join(opt.our_test_temp_images, "live.jpeg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        # temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 0

            # write logic for the data if it is test data

    def __len__(self):
        return len(self.img_label)
