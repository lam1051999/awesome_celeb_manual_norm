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

class myDataCrop(torch.utils.data.Dataset):

    def __init__(self, filelists="", image_size=224, transform=None, test=False, data_source=None, type_train="train", base_dir=""):
        self.transform = transform
        self.test = test
        self.img_label = []
        self.image_size = image_size
        self.type_train = type_train
        print('myData, test=', self.test)

        if self.test == False:
            # Loading and processing intra_test json
            json_dict = json.load(open(filelists[1]))
            json_dict_keys = list(json_dict.keys())
            if type_train == "train":
                for k in json_dict_keys[:int(len(json_dict_keys) * 85/100)]:
                    p_ = os.path.join(base_dir, k)
                    partitions = p_.split(".")
                    if len(partitions) == 2:
                        self.img_label.append({'path': (
                            partitions[0]+"_crop." + partitions[1]).replace("Data", "crop"), 'class': json_dict[k][-1]})
            elif type_train == "val":
                for k in json_dict_keys[int(len(json_dict_keys) * 85/100):]:
                    p_ = os.path.join(base_dir, k)
                    partitions = p_.split(".")
                    if len(partitions) == 2:
                        self.img_label.append({'path': (
                            partitions[0]+"_crop." + partitions[1]).replace("Data", "crop"), 'class': json_dict[k][-1]})
            elif type_train == "test":
                for k in json_dict_keys:
                    p_ = os.path.join(base_dir, k)
                    partitions = p_.split(".")
                    if len(partitions) == 2:
                        self.img_label.append({'path': (
                            partitions[0]+"_crop." + partitions[1]).replace("Data", "crop"), 'class': json_dict[k][-1]})
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
                if self.transform is not None:
                    # print(self.transform)
                    img = self.transform(img)
                return np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)), int(label)

            except Exception as e:
                # get broken images
                check_path_exist(opt.rqds_crop_broken_images_train)
                if self.type_train == "train" or self.type_train == "val":
                    with open(os.path.join(opt.rqds_crop_broken_images_train, "broken.txt"), "a") as file_object:
                        file_object.write(image_path + "\n")
                else:
                    with open(os.path.join(opt.rqds_crop_broken_images_train, "broken_test.txt"), "a") as file_object:
                        file_object.write(image_path + "\n")
                if "train" in image_path:
                    if "spoof" in image_path:
                        temp = cv2.imread(os.path.join(opt.train_temp_images, "474951_crop.jpg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 1
                    else:
                        temp = cv2.imread(os.path.join(opt.train_temp_images, "052210_crop.jpg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 0
                else:
                    if "spoof" in image_path:
                        temp = cv2.imread(os.path.join(opt.test_temp_images, "495026_crop.png"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 1
                    else:
                        temp = cv2.imread(os.path.join(opt.test_temp_images, "498269_crop.png"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 0

            # write logic for the data if it is test data

    def __len__(self):
        return len(self.img_label)
