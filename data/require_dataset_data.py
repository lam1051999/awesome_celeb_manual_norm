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

class mergedData(torch.utils.data.Dataset):

    def __init__(self, filelists="", data_filelists="", image_size=224, transform=None, test=False, data_source=None, type_train="train", base_dir=""):
        self.transform = transform
        self.test = test
        self.img_label = []
        self.image_size = image_size
        self.type_train = type_train
        self.base_dir = base_dir
        print('myData, test=', self.test)

        if self.test == False:
            self.load_filelists(filelists)
            self.load_filelists(data_filelists)

        random.shuffle(self.img_label)

    def load_filelists(self, filelists):
        json_dict = json.load(open(filelists[1]))
        json_dict_keys = list(json_dict.keys())
        json_dict_keys.sort()
        if self.type_train == "train":
            for k in json_dict_keys[:int(len(json_dict_keys) * 85/100)]:
                p_ = os.path.join(self.base_dir, k.replace("Data", "photo_crop"))
                if os.path.exists(p_[:len(p_) - len(p_.split("/")[-1])]):
                    for f in os.listdir(p_[:len(p_) - len(p_.split("/")[-1])]):
                        if (p_.split("/")[-1]).split(".")[0] in f:
                            self.img_label.append({'path': p_[:len(p_) - len(p_.split("/")[-1])] + f, 'class': json_dict[k][-1]})
        elif self.type_train == "val":
            for k in json_dict_keys[int(len(json_dict_keys) * 85/100):]:
                p_ = os.path.join(self.base_dir, k.replace("Data", "photo_crop"))
                if os.path.exists(p_[:len(p_) - len(p_.split("/")[-1])]):
                    for f in os.listdir(p_[:len(p_) - len(p_.split("/")[-1])]):
                        if (p_.split("/")[-1]).split(".")[0] in f:
                            self.img_label.append({'path': p_[:len(p_) - len(p_.split("/")[-1])] + f, 'class': json_dict[k][-1]})
        elif self.type_train == "test":
            for k in json_dict_keys:
                p_ = os.path.join(self.base_dir, k.replace("Data", "photo_crop"))
                if os.path.exists(p_[:len(p_) - len(p_.split("/")[-1])]):
                    for f in os.listdir(p_[:len(p_) - len(p_.split("/")[-1])]):
                        if (p_.split("/")[-1]).split(".")[0] in f:
                            self.img_label.append({'path': p_[:len(p_) - len(p_.split("/")[-1])] + f, 'class': json_dict[k][-1]})
        else:
            raise Exception("No available data type")


    def __getitem__(self, index):  # 第二步装载数据，返回[img,label]
        if self.test == False:
            image_path = self.img_label[index]['path']
            label = self.img_label[index]['class']
            try:
                img = cv2.imread(image_path)
                img = Image.fromarray(img)
                if self.transform is not None:
                    img = self.transform(img)
                return img, int(label)

            except Exception as e:
                # get broken images
                print(e)
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
                        temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 1
                    else:
                        temp = cv2.imread(os.path.join(opt.our_train_temp_images, "live.jpeg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 0
                else:
                    if "spoof" in image_path:
                        temp = cv2.imread(os.path.join(opt.our_test_temp_images, "spoof.jpg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 1
                    else:
                        temp = cv2.imread(os.path.join(opt.our_test_temp_images, "live.jpeg"))
                        temp = cv2.resize(
                            temp, (self.image_size, self.image_size))
                        temp = temp/255.0
                        return np.transpose(np.array(temp, dtype=np.float32), (2, 0, 1)), 0

            # write logic for the data if it is test data

    def __len__(self):
        return len(self.img_label)
