import json
import sys
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
from config import opt
import cv2
import numpy as np
import math
from tqdm import tqdm
from EDABK_utils import check_path_exist
import datetime
import glob
import time

from retinaface import RetinaFace
import reprlib

r = reprlib.Repr()
r.maxstring = 500

type_dir = sys.argv[1]

# python make_crop_data.py train
# python make_crop_data.py test


def make_crop_each_insight(filelists):
    json_dict = json.load(open(filelists[1]))
    temp_dict = {}
    for k in tqdm(json_dict.keys()):
        im_p = os.path.join(filelists[0], k)
        if os.path.exists(im_p):
            partitions = k.split(".")
            x = partitions[0]+"_crop." + partitions[1]
            x = x.replace("Data", "replay_crop")
            x = os.path.join(opt.base_dir, x)
            if not os.path.exists(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "replay_crop"))):
                if not os.path.exists(x[:len(x) - len(x.split("/")[-1])]):
                    os.makedirs(x[:len(x) - len(x.split("/")[-1])])
                try:
                    im = cv2.imread(im_p)
                    scales = [1024, 1980]
                    count = 0
                    im_shape = im.shape
                    target_size = scales[0]
                    max_size = scales[1]
                    im_size_min = np.min(im_shape[0:2])
                    im_size_max = np.max(im_shape[0:2])
                    im_scale = float(target_size) / float(im_size_min)
                    if np.round(im_scale * im_size_max) > max_size:
                        im_scale = float(max_size) / float(im_size_max)

                    scales = [im_scale]
                    flip = False

                    faces, landmarks = detector.detect(im, thresh, scales=scales, do_flip=flip)
                    if faces is not None:
                        for i in range(faces.shape[0]):
                            box = faces[i].astype(np.int)
                            color = (0, 0, 255)
                            (startX, startY, endX, endY) = (box[0] - PADDING, box[1] - PADDING, box[2] + PADDING, box[3] + PADDING)
                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(endX, im_shape[1])
                            endY = min(endY, im_shape[0])
                            crop_face = im[startY:endY, startX:endX]
                            count += 1
                            cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_{}.{}".format(partitions[0], count, partitions[1]).replace("Data", "replay_crop")), crop_face)
                    if count > 0:
                        temp_dict[k] = json_dict[k]

                    if count == 0:
                        # cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "replay_crop")), im)
                        pass
                except Exception as e:
                    # print("IF YOU SEE THIS LINE LIKE 1000 TIMES OR ABOVE, PLEASE STOP THE CODE. THERE MIGHT BE SOME PROBLEMS WITH GPU MEMORY, PLEASE CHECK THOSE")
                    print(r.repr(str(e)))
                    check_path_exist(opt.make_crop_our_broken_images)
                    with open(os.path.join(opt.make_crop_our_broken_images, "broken.txt"), "a") as file_object:
                        file_object.write(im_p + "\n")
                    pass

    if temp_dict:
        with open(filelists[1], 'w') as outfile:
            json.dump(temp_dict, outfile)
        
    

def make_crop(filelists, data_filelists):
    # celeb
    print("Crop CelebA-Spoof replay data!!!")
    make_crop_each_insight(filelists)
    # # our data
    # print("Crop our data!!!")
    # make_crop_each_insight(data_filelists)

if __name__ == "__main__":

    if type_dir != "train" and type_dir != "test":
        print("Wrong type dir")
    else:
        thresh = 0.8
        gpuid = 0
        detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
        PADDING = 20
        if type_dir == "train":
            make_crop(opt.celeb_replay_train_filelists, opt.data_train_filelists)
        elif type_dir == "test":
            make_crop(opt.celeb_replay_test_filelists, opt.data_test_filelists)