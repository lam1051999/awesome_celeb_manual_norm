import json
import sys
import os
from config import opt
import cv2
import numpy as np
import math
from tqdm import tqdm
from EDABK_utils import check_path_exist

type_dir = sys.argv[1]

# python make_crop_image.py train
# python make_crop_image.py test


def make_crop(filelists):
    json_dict = json.load(open(filelists[1]))
    for k in tqdm(json_dict.keys()):
        im_p = os.path.join(filelists[0], k)
        if os.path.exists(im_p):
            partitions = k.split(".")
            x = partitions[0]+"_crop." + partitions[1]
            x = x.replace("Data", "crop")
            x = os.path.join(opt.base_dir, x)
            if not os.path.exists(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop"))):
                try:
                    im = cv2.imread(im_p)
                    (h, w) = im.shape[:2]
                    blob = cv2.dnn.blobFromImage(
                        im, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()
                    if len(detections) > 0:
                        count = 0
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence >= 0.5:
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")
                                if not os.path.exists(x[:len(x) - len(x.split("/")[-1])]):
                                    os.makedirs(x[:len(x) - len(x.split("/")[-1])])
                                if startX <= w and endX <= w and startY <= h and endY <= h:
                                    count += 1
                                    face = im[startY:endY, startX:endX]
                                    cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_{}.{}".format(partitions[0], count, partitions[1]).replace("Data", "crop")), face)
                        if count == 0:
                            cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
                except Exception as e:
                    try:
                        cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
                    except Exception as e1:
                        check_path_exist(opt.make_crop_broken_images)
                        with open(os.path.join(opt.make_crop_broken_images, "broken.txt"), "a") as file_object:
                            file_object.write(im_p + "\n")
                        if "train" in im_p:
                            if "spoof" in im_p:
                                im_temp = cv2.imread(os.path.join(opt.train_temp_images, "006485_crop.jpg"))
                                cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
                            else:
                                im_temp = cv2.imread(os.path.join(opt.train_temp_images, "006159_crop.jpg"))
                                cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
                        else:
                            if "spoof" in im_p:
                                im_temp = cv2.imread(os.path.join(opt.test_temp_images, "501103_crop.png"))
                                cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
                            else:
                                im_temp = cv2.imread(os.path.join(opt.test_temp_images, "495823_crop.png"))
                                cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)


if __name__ == "__main__":

    if type_dir != "train" and type_dir != "test":
        print("Wrong type dir")
    else:
        IMG_WIDTH = opt.IMG_WIDTH
        IMG_HEIGHT = opt.IMG_HEIGHT
        protoPath = opt.protoPath
        modelPath = opt.modelPath
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        if type_dir == "train":
            make_crop(opt.celeb_train_filelists)
        elif type_dir == "test":
            make_crop(opt.celeb_test_filelists)
