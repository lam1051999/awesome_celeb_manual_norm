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


def make_crop(base_height, label):
    root_split = opt.our_data.split("/")
    for i in range(len(root_split) - 1, -1, -1):
        if len(root_split[i]) != 0:
            break

    root_temp = opt.our_data[:len(opt.our_data) - (len(root_split[i]) + (len(root_split) - i - 1))]
    root_temp = os.path.join(root_temp, "our_crop_data")

    dir_type = os.path.join(opt.our_data, label)
    for class_ in os.listdir(dir_type):
        class_type = os.path.join(dir_type, class_)
        images = os.listdir(class_type)
        for image_ in tqdm(images):
            im_p = os.path.join(class_type, image_)
            partitions = image_.split(".")
            x = partitions[0] + "_crop." + partitions[1]
            x = os.path.join(root_temp, label, class_, x)
            if not os.path.exists(x):
                try:
                    im = cv2.imread(im_p)
                    # if im.shape[0] > base_height:
                    #     down_scale = math.ceil(im.shape[0] / base_height)
                    #     im = cv2.resize(im, (im.shape[1]//down_scale, im.shape[0]//down_scale),
                    #                     interpolation=cv2.INTER_AREA)
                    (h, w) = im.shape[:2]
                    blob = cv2.dnn.blobFromImage(
                        im, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()
                    if len(detections) > 0:
                        i = np.argmax(detections[0, 0, :, 2])
                        confidence = detections[0, 0, i, 2]

                        if confidence > 0.5:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            face = im[startY:endY, startX:endX]
                            if not os.path.exists(x[:len(x) - len(x.split("/")[-1])]):
                                os.makedirs(x[:len(x) - len(x.split("/")[-1])])

                            cv2.imwrite(x, face)
                        else:
                            cv2.imwrite(x, im)
                except Exception as e:
                    try:
                        cv2.imwrite(x, im)
                    except Exception as e1:
                        check_path_exist(opt.make_crop_our_broken_images)
                        with open(os.path.join(opt.make_crop_our_broken_images, "broken.txt"), "a") as file_object:
                            file_object.write(im_p + "\n")
                        if "train" in im_p:
                            if "spoof" in im_p:
                                im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "spoof.jpg"))
                                cv2.imwrite(x, im_temp)
                            else:
                                im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "live.jpeg"))
                                cv2.imwrite(x, im_temp)
                        else:
                            if "spoof" in im_p:
                                im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "spoof.jpg"))
                                cv2.imwrite(x, im_temp)
                            else:
                                im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "live.jpeg"))
                                cv2.imwrite(x, im_temp)


if __name__ == "__main__":

    if type_dir != "train" and type_dir != "test":
        print("Wrong type dir")
    else:
        IMG_WIDTH = opt.IMG_WIDTH
        IMG_HEIGHT = opt.IMG_HEIGHT
        protoPath = opt.protoPath
        modelPath = opt.modelPath
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        BASE_HEIGHT = 800
        if type_dir == "train":
            make_crop(BASE_HEIGHT, "train")
        elif type_dir == "test":
            make_crop(BASE_HEIGHT, "test")
