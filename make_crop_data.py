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

# def make_crop_each_yolo(filelists):
#     json_dict = json.load(open(filelists[1]))
#     for k in tqdm(json_dict.keys()):
#         im_p = os.path.join(filelists[0], k)
#         if os.path.exists(im_p):
#             partitions = k.split(".")
#             x = partitions[0]+"_crop." + partitions[1]
#             x = x.replace("Data", "crop")
#             x = os.path.join(opt.base_dir, x)
#             if not os.path.exists(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop"))):
#                 if not os.path.exists(x[:len(x) - len(x.split("/")[-1])]):
#                     os.makedirs(x[:len(x) - len(x.split("/")[-1])])
#                 try:
#                     im = cv2.imread(im_p)
#                     (h, w) = im.shape[:2]
#                     classes, scores, boxes = model.detect(im, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
#                     count = 0
#                     for (classid, score, box) in zip(classes, scores, boxes):
#                         if score >= 0.5:
#                             (startX, startY, endX, endY) = (box[0], box[1], box[0] + box[2], box[1] + box[3])
#                             if startX <= w and endX <= w and startY <= h and endY <= h:
#                                 (startX, startY, endX, endY) = (startX - PADDING, startY - PADDING, endX + PADDING, endY + PADDING)
#                                 if startX < 0:
#                                     startX = 0
#                                 if startY < 0:
#                                     startY = 0
#                                 face = im[startY:endY, startX:endX]
#                                 count += 1
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_{}.{}".format(partitions[0], count, partitions[1]).replace("Data", "crop")), face)

#                     if count == 0:
#                         cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
#                 except Exception as e:
#                     print("Depth 1")
#                     try:
#                         cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
#                     except Exception as e1:
#                         print("Depth 2")
#                         check_path_exist(opt.make_crop_our_broken_images)
#                         with open(os.path.join(opt.make_crop_our_broken_images, "broken.txt"), "a") as file_object:
#                             file_object.write(im_p + "\n")
#                         if "train" in im_p:
#                             if "spoof" in im_p:
#                                 im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "spoof.jpg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
#                             else:
#                                 im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "live.jpeg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
#                         else:
#                             if "spoof" in im_p:
#                                 im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "spoof.jpg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
#                             else:
#                                 im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "live.jpeg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)


# def make_crop_each(filelists):
#     json_dict = json.load(open(filelists[1]))
#     for k in tqdm(json_dict.keys()):
#         im_p = os.path.join(filelists[0], k)
#         if os.path.exists(im_p):
#             partitions = k.split(".")
#             x = partitions[0]+"_crop." + partitions[1]
#             x = x.replace("Data", "crop")
#             x = os.path.join(opt.base_dir, x)
#             if not os.path.exists(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop"))):
#                 if not os.path.exists(x[:len(x) - len(x.split("/")[-1])]):
#                     os.makedirs(x[:len(x) - len(x.split("/")[-1])])
#                 try:
#                     im = cv2.imread(im_p)
#                     (h, w) = im.shape[:2]
#                     blob = cv2.dnn.blobFromImage(
#                         im, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0))
#                     net.setInput(blob)
#                     detections = net.forward()
#                     if len(detections) > 0:
#                         count = 0
#                         for i in range(detections.shape[2]):
#                             confidence = detections[0, 0, i, 2]
#                             if confidence >= 0.5:
#                                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                                 (startX, startY, endX, endY) = box.astype("int")
#                                 if startX <= w and endX <= w and startY <= h and endY <= h:
#                                     count += 1
#                                     face = im[startY:endY, startX:endX]
#                                     cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_{}.{}".format(partitions[0], count, partitions[1]).replace("Data", "crop")), face)
#                         if count == 0:
#                             cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
#                 except Exception as e:
#                     try:
#                         cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
#                     except Exception as e1:
#                         check_path_exist(opt.make_crop_our_broken_images)
#                         with open(os.path.join(opt.make_crop_our_broken_images, "broken.txt"), "a") as file_object:
#                             file_object.write(im_p + "\n")
#                         if "train" in im_p:
#                             if "spoof" in im_p:
#                                 im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "spoof.jpg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
#                             else:
#                                 im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "live.jpeg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
#                         else:
#                             if "spoof" in im_p:
#                                 im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "spoof.jpg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
#                             else:
#                                 im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "live.jpeg"))
#                                 cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)

def make_crop_each_insight(filelists):
    json_dict = json.load(open(filelists[1]))
    temp_dict = {}
    for k in tqdm(json_dict.keys()):
        im_p = os.path.join(filelists[0], k)
        if os.path.exists(im_p):
            partitions = k.split(".")
            x = partitions[0]+"_crop." + partitions[1]
            x = x.replace("Data", "photo_crop")
            x = os.path.join(opt.base_dir, x)
            if not os.path.exists(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "photo_crop"))):
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
                            cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_{}.{}".format(partitions[0], count, partitions[1]).replace("Data", "photo_crop")), crop_face)
                    if count > 0:
                        temp_dict[k] = json_dict[k]

                    if count == 0:
                        # cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
                        pass
                except Exception as e:
                    # print("IF YOU SEE THIS LINE LIKE 1000 TIMES OR ABOVE, PLEASE STOP THE CODE. THERE MIGHT BE SOME PROBLEMS WITH GPU MEMORY, PLEASE CHECK THOSE")
                    print(r.repr(str(e)))
                    check_path_exist(opt.make_crop_our_broken_images)
                    with open(os.path.join(opt.make_crop_our_broken_images, "broken.txt"), "a") as file_object:
                        file_object.write(im_p + "\n")
                    pass
                    # try:
                    #     cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im)
                    # except Exception as e1:
                    #     print("IF YOU SEE THIS LINE LIKE 1000 TIMES OR ABOVE, PLEASE STOP THE CODE. THERE MIGHT BE SOME PROBLEMS WITH GPU MEMORY, PLEASE CHECK THOSE")
                    #     print(r.repr(str(e1)))
                    #     check_path_exist(opt.make_crop_our_broken_images)
                    #     with open(os.path.join(opt.make_crop_our_broken_images, "broken.txt"), "a") as file_object:
                    #         file_object.write(im_p + "\n")
                    #     if "train" in im_p:
                    #         if "spoof" in im_p:
                    #             im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "spoof.jpg"))
                    #             cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
                    #         else:
                    #             im_temp = cv2.imread(os.path.join(opt.our_train_temp_images, "live.jpeg"))
                    #             cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
                    #     else:
                    #         if "spoof" in im_p:
                    #             im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "spoof.jpg"))
                    #             cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)
                    #         else:
                    #             im_temp = cv2.imread(os.path.join(opt.our_test_temp_images, "live.jpeg"))
                    #             cv2.imwrite(os.path.join(opt.base_dir, "{}_crop_1.{}".format(partitions[0], partitions[1]).replace("Data", "crop")), im_temp)

    if temp_dict:
        with open(filelists[1], 'w') as outfile:
            json.dump(temp_dict, outfile)
        
    

def make_crop(filelists, data_filelists):
    # celeb
    print("Crop CelebA-Spoof data!!!")
    make_crop_each_insight(filelists)
    # our data
    print("Crop our data!!!")
    make_crop_each_insight(data_filelists)

if __name__ == "__main__":

    if type_dir != "train" and type_dir != "test":
        print("Wrong type dir")
    # else:
    #     YOLO_IMG_WIDTH = opt.YOLO_IMG_WIDTH
    #     YOLO_IMG_HEIGHT = opt.YOLO_IMG_HEIGHT
    #     CONFIDENCE_THRESHOLD = 0.2
    #     NMS_THRESHOLD = 0.4
    #     PADDING = 20
    #     net = cv2.dnn.readNet(opt.yolo_weights_path, opt.yolo_config_path)
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    #     model = cv2.dnn_DetectionModel(net)
    #     model.setInputParams(size=(YOLO_IMG_WIDTH, YOLO_IMG_HEIGHT), scale=1/255, swapRB=True)
    #     if type_dir == "train":
    #         make_crop(opt.celeb_train_filelists, opt.data_train_filelists)
    #     elif type_dir == "test":
    #         make_crop(opt.celeb_test_filelists, opt.data_test_filelists)
    else:
        thresh = 0.8
        gpuid = 0
        detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
        PADDING = 20
        if type_dir == "train":
            make_crop(opt.celeb_train_filelists, opt.data_train_filelists)
        elif type_dir == "test":
            make_crop(opt.celeb_test_filelists, opt.data_test_filelists)