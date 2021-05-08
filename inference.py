from config import opt
import os
import models
from tqdm import tqdm
import torch
from torchsummary import summary
import numpy as np
import cv2
import glob
import math
import sys

from retinaface import RetinaFace


# def inference(**kwargs):
#     path = kwargs["image"]

#     # load crop model
#     IMG_WIDTH = opt.IMG_WIDTH
#     IMG_HEIGHT = opt.IMG_HEIGHT
#     protoPath = opt.protoPath
#     modelPath = opt.modelPath
#     net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#     color = (255, 0, 0)
#     thickness = 2
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1
#     # load model
#     pths = glob.glob('checkpoints/%s/*.pth' % (opt.model))
#     pths.sort(key=os.path.getmtime, reverse=True)
#     print(pths)
#     # 模型
#     opt.load_model_path = pths[0]
#     model = getattr(models, opt.model)().eval()
#     assert os.path.exists(opt.load_model_path)
#     if opt.load_model_path:
#         model.load(opt.load_model_path)
#     if opt.use_gpu:
#         model.cuda()
#     model.train(False)
#     fopen = open('result/inference.txt', 'w')
#     im = cv2.imread(path)
#     (h, w) = im.shape[:2]
#     blob = cv2.dnn.blobFromImage(
#         im, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     if len(detections) > 0:
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             if confidence >= 0.6:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 if startX <= w and endX <= w and startY <= h and endY <= h:
#                     face = im[startY:endY, startX:endX]
#                     if 0 not in face.shape:
#                         face = cv2.resize(face, (224, 224))
#                         face = np.transpose(np.array(face, dtype=np.float32), (2, 0, 1))
#                         face = face[np.newaxis, :]
#                         face = torch.FloatTensor(face)
#                         with torch.no_grad():
#                             if opt.use_gpu:
#                                 face = face.cuda()
#                             outputs = model(face)
#                             outputs = torch.softmax(outputs, dim=-1)
#                             preds = outputs.to('cpu').numpy()
#                             attack_prob = preds[:, opt.ATTACK]
#                             im = cv2.putText(im, "Spoof {:.2f}".format(sum(attack_prob)), (startX - 5, startY - 5), font, fontScale, color, thickness, cv2.LINE_AA)
#                             im = cv2.rectangle(im, (startX, startY), (endX, endY), color, thickness)
#                             cv2.imwrite(path.split(".")[0]+"_evaluated." + path.split(".")[1], im)
#                             print('Inference %s attack_prob=%f' % (path, attack_prob), file=fopen)

#     fopen.close()

# def inference(**kwargs):
#     images = kwargs["images"]
#     crop_threshold = kwargs["crop_threshold"]
#     spoof_threshold = kwargs["spoof_threshold"]

#     YOLO_IMG_WIDTH = opt.YOLO_IMG_WIDTH
#     YOLO_IMG_HEIGHT = opt.YOLO_IMG_HEIGHT
#     CONFIDENCE_THRESHOLD = 0.2
#     NMS_THRESHOLD = 0.4
#     PADDING = 20
#     net = cv2.dnn.readNet(opt.yolo_weights_path, opt.yolo_config_path)
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#     model_yolo = cv2.dnn_DetectionModel(net)
#     model_yolo.setInputParams(size=(YOLO_IMG_WIDTH, YOLO_IMG_HEIGHT), scale=1/255, swapRB=True)
  
#     # load crop model
    
#     color = (0, 0, 255)
#     thickness = 3
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 3
#     # load model
#     pths = glob.glob('checkpoints-photo-celeb/%s/*.pth' % (opt.model))
#     pths.sort(key=os.path.getmtime, reverse=True)
#     print(pths)

#     opt.load_model_path = pths[0]
#     model = getattr(models, opt.model)().eval()
#     assert os.path.exists(opt.load_model_path)
#     if opt.load_model_path:
#         model.load(opt.load_model_path)
#     if opt.use_gpu:
#         model.cuda()
#     model.train(False)

#     count = 0
#     for image in os.listdir(images):
#         if image.split(".")[-1] == "jpg" or image.split(".")[-1] == "png" or image.split(".")[-1] == "jpeg" or image.split(".")[-1] == "JPG" or image.split(".")[-1] == "PNG" or image.split(".")[-1] == "JPEG":
#             path = os.path.join(images, image)
#             im = cv2.imread(path)
#             (h, w) = im.shape[:2]
#             if h < 800 or w < 800:
#                 fontScale = 1
#                 thickness = 1
#             classes, scores, boxes = model_yolo.detect(im, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

#             for (classid, score, box) in zip(classes, scores, boxes):
#                 if score >= float(crop_threshold):
#                     (startX, startY, endX, endY) = (box[0], box[1], box[0] + box[2], box[1] + box[3])
#                     if startX <= w and endX <= w and startY <= h and endY <= h:
#                         (startX, startY, endX, endY) = (startX - PADDING, startY - PADDING, endX + PADDING, endY + PADDING)
#                         if startX < 0:
#                             startX = 0
#                         if startY < 0:
#                             startY = 0
#                         face = im[startY:endY, startX:endX]

#                         if 0 not in face.shape:
#                             face = cv2.resize(face, (opt.image_size, opt.image_size))
#                             face = face/255
#                             face = np.transpose(np.array(face, dtype=np.float32), (2, 0, 1))
#                             face = face[np.newaxis, :]
#                             face = torch.FloatTensor(face)
#                             with torch.no_grad():
#                                 if opt.use_gpu:
#                                     face = face.cuda()
#                                 outputs = model(face)
#                                 outputs = torch.softmax(outputs, dim=-1)
#                                 preds = outputs.to('cpu').numpy()
#                                 attack_prob = preds[:, opt.ATTACK]
#                                 if sum(attack_prob) >= float(spoof_threshold):
#                                     count += 1
#                                 im = cv2.putText(im, "Spoof {:.2f}".format(sum(attack_prob)), (startX - 5 if startX - 5 > 0 else startX + 5, startY - 5 if startY - 5 > 0 else startY + 5), font, fontScale, color, thickness, cv2.LINE_AA)
#                                 im = cv2.rectangle(im, (startX, startY), (endX, endY), color, thickness)
#                                 cv2.imwrite(path.split(".")[0]+"_evaluated." + path.split(".")[1], im)

#     print("Number of spoof faces in the images in {} is: {}".format(images, count))

def inference(**kwargs):
    images = kwargs["images"]
    spoof_threshold = kwargs["spoof_threshold"]

    # load crop model

    thresh = 0.8
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    PADDING = 10
    
    color = (0, 0, 255)
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
    # load model
    pths = glob.glob('checkpoints-photo-celeb/%s/*.pth' % (opt.model))
    pths.sort(key=os.path.getmtime, reverse=True)
    print(pths)

    opt.load_model_path = pths[0]
    model = getattr(models, opt.model)().eval()
    assert os.path.exists(opt.load_model_path)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    model.train(False)

    count = 0
    for image in os.listdir(images):
        if image.split(".")[-1] == "jpg" or image.split(".")[-1] == "png" or image.split(".")[-1] == "jpeg" or image.split(".")[-1] == "JPG" or image.split(".")[-1] == "PNG" or image.split(".")[-1] == "JPEG":
            path = os.path.join(images, image)
            img = cv2.imread(path)

            scales = [1024, 1980]
            im_shape = img.shape
            target_size = scales[0]
            max_size = scales[1]
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            scales = [im_scale]
            flip = False
            faces, landmarks = detector.detect(img,
                                            thresh,
                                            scales=scales,
                                            do_flip=flip)


            if img.shape[0] < 800 or img.shape[1] < 800:
                fontScale = 1
                thickness = 1

            if faces is not None:
                for i in range(faces.shape[0]):
                    box = faces[i].astype(np.int)
                    (startX, startY, endX, endY) = (box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10)
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(endX, im_shape[1])
                    endY = min(endY, im_shape[0])
                    crop_face = img[startY:endY, startX:endX]

                    if 0 not in crop_face.shape:
                        crop_face = cv2.resize(crop_face, (opt.image_size, opt.image_size))
                        crop_face = crop_face/255
                        crop_face = np.transpose(np.array(crop_face, dtype=np.float32), (2, 0, 1))
                        crop_face = crop_face[np.newaxis, :]
                        crop_face = torch.FloatTensor(crop_face)
                        with torch.no_grad():
                            if opt.use_gpu:
                                crop_face = crop_face.cuda()
                            outputs = model(crop_face)
                            outputs = torch.softmax(outputs, dim=-1)
                            preds = outputs.to('cpu').numpy()
                            attack_prob = preds[:, opt.ATTACK]
                            if sum(attack_prob) >= float(spoof_threshold):
                                count += 1
                            img = cv2.putText(img, "Spoof {:.2f}".format(sum(attack_prob)), (startX - 5 if startX - 5 > 0 else startX + 5, startY - 5 if startY - 5 > 0 else startY + 5), font, fontScale, color, thickness, cv2.LINE_AA)
                            img = cv2.rectangle(img, (startX, startY), (endX, endY), color, thickness)
                            cv2.imwrite(path.split(".")[0]+"_evaluated." + path.split(".")[1], img)

    print("Number of spoof faces in the images in {} is: {}".format(images, count))


if __name__ == '__main__':
    import fire
    fire.Fire()
