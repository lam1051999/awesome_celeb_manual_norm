from config import opt
import os
import models
import torch
import numpy as np
import cv2
import glob

from retinaface import RetinaFace

def shot(image_path):
    # change threshold here
    spoof_threshold = 0.056

    # load crop model
    thresh = 0.8
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    PADDING = 10


    # load model
    pths = glob.glob('checkpoints/%s/*.pth' % (opt.model))
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

    # read image
    img = cv2.imread(image_path)

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

    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            (startX, startY, endX, endY) = (box[0] - PADDING, box[1] - PADDING, box[2] + PADDING, box[3] + PADDING)
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

                    # if there is a spoof face in the image
                    if sum(attack_prob) >= float(spoof_threshold):
                        print("Attack probability: {}".format(sum(attack_prob)))
                        return True

    # if there is no spoof face in the image
    return False

if __name__ == '__main__':

    # change image path here
    image_path = "test.jpg"
    shot(image_path)
