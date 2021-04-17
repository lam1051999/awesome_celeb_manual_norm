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

def inference(**kwargs):
    path = kwargs["image"]

    # load crop model
    IMG_WIDTH = opt.IMG_WIDTH
    IMG_HEIGHT = opt.IMG_HEIGHT
    protoPath = opt.protoPath
    modelPath = opt.modelPath
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    color = (255, 0, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    # load model
    pths = glob.glob('checkpoints/%s/*.pth' % (opt.model))
    pths.sort(key=os.path.getmtime, reverse=True)
    print(pths)
    # 模型
    opt.load_model_path = pths[0]
    model = getattr(models, opt.model)().eval()
    assert os.path.exists(opt.load_model_path)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    model.train(False)
    fopen = open('result/inference.txt', 'w')
    im = cv2.imread(path)
    (h, w) = im.shape[:2]
    blob = cv2.dnn.blobFromImage(
        im, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    if len(detections) > 0:
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startX <= w and endX <= w and startY <= h and endY <= h:
                    face = im[startY:endY, startX:endX]
                    if 0 not in face.shape:
                        face = cv2.resize(face, (224, 224))
                        face = np.transpose(np.array(face, dtype=np.float32), (2, 0, 1))
                        face = face[np.newaxis, :]
                        face = torch.FloatTensor(face)
                        with torch.no_grad():
                            if opt.use_gpu:
                                face = face.cuda()
                            outputs = model(face)
                            outputs = torch.softmax(outputs, dim=-1)
                            preds = outputs.to('cpu').numpy()
                            attack_prob = preds[:, opt.ATTACK]
                            im = cv2.putText(im, "Spoof {:.2f}".format(sum(attack_prob)), (startX - 5, startY - 5), font, fontScale, color, thickness, cv2.LINE_AA)
                            im = cv2.rectangle(im, (startX, startY), (endX, endY), color, thickness)
                            cv2.imwrite(path.split(".")[0]+"_evaluated." + path.split(".")[1], im)
                            print('Inference %s attack_prob=%f' % (path, attack_prob), file=fopen)

    fopen.close()

if __name__ == '__main__':
    import fire
    fire.Fire()
