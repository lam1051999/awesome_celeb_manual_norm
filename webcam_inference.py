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

def webcam_inference():

    # load crop model
    IMG_WIDTH = opt.IMG_WIDTH
    IMG_HEIGHT = opt.IMG_HEIGHT
    protoPath = opt.protoPath
    modelPath = opt.modelPath
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    BASE_HEIGHT = 800

    # initialize webcam
    cap = cv2.VideoCapture(0)
    color = (255, 0, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7


    # load model
    pths = glob.glob('checkpoints-photo-celeb/%s/*.pth' % (opt.model))
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
    while(True):
        ret, im = cap.read()    
        if not ret:
            print('==> Done processing!!!')
            cv2.waitKey(1000)
            break
        state = im.shape[0] > BASE_HEIGHT
        if state:
            down_scale = math.ceil(im.shape[0] / BASE_HEIGHT)
            im = cv2.resize(im, (im.shape[1]//down_scale, im.shape[0]//down_scale),
                            interpolation=cv2.INTER_AREA)
        (h, w) = im.shape[:2]
        blob = cv2.dnn.blobFromImage(
            im, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        if len(detections) > 0:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence >= 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = im[startY:endY, startX:endX]
                    if 0 not in face.shape:
                        im = cv2.rectangle(im, (startX, startY), (endX, endY), color, thickness)
                        face = cv2.resize(face, (opt.image_size, opt.image_size))
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
        
                else:
                    continue
                
        if state:
            im = cv2.resize(im, (int(im.shape[1]*down_scale), int(im.shape[0]*down_scale)),
                            interpolation=cv2.INTER_AREA)
        cv2.imshow('frame',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import fire
    fire.Fire()
