from config import opt
import os
import models
import torch
import numpy as np
import cv2
import glob

def webcam_inference_yoloface(**kwargs):
    spoof_threshold = kwargs["spoof_threshold"]
    crop_threshold = kwargs["crop_threshold"]

    # load crop model
    YOLO_IMG_WIDTH = opt.YOLO_IMG_WIDTH
    YOLO_IMG_HEIGHT = opt.YOLO_IMG_HEIGHT
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    PADDING = 20
    net = cv2.dnn.readNet(opt.yolo_weights_path, opt.yolo_config_path)
    model_yolo = cv2.dnn_DetectionModel(net)
    model_yolo.setInputParams(size=(YOLO_IMG_WIDTH, YOLO_IMG_HEIGHT), scale=1/255, swapRB=True)

    # initialize webcam
    cap = cv2.VideoCapture(0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1


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
    while(True):
        ret, im = cap.read()    
        if not ret:
            print('==> Done processing!!!')
            cv2.waitKey(1000)
            break
        (h, w) = im.shape[:2]
        classes, scores, boxes = model_yolo.detect(im, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for (classid, score, box) in zip(classes, scores, boxes):
            if score >= float(crop_threshold):
                (startX, startY, endX, endY) = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                if startX <= w and endX <= w and startY <= h and endY <= h:
                    (startX, startY, endX, endY) = (startX - PADDING, startY - PADDING, endX + PADDING, endY + PADDING)
                    if startX < 0:
                        startX = 0
                    if startY < 0:
                        startY = 0
                    face = im[startY:endY, startX:endX]
                    if 0 not in face.shape:
                        face = cv2.resize(face, (opt.image_size, opt.image_size))
                        face = face/255
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
                            im = cv2.putText(im, "Spoof {:.5f}".format(sum(attack_prob)), (startX - 5, startY - 5), font, fontScale, (0, 255, 0) if attack_prob < float(spoof_threshold) else (0, 0, 255), thickness, cv2.LINE_AA)
                            im = cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0) if attack_prob < float(spoof_threshold) else (0, 0, 255), thickness)

        cv2.imshow('frame',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import fire
    fire.Fire()