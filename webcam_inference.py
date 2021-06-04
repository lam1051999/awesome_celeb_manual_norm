from config import opt
import os
import models
import torch
import numpy as np
import cv2
import glob

from retinaface import RetinaFace

# insightface detector
def webcam_inference():

    # load crop model
    thresh = 0.8
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    PADDING = 10   

    # initialize webcam
    cap = cv2.VideoCapture(0)
    color = (255, 0, 0)
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
        ret, img = cap.read()    
        if not ret:
            print('==> Done processing!!!')
            cv2.waitKey(1000)
            break

        # prepare input for face detector
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
        
        # detect faces in a frame
        faces, landmarks = detector.detect(img,
                                        thresh,
                                        scales=scales,
                                        do_flip=flip)
        
        # get each face
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
                    img = cv2.rectangle(img, (startX, startY), (endX, endY), color, thickness)
                    crop_face = cv2.resize(crop_face, (opt.image_size, opt.image_size))
                    crop_face = crop_face/255
                    crop_face = np.transpose(np.array(crop_face, dtype=np.float32), (2, 0, 1))
                    crop_face = crop_face[np.newaxis, :]
                    crop_face = torch.FloatTensor(crop_face)
                    
                    # get prediction
                    with torch.no_grad():
                        if opt.use_gpu:
                            crop_face = crop_face.cuda()
                        outputs = model(crop_face)
                        outputs = torch.softmax(outputs, dim=-1)
                        preds = outputs.to('cpu').numpy()
                        attack_prob = preds[:, opt.ATTACK]
                        img = cv2.putText(img, "Spoof {:.2f}".format(sum(attack_prob)), (startX - 5, startY - 5), font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    import fire
    fire.Fire()
