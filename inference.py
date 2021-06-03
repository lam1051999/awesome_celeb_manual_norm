from PIL import Image
from config import opt
import os
import models
import torch
import numpy as np
import cv2
import glob
from torchvision import transforms
from EDABK_utils import check_path_exist

from retinaface import RetinaFace

data_transforms = {
	'train' : transforms.Compose([
		#transforms.RandomRotation((45)),
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomVerticalFlip(),
		#transforms.Lambda(maxcrop),
		#transforms.Lambda(blur),
		transforms.Resize((224,224)) ,
	   	transforms.ToTensor(),
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
	]) ,
    'train_aug': transforms.Compose([
		#transforms.RandomRotation((45)),
		# transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		#transforms.Lambda(maxcrop),
		#transforms.Lambda(blur),
		transforms.Resize((224,224)) ,
	   	transforms.ToTensor(),
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ]),
	'val' : transforms.Compose([
		#transforms.Lambda(maxcrop),
		transforms.Resize((224,224)),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
	]),
	'test' : transforms.Compose([
		#transforms.Lambda(maxcrop),
		transforms.Resize((224,224)) ,
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
	]) ,}


def inference(**kwargs):
    images = kwargs["images"]
    spoof_threshold = kwargs["spoof_threshold"]
    output_images = kwargs["output_images"]
    check_path_exist(images)
    check_path_exist(output_images)

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

    count = 0
    for image in os.listdir(images):
        if image.split(".")[-1] == "jpg" or image.split(".")[-1] == "png" or image.split(".")[-1] == "jpeg" or image.split(".")[-1] == "JPG" or image.split(".")[-1] == "PNG" or image.split(".")[-1] == "JPEG":
            path = os.path.join(images, image)
            output_path = os.path.join(output_images, image)
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
                    (startX, startY, endX, endY) = (box[0] - PADDING, box[1] - PADDING, box[2] + PADDING, box[3] + PADDING)
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(endX, im_shape[1])
                    endY = min(endY, im_shape[0])
                    crop_face = img[startY:endY, startX:endX]

                    if 0 not in crop_face.shape:
                        crop_face = Image.fromarray(crop_face)
                        crop_face = data_transforms["test"](crop_face)
                        crop_face = torch.unsqueeze(crop_face, 0)

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
                            cv2.imwrite(output_path, img)

    print("Number of spoof faces in the images in {} is: {}".format(images, count))


if __name__ == '__main__':
    import fire
    fire.Fire()
