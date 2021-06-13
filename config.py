# coding: utf8
import torch
import warnings
import os

class DefaultConfig(object):
    model = 'MyresNet34'
    env = model
    ATTACK = 1
    GENUINE = 0

    NUMBER_OF_LIVE_SAMPLES_TRAIN = 3000
    NUMBER_OF_LIVE_SAMPLES_TEST = 1000

    GET_ALL_CELEB_PHOTO = False
    NUMBER_OF_PHOTO_TRAIN = 1500
    NUMBER_OF_PHOTO_TEST = 500
    NUMBER_OF_POSTER_TRAIN = 1500
    NUMBER_OF_POSTER_TEST = 500
    NUMBER_OF_A4_TRAIN = 1500
    NUMBER_OF_A4_TEST = 500

    # CelebA-Spoof root folder
    root = "/home/tranlam/Downloads/celeb/CelebA_Spoof/"
    # our working directory
    base_dir = "/media/tranlam/Data_Storage/AI/Liveness/complete/"

    # CelebA-Spoof photo images root folder
    print_root = os.path.join(base_dir, "photo_celeb/CelebA_Spoof/") 

    intra_test_photo_temp = os.path.join(print_root, "metas/intra_test/")
    
    our_data = os.path.join(base_dir, "our_data/")
    our_label = os.path.join(our_data, "label/")

    make_crop_our_broken_images = os.path.join(base_dir, "awesome_celeb_manual_norm/our_broken_images/make_crop_image/")
    rqds_crop_our_broken_images_train = os.path.join(base_dir, "awesome_celeb_manual_norm/our_broken_images/require_dataset_crop/")
    our_train_temp_images = os.path.join(base_dir, "awesome_celeb_manual_norm/our_train_temp/")
    our_test_temp_images = os.path.join(base_dir, "awesome_celeb_manual_norm/our_test_temp/")

    celeb_train_filelists = [
        print_root, os.path.join(intra_test_photo_temp, "train_label.json")
    ]
    celeb_test_filelists = [
        print_root, os.path.join(intra_test_photo_temp, "test_label.json")
    ]

    data_train_filelists = [
        our_data, os.path.join(our_label, "train.json")
    ]
    data_test_filelists = [
        our_data, os.path.join(our_label, "test.json")
    ]

    # you can use resnet10 or yolo-face detector if you want the face detector run faster
    # here are the components to embedd in dnn module of open-cv
    protoPath = os.path.join(base_dir, "awesome_celeb_manual_norm/crop_model/deploy.prototxt") 
    modelPath = os.path.join(base_dir, "awesome_celeb_manual_norm/crop_model/res10_300x300_ssd_iter_140000.caffemodel")
    yolo_config_path = os.path.join(base_dir, "awesome_celeb_manual_norm/crop_model/yolo-face-500k.cfg")
    yolo_weights_path = os.path.join(base_dir, "awesome_celeb_manual_norm/crop_model/yolo-face-500k.weights")
    IMG_WIDTH = 300
    IMG_HEIGHT = 300
    YOLO_IMG_WIDTH = 320
    YOLO_IMG_HEIGHT = 320

    load_model_path = None 

    batch_size = 16  # batch size
    use_gpu = torch.cuda.is_available()  # use GPU or not
    # use_gpu = False  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_name = 'result'

    max_epoch = 20
    lr = 0.01  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_stepsize = 3  # learning step size
    weight_decay = 1e-5 
    cropscale = 3.5
    image_size = 224


def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
