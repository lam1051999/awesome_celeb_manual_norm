# coding: utf8
import torch
import warnings
import os

class DefaultConfig(object):
    model = 'MyresNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    env = model
    ATTACK = 1
    GENUINE = 0

    NUMBER_OF_LIVE_SAMPLES_TRAIN = 15000
    NUMBER_OF_LIVE_SAMPLES_TEST = 5000 
    root = "/home/tranlam/Downloads/celeb/CelebA_Spoof/"
    base_dir = "/media/tranlam/Data_Storage/AI/Liveness/complete/"
    print_root = os.path.join(base_dir, "photo_celeb/CelebA_Spoof/") 
    intra_test_temp = os.path.join(base_dir, "metas/intra_test/")
    our_data = os.path.join(base_dir, "our_data/")
    our_label = os.path.join(our_data, "label/")

    make_crop_broken_images = os.path.join(base_dir, "awesome_celeb/broken_images/make_crop_image/")
    rqds_crop_broken_images_train = os.path.join(base_dir, "awesome_celeb/broken_images/require_dataset_crop/")
    train_temp_images = os.path.join(base_dir, "awesome_celeb/train_temp/")
    test_temp_images = os.path.join(base_dir, "awesome_celeb/test_temp/")

    make_crop_our_broken_images = os.path.join(base_dir, "awesome_celeb/our_broken_images/make_crop_image/")
    rqds_crop_our_broken_images_train = os.path.join(base_dir, "awesome_celeb/our_broken_images/require_dataset_crop/")
    our_train_temp_images = os.path.join(base_dir, "awesome_celeb/our_train_temp/")
    our_test_temp_images = os.path.join(base_dir, "awesome_celeb/our_test_temp/")

    celeb_train_filelists = [
        print_root, os.path.join(intra_test_temp, "train_label.json")
    ]
    celeb_test_filelists = [
        print_root, os.path.join(intra_test_temp, "test_label.json")
    ]

    data_train_filelists = [
        our_data, os.path.join(our_label, "train.json")
    ]
    data_test_filelists = [
        our_data, os.path.join(our_label, "test.json")
    ]

    protoPath = os.path.join(base_dir, "awesome_celeb/crop_model/deploy.prototxt") 
    modelPath = os.path.join(base_dir, "awesome_celeb/crop_model/res10_300x300_ssd_iter_140000.caffemodel")
    IMG_WIDTH = 300
    IMG_HEIGHT = 300
    # load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 16  # batch size
    # use_gpu = torch.cuda.is_available()  # use GPU or not
    use_gpu = False  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_name = 'result'

    max_epoch = 20
    lr = 0.01  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_stepsize = 3  # learning step size
    weight_decay = 1e-5  # 损失函数
    cropscale = 3.5
    image_size = 224


def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
