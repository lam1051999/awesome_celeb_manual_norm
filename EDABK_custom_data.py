# Necessary packages
import os
import json

from EDABK_utils import get_path, extract_sub_images, check_path_exist
from config import opt

# get json file for small amount of images we get
def main():
    # Paths
    data_path = os.path.join(opt.print_root, 'Data')
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')


    imgs_train = get_path(train_path, mode='train')
    imgs_test = get_path(test_path, mode='test')

    # Metas path
    metas = os.path.join(opt.root, 'metas')
    intra_test_path = os.path.join(metas, 'intra_test')
    #Loading and processing intra_test json
    train_json = json.load(open(os.path.join(intra_test_path, 'train_label.json')))
    test_json = json.load(open(os.path.join(intra_test_path, 'test_label.json')))


    train_dict = extract_sub_images(train_json, imgs_train)
    test_dict = extract_sub_images(test_json, imgs_test)
    
    check_path_exist(opt.intra_test_photo_temp)

    with open(os.path.join(opt.intra_test_photo_temp, 'train_label.json'), 'w') as outfile:
        json.dump(train_dict, outfile)

    
    with open(os.path.join(opt.intra_test_photo_temp, 'test_label.json'), 'w') as outfile:
        json.dump(test_dict, outfile)


if __name__ == "__main__":
    main()