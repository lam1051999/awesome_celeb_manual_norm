import os
import json
import random

from config import opt
from EDABK_utils import check_path_exist, move_files

def main() -> None:
    check_path_exist(opt.print_root)
    train_json = json.load(open(os.path.join(opt.root, "metas/intra_test", 'train_label.json')))
    test_json = json.load(open(os.path.join(opt.root, "metas/intra_test", 'test_label.json')))

    photo_train = [key for (key, value) in train_json.items() if value[40] == 1]
    photo_test = [key for (key, value) in test_json.items() if value[40] == 1]

    poster_train = [key for (key, value) in train_json.items() if value[40] == 2]
    poster_test = [key for (key, value) in test_json.items() if value[40] == 2]

    a4_train = [key for (key, value) in train_json.items() if value[40] == 3]
    a4_test = [key for (key, value) in test_json.items() if value[40] == 3]

    if opt.GET_ALL_CELEB_PHOTO:
        move_files(opt.root, opt.print_root, photo_train, "photo train")
        move_files(opt.root, opt.print_root, photo_test, "photo test")

        move_files(opt.root, opt.print_root, poster_train, "poster train")
        move_files(opt.root, opt.print_root, poster_test, "poster test")

        move_files(opt.root, opt.print_root, a4_train, "A4 train")
        move_files(opt.root, opt.print_root, a4_test, "A4 test")
    else:
        random.shuffle(photo_train)
        random.shuffle(photo_test)
        random.shuffle(poster_train)
        random.shuffle(poster_test)
        random.shuffle(a4_train)
        random.shuffle(a4_test)

        move_files(opt.root, opt.print_root, photo_train[:opt.NUMBER_OF_PHOTO_TRAIN], "photo train")
        move_files(opt.root, opt.print_root, photo_test[:opt.NUMBER_OF_PHOTO_TEST], "photo test")

        move_files(opt.root, opt.print_root, poster_train[:opt.NUMBER_OF_POSTER_TRAIN], "poster train")
        move_files(opt.root, opt.print_root, poster_test[:opt.NUMBER_OF_POSTER_TEST], "poster test")

        move_files(opt.root, opt.print_root, a4_train[:opt.NUMBER_OF_A4_TRAIN], "A4 train")
        move_files(opt.root, opt.print_root, a4_test[:opt.NUMBER_OF_A4_TEST], "A4 test")


if __name__ == "__main__":
    main()