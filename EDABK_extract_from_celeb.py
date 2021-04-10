import os
import json

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

    move_files(opt.root, opt.print_root, photo_train, "photo train")
    move_files(opt.root, opt.print_root, photo_test, "photo test")

    move_files(opt.root, opt.print_root, poster_train, "poster train")
    move_files(opt.root, opt.print_root, poster_test, "poster test")

    move_files(opt.root, opt.print_root, a4_train, "A4 train")
    move_files(opt.root, opt.print_root, a4_test, "A4 test")

if __name__ == "__main__":
    main()