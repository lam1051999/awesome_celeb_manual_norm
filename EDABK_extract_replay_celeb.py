import os
import json
import random

from config import opt
from EDABK_utils import check_path_exist, move_files

def main() -> None:
    check_path_exist(opt.replay_root)
    train_json = json.load(open(os.path.join(opt.root, "metas/intra_test", 'train_label.json')))
    test_json = json.load(open(os.path.join(opt.root, "metas/intra_test", 'test_label.json')))

    PC_train = [key for (key, value) in train_json.items() if value[40] == 7]
    PC_test = [key for (key, value) in test_json.items() if value[40] == 7]

    Pad_train = [key for (key, value) in train_json.items() if value[40] == 8]
    Pad_test = [key for (key, value) in test_json.items() if value[40] == 8]

    Phone_train = [key for (key, value) in train_json.items() if value[40] == 9]
    Phone_test = [key for (key, value) in test_json.items() if value[40] == 9]

    if opt.GET_ALL_CELEB_REPLAY:
        move_files(opt.root, opt.replay_root, PC_train, "PC train")
        move_files(opt.root, opt.replay_root, PC_test, "PC test")

        move_files(opt.root, opt.replay_root, Pad_train, "Pad train")
        move_files(opt.root, opt.replay_root, Pad_test, "Pad test")

        move_files(opt.root, opt.replay_root, Phone_train, "Phone train")
        move_files(opt.root, opt.replay_root, Phone_test, "Phone test")
    else:
        random.shuffle(PC_train)
        random.shuffle(PC_test)
        random.shuffle(Pad_train)
        random.shuffle(Pad_test)
        random.shuffle(Phone_train)
        random.shuffle(Phone_test)

        move_files(opt.root, opt.replay_root, PC_train[:opt.NUMBER_OF_PC_TRAIN], "PC train")
        move_files(opt.root, opt.replay_root, PC_test[:opt.NUMBER_OF_PC_TEST], "PC test")

        move_files(opt.root, opt.replay_root, Pad_train[:opt.NUMBER_OF_PAD_TRAIN], "Pad train")
        move_files(opt.root, opt.replay_root, Pad_test[:opt.NUMBER_OF_PAD_TEST], "Pad test")

        move_files(opt.root, opt.replay_root, Phone_train[:opt.NUMBER_OF_PHONE_TRAIN], "Phone train")
        move_files(opt.root, opt.replay_root, Phone_test[:opt.NUMBER_OF_PHONE_TEST], "Phone test")


if __name__ == "__main__":
    main()