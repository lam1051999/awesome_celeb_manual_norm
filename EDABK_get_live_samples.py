import os
import json

from config import opt
from EDABK_utils import check_path_exist, move_files

def main() -> None:
    check_path_exist(opt.print_root)
    train_json = json.load(open(os.path.join(opt.root, "metas/intra_test", 'train_label.json')))
    test_json = json.load(open(os.path.join(opt.root, "metas/intra_test", 'test_label.json')))

    live_train = [key for (key, value) in train_json.items() if value[-1] == 0]
    live_test = [key for (key, value) in test_json.items() if value[-1] == 0]

    move_files(opt.root, opt.print_root, live_train[:opt.NUMBER_OF_LIVE_SAMPLES_TRAIN], "live train")
    move_files(opt.root, opt.print_root, live_test[:opt.NUMBER_OF_LIVE_SAMPLES_TEST], "live test")

if __name__ == "__main__":
    main()