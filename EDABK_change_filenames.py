import os
import sys
from config import opt
from tqdm import tqdm
from EDABK_utils import check_path_exist, shuffle_dictionary
import json

# change F19 image filenames and create label json file
def main() -> None:
    type_dir = sys.argv[1]
    if type_dir != "train" and type_dir != "test":
        print("Wrong type dir")
    else:
        check_path_exist(opt.our_label)

        # image attributes
        spoof_type = 3
        environment = 1

        live_label = 0
        spoof_label = 1
        data = os.path.join(opt.our_data, "Data", type_dir)
        type_label = {}

        for A_x in os.listdir(data):
            A_dir = os.path.join(data, A_x)
            for label_x in os.listdir(A_dir):
                label_dir = os.path.join(A_dir, label_x)
                if label_x == "live":
                    image_paths = os.listdir(label_dir)
                    image_paths.sort()
                    for i, image_path in tqdm(enumerate(image_paths)):
                        new_name = "{}{}{:06d}.{}".format(A_x[1:], live_label if label_x == "live" else spoof_label, i, image_path.split(".")[-1])
                        temp = os.path.join("Data", type_dir, A_x, label_x, new_name)
                        new_name = os.path.join(label_dir, new_name)
                        if not os.path.exists(new_name):
                            os.rename(os.path.join(label_dir, image_path), new_name)
                    
                        if temp not in type_label:
                            type_label[temp] = [0, 0, 0, 0, live_label]
                else:
                    illuminations = os.listdir(label_dir)
                    for illumination in illuminations:
                        image_paths = os.listdir(os.path.join(label_dir, illumination))
                        image_paths.sort()
                        for i, image_path in tqdm(enumerate(image_paths)):
                            new_name = "{}{}{:06d}.{}".format(A_x[1:], live_label if label_x == "live" else spoof_label, i, image_path.split(".")[-1])
                            temp = os.path.join("Data", type_dir, A_x, label_x, illumination, new_name)
                            new_name = os.path.join(label_dir, illumination, new_name)
                            if not os.path.exists(new_name):
                                os.rename(os.path.join(label_dir, illumination, image_path), new_name)

                            if temp not in type_label:
                                if illumination == "other":
                                    type_label[temp] = [0, 1, 1, 1, 1]
                                elif illumination == "dark":
                                    type_label[temp] = [i%5, spoof_type, 4, environment, spoof_label]
                                else:
                                    type_label[temp] = [i%5, spoof_type, 2, environment, spoof_label]

        type_label = shuffle_dictionary(type_label)
        if type_dir == "train":
            with open(os.path.join(opt.our_label, 'train.json'), 'w') as outfile:
                json.dump(type_label, outfile)

        if type_dir == "test":
            with open(os.path.join(opt.our_label, 'test.json'), 'w') as outfile:
                json.dump(type_label, outfile)

if __name__ == "__main__":
    main()
