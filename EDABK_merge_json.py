# import os
# import sys
# import json
# from config import opt
# from EDABK_utils import shuffle_dictionary

# def main() -> None:
#     # load our data labels
#     our_train_label = json.load(open(os.path.join(opt.our_label, 'train.json')))
#     our_test_label = json.load(open(os.path.join(opt.our_label, 'test.json')))

#     # load celeb labels
#     train_label = json.load(open(os.path.join(opt.intra_test_temp, 'train_label.json')))
#     test_label = json.load(open(os.path.join(opt.intra_test_temp, 'train_label.json')))

#     # merge
#     for key in train_label.keys():
#         our_train_label[key] = train_label[key]
#     for key in test_label.keys():
#         our_test_label[key] = test_label[key]

#     # shuffle
#     our_train_label = shuffle_dictionary(our_train_label)
#     our_test_label = shuffle_dictionary(our_test_label)

#     # write labels
#     with open(os.path.join(opt.our_label, 'train.json'), 'w') as outfile:
#         json.dump(our_train_label, outfile)
#     with open(os.path.join(opt.our_label, 'test.json'), 'w') as outfile:
#         json.dump(our_test_label, outfile)

# if __name__ == "__main__":
#     main()