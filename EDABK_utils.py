# Necessary packages
import os
from typing import List
import shutil
from tqdm import tqdm

def get_path(path, mode):
  print("Getting {} label!!!".format(mode))
  list_paths = []
  for sub_img in tqdm(os.listdir(path)):
    sub_img_path = os.path.join(path, sub_img)
    live_path = os.path.join(sub_img_path, 'live')
    spoof_path = os.path.join(sub_img_path, 'spoof')

    live_tempt = os.path.join('Data', mode, sub_img, 'live')
    spoof_tempt = os.path.join('Data', mode, sub_img, 'spoof')

    if os.path.exists(live_path):
      for file in os.listdir(live_path):
        if '.png' in file or '.jpg' in file:
          list_paths.append(os.path.join(live_tempt, file))
    if os.path.exists(spoof_path):
      for file in os.listdir(spoof_path):
        if '.png' in file or '.jpg' in file:
          list_paths.append(os.path.join(spoof_tempt, file))
  return list_paths



def extract_sub_images(dictionary, sub_key):
  print("Extracting sub images!!!")
  sub_dictionary = {}
  for key, value in tqdm(dictionary.items()):
    if key in sub_key:
      sub_dictionary.update({key: value})
  return sub_dictionary

def check_path_exist(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def move_files(root_dir: str ,destination_dir: str, source_dirs: List, label: str) -> None:
    count = 0
    for p_ in tqdm(source_dirs):
        if os.path.exists(os.path.join(root_dir, p_)):
            count += 1
            parent_dir = os.path.join(destination_dir, p_)
            parent_dir = parent_dir[:len(parent_dir) - len(os.path.basename(parent_dir))]
            check_path_exist(parent_dir)
            shutil.copy2(os.path.join(root_dir, p_), parent_dir)

    print("Extracted {} {} !!!".format(count, label))

