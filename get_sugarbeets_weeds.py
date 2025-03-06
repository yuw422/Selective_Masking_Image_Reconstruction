import os
import numpy as np
import random
import config
from utilities.io import load_image
from datasets.sugarbeet import SugarBeets
from datasets.sugarbeet_seg import SugarBeetsSeg

dir_sugarbeets = 'sugarbeets/ijrr_sugarbeets_2016_annotations'  # server location
directory = os.path.join(config.DATASET_DIRECTORY, dir_sugarbeets)

data_list_fn = directory + "/labels.txt"

f_weeds = open('sugarbeets_weeds.txt', 'w')
count = 0
file_list = []
with open(data_list_fn) as f:
    for line in f:
        lbl_path = directory+SugarBeets.get_label_image_path(line[:-1])
        img_path = SugarBeets.get_rgb_path(lbl_path)
        # print(img_path, lbl_path)
        raw_label = load_image(lbl_path)
        label = SugarBeetsSeg.convert2trainid(raw_label)
        if 2 in label:
            print(SugarBeets.get_label_image_path(line[:-1]))
            print(label.shape, np.unique(label))
            file_list.append(SugarBeets.get_label_image_path(line[:-1]))
            # f_weeds.write(SugarBeetsSeg.get_label_image_path(line[:-1])+"\n")
            count+=1
            # input("CHECK")
print(f"Found {count} labels with weed class")
random.shuffle(file_list)
for weed_fn in file_list:
    f_weeds.write(weed_fn+"\n")
f_weeds.close()