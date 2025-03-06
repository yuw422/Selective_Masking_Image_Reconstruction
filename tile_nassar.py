import PIL
import numpy as np
from skimage import io

PIL.Image.MAX_IMAGE_PIXELS = None

#TODO change to nassar orthomosaic path
nassar_path = ""
#TODO change to dataset path
dataset_path=""

otho = io.imread(f'{nassar_path}/Nassar/orthomosaic/data/RGB_reference_png.png')
# print(otho.shape)
# otho = np.zeros((90599, 29395, 3), dtype=np.uint8)

x_ranges = np.arange(0, otho.shape[0], 256)
y_ranges = np.arange(0, otho.shape[1], 256)

x_ranges[-1] = otho.shape[0] - 256
y_ranges[-1] = otho.shape[1] - 256

print(x_ranges, y_ranges)
print(len(x_ranges), len(y_ranges))
# input()
f = open(f'{dataset_path}/data_collect/nassar/nassar.txt', 'w')
count = 0
for x in x_ranges:
    for y in y_ranges:
        tile = otho[x:x + 256, y:y + 256, :]
        io.imsave(f'{dataset_path}/data_collect/nassar/nassar_{count:05d}.png', tile)
        f.write(f'nassar_{count:05d}.png\n')
        count += 1
        # print(tile.shape)
        # print(x, y)
        # input()
f.close()
