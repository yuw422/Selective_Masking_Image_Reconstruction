import numpy as np
import skimage
import matplotlib.pyplot as plt
from datasets.nassar_unlabelled import NassarSelective, NassarSelectiveMask
from datasets.pascal_voc_unlabelled import PascalSelective
from datasets.cityscapes_unlabelled import CitySelective

# d = NassarSelective(64, 10)
# d = PascalSelective(64, 500)
d = CitySelective(64, 500)
print(len(d.data_list))
print(len(d.partition_list))
for l in d.partition_list:
    print(len(l))

# pd = d.create_partition_dataset(1)
aug = d.aug_selective
# aug["args"]["augmentation_steps"] = []
pd = d.create_partition_selective_dataset(1, aug)
data_set = pd.get_aug_dataset('train')
s = 5
print(type(data_set))
for i in range(len(data_set)):
    _, ax = plt.subplots(2, s + 1)
    for j in range(s):
        sample = data_set.get_item_with_path(i)
        x, y = sample["data"]
        print(x.dtype, y.dtype)
        print(x.min(), x.max(), y.min(), y.max())
        input()
        i0 = np.transpose(x.numpy(), (1, 2, 0))
        i1 = np.transpose(y.numpy(), (1, 2, 0))
        ax[0][j].imshow(((i0 - np.min(i0)) / (np.max(i0) - np.min(i0)) * 255).astype(np.uint8))
        ax[1][j].imshow(((i1 - np.min(i0)) / (np.max(i1) - np.min(i1)) * 255).astype(np.uint8))
    # x,y = sample["data"]
    print(x.shape, y.shape)
    # i0 = np.transpose(x.numpy(), (1, 2, 0))
    # i1 = np.transpose(y.numpy(), (1, 2, 0))
    # _, ax = plt.subplots(1,3)
    # ax[0].imshow(((i0 - np.min(i0)) / (np.max(i0) - np.min(i0)) * 255).astype(np.uint8))
    print(sample["path"])
    ii = skimage.io.imread(sample["path"][1])
    # ii = skimage.io.imread(sample["path"])
    ax[0][-1].imshow(ii)
    ax[1][-1].imshow(ii)
    plt.show()

    input()
