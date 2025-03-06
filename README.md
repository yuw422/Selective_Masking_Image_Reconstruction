Selective Masking Image Reconstruction
================================================================================

This repository hosts a cleaned up implementation of the **Selective Masking Image Reconstruction** SSL method introduced in paper:

**Data-efficient Self-supervised Learning for Semantic Segmentation via Selective Masking Image Reconstruction**

Dataset Structure
--------------------------------------------------------------------------------
All datasets should be organized into the structure below.

Copy data list files from **data_list** to the below listed locations.
- **dataset_root_path**
  - **cityscapes**
    - leftImg8bit_trainextra
      - leftImg8bit/train_extra/data_list.txt
    - leftImg8bit_trainvaltes/
      - leftImg8bit/train/train.txt
      - leftImg8bit/val/val.txt
    - gtFine_trainvaltest
  - **pascal**
    - VOCtrainvaltest/VOCdevkit
      - VOC2012
      - VOC2007
  - **nassar**
    - ortho_tiles (generate use **tile_nassar.py** )
      - datalist.txt
    - tiled_dataset
      - train/data_list
        - train.txt
        - val.txt
        - test.txt
  - **sugarbeets**
    - ijrr_sugarbeets_2016_annotation
      - sugarbeets.txt
      - sugarbeets_weeds.txt


Environment
--------------------------------------------------------------------------------
Fill **setupenv.sh** with the correct path to dataset root and desired path for checkpoint storage.

This is tested in **Python 3.10**

See `requirements.txt` for details.

Pretraining
--------------------------------------------------------------------------------

Selective masking pretraining follows the below command structure:

```sh
python -m selectvie_loop.py --gpu {gpu number} --seed {random seed} --config {config file path}
```
For example, to pretrain selective masking on Cityscapes:
```sh
python -m selectvie_loop.py --gpu 0 --seed 0 --config experiments/image_reconstruction/selective_ir/reconstruction_selective_rec_city.yml 
```

Random masking pretraining is similar, only using **train.py** instread of **selectvie_loop.py**

For example, to pretrain random masking on Cityscapes
```sh
python -m train.py --gpu 0 --seed 0 --config experiments/image_reconstruction/random_ir/reconstruction_random_rec_city.yml
```


Downstream Task
--------------------------------------------------------------------------------
Once the pretraining is completed, the downstream training and testing can be performed.

The downstream tasks follow the below command structure:
```sh
python -m train.py --gpu {gpu number} --seed {random seed} --config {downsteam nconfig file path}
```

For example to perform the downstream task for selective masking pretraining
```sh
python -m train.py --gpu 0 --seed 0 --config experiments/downstream/cityscapes/downstream_selective_rec_cityscapes_cityscapes.yml
```

The naming conventions for the downstream config files are:
- \*no_pretrain\*.yml: no pretraining
- \*pretrained\*.yml: ImageNet pretrained backbone
- \*random_rec\*.yml: Random masking IR pretraining
- \*selective_rec_{downstream_set}_{pretrain_set}.yml: Selective masking IR pretraining on **pretrain_set**
  - downstream_selective_rec_cityscapes_cityscapes.yml is pretrained on Cityscapes
  - downstream_selective_rec_cityscapes_pascal.yml is pretrained on Pascal
