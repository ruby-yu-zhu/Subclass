

## Requirements 
* [Pytorch](https://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/)

## Quick start

### Training

#### COCO-MLT
```
python tools/train.py configs/coco/LT_resnet50_pfc_DB.py 
```

#### VOC-MLT
```
python tools/train.py configs/voc/LT_resnet50_pfc_DB.py 
```

### Testing

#### COCO-MLT
```
bash tools/dist_test.sh configs/coco/LT_resnet50_pfc_DB.py work_dirs/LT_coco_resnet50_pfc_DB/epoch_8.pth 1
```

#### VOC-MLT
```
bash tools/dist_test.sh configs/voc/LT_resnet50_pfc_DB.py work_dirs/LT_voc_resnet50_pfc_DB/epoch_8.pth 1
```

## Pre-trained models

#### COCO-MLT

|   Backbone  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: | :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |      53.55      |      51.13     |    57.05     |     51.06    |     [model](https://drive.google.com/file/d/1HPQMmPVfqiDUTmzrTxNv3clhYa662QKb/view?usp=sharing)      |

####  VOC-MLT

|   Backbone  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: | :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |      78.94      |      73.22     |    84.18     |     79.30    |     [model](https://drive.google.com/file/d/1jGHiCfQKDNjdYxjKXfp8ifFadW2BuGWm/view?usp=sharing)      |

## Datasets

<img src='./assets/dataset.png' width=400>



