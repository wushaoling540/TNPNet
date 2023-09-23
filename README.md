# TNPNet: A Novel Approach to Few-Shot Open-Set Recognition via Contextual Transductive Learning
Office code repository for "TNPNet: A Novel Approach to Few-Shot Open-Set Recognition via Contextual Transductive Learning"

## Prerequisites

### 1. Prepare Dataset

Dataset Source can be downloaded here.

- [MiniImageNet](https://drive.google.com/file/d/12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D/view?usp=sharing)
- [TieredImageNet](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07)
- [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

Download them and move datasets under `datasets` folder.

### 2. Prepare Pretrain Weights

Download pretrain weights [here](https://drive.google.com/file/d/13qurSQ0PjLm7cJeWd3GGb_IMyQU1dW0L/view?usp=drive_link) and move them under `initialization` folder.

### 3. Running 
Just run ` main.py ` for training and testing

For example, to train and test the 5-way 1-shot/5-shot setting on MiniImageNet:

```
python main.py --dataset MiniImageNet --max_epoch 80 --shot 1 --eval_shot 1 --init_weights [/path/to/pretrained/weights] --sigma 10000 --vector 48
```

to train and test the 5-way 1-shot/5-shot setting on TieredImageNet:

```
python main.py --dataset TieredImageNet --max_epoch 80 --shot 1 --eval_shot 1 --init_weights [/path/to/pretrained/weights] --sigma 1e8 --vector 42
```

to train and test the 5-way 1-shot/5-shot setting on CIFAR-FS:

```
python main.py --dataset CIFAR-FS --max_epoch 12 --shot 1 --eval_shot 1 --init_weights [/path/to/pretrained/weights] --sigma 10000 --vector 72
```

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [FEAT](https://github.com/Sha-Lab/FEAT)
- [GEL](https://github.com/00why00/Glocal)
- [TANE](https://github.com/shiyuanh/TANE)
- [few-shot-ssl](https://github.com/renmengye/few-shot-ssl-public)

