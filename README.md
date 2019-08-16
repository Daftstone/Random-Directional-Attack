# Random Directional Attack for Fooling Deep Neural Networks

This project is for the paper "Random Directional Attack for Fooling
Deep Neural Networks". Our implementation is based 
on [cleverhans](https://github.com/tensorflow/cleverhans/tree/v.3.0.1) .

The code was developed on Python 3.6


## 1. Install dependencies.
Our experiment runs on GPU,, install this list:
```bash
pip install -r requirements_gpu.txt
```

## 2. Download dataset and pre-trained models.
Download [dataset](https://drive.google.com/open?id=1nTufCMOgYWhLEhu0MJ4YQLNvqKlG0jtT) 
and extract it to the root of the program, which contains the MNIST, SVHN, and ImageNet-10 dataset.

Download [pre-trained models](https://drive.google.com/open?id=1HxaF_pp7THAc0RLD6XPLMMIA3uHPgcFH)
and extract it to the root of the program. 
## 3. Usage of `python white_box.py`
```
usage: python white_box.py [--data DATA_NAME] [--max_angle MAX_ANGLE]
               [--nb_dimensions NB_DIMENSIONS] [--is_train [IS_TRAIN]]
               [--eps EPS] [nb_epochs EPOCHS_NUMBER] [batch_size BATCH_SIZE]

optional arguments:
  --data DATA_NAME
                        Supported: MNIST, SVHN, CIFAR-10, ImageNet-10.
  --max_angle MAX_ANGLE
                        Maximum angle of rotation.
  --nb_dimensions NB_DIMENSIONS
                        Number of dimensions selected.
  --is_train [IS_TRAIN]
                        User this parameter to train online, otherwise remove the parameter.
  --eps EPS
                        The size of perturbations
  --nb_epochs EPOCHS_NUMBER
                        Number of epochs the classifier is trained.
  --batch_size BATCH_SIZE
                        Size of each batch of data
```

### 4. Example.
Use pre-trained model.
```bash
python white_box.py --data mnist --max_angle 180 --nb_dimensions 10 --eps 0.05
```
Train model online.
```bash
python white_box.py --data mnist --max_angle 180 --nb_dimensions 10 --eps 0.05 --is_train --batch_size 128
```


## 5. Usage of `python black_box.py`
```
usage: python black_box.py [--data DATA_NAME] [--max_angle MAX_ANGLE]
               [--nb_dimensions NB_DIMENSIONS] [--is_train [IS_TRAIN]] [--sub_is_train SUB_IS_TRAIN]
               [--eps EPS] [nb_epochs EPOCHS_NUMBER] [batch_size BATCH_SIZE]

optional arguments:
  --data DATA_NAME
                        Supported: MNIST, SVHN, CIFAR-10, ImageNet-10.
  --max_angle MAX_ANGLE
                        Maximum angle of rotation.
  --nb_dimensions NB_DIMENSIONS
                        Number of dimensions selected.
  --is_train [IS_TRAIN]
                        User this parameter to train target model online, otherwise remove the parameter.
  --sub_is_train [SUB_IS_TRAIN]
                        User this parameter to train subtitute model online, otherwise remove the parameter.
  --eps EPS
                        The size of perturbations
  --nb_epochs EPOCHS_NUMBER
                        Number of epochs the classifier is trained.
  --batch_size BATCH_SIZE
                        Size of each batch of data
```

### 6. Example.
Use pre-trained model.
```bash
python black_box.py --data mnist --max_angle 180 --nb_dimensions 10 --eps 0.05
```
Train model online.
```bash
python black_box.py --data mnist --max_angle 180 --nb_dimensions 10 --eps 0.05 --is_train --sub_is_train
```


## 7. Citation
If you want to use random directional atttack for attack in academic research, you are expected to cite
@article{luo2019random,
  title={Random Directional Attack for Fooling Deep Neural Networks},
  author={Luo, Wenjian and Wu, Chenwang and Zhou, Nan and Ni, Li},
  journal={arXiv preprint arXiv:1908.02658},
  year={2019}
}
