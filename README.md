# Implicit Equivariance in Convolutional Networks

#### 1. Overview
This repository consists of the official implementation of the paper "[Implicit Equivariance in Convolutional Networks](https://arxiv.org/abs/2111.14157)". Standard CNNs are inherently equivariant under translations, however they are not equivariant under other transformations such as rotation and change in scale. In our works we induced equivariance implicitly in the different layers of a standard CNN model by optimizing a multi-objective loss function. More details can be found [here](https://arxiv.org/abs/2111.14157).

#### 2. Dependencies

- Python 3.x
- PyTorch
- Torchvision
- Pillow

#### 3. Download Datasets

- <b>Rot-MNIST</b>: It is a variation of the popular MNIST dataset containing handwritten digits. In Rot-MNIST, the digits are rotated by an angle generated uniformly between 0 and 2π radians.[Link to this dataset](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip)
- <b>Rot-TIM</b>: It is a variation of the variation of TinyImageNet dataset which is a miniature version of the ImageNet Dataset. In Rot-TIM, images are rotated by an angle generated uniformly between 0 and 2π radians.[Link to this dataset ](https://drive.google.com/file/d/1bkFvJQNNM1kczghmkHIcDQ2aO_rqh_3p/view)
- <b>R2-TIM</b>: This dataset is the same as Rot-TIM apart from the fact that some images were flipped at random along the horizontal or vertical axis.[LLink to this dataset](https://drive.google.com/file/d/1fkGRK0MQrEHxq4VTAYUxhVvkZNieDF-a/view)

#### 4. Train Model
Edit params dictionary in ```main.py```. Choose parameters from the following options: <br>

Training Settings:
> ```dataset``` : ```rot_mnist```, ```rot_tim```. For R2-TIM, choose ```rot_tim``` dataset and set ```reflection : True``` <br>
> ```model``` : ```BasicCNN``` (for Rot-MNIST), ```Resnet18```, ```VGG``` (for Rot-TIM & R2-TIM). <br>
> ```reflection``` : ```True``` or ```False```. For learning equivariance under reflections. ```Default : False```<br>
> ```heterogeneous``` : ```True``` or ```False```. For testing equivariance under heterogeneous filter groups. ```Default : False```<br>
> ```mode``` : ```R4``` or ```R8```. R4 and R8 denote equivariance to 4 and 8 equidistant orientations respectively. ```Default : R4``` <br>
> ```save_dir``` : Specify folder for saving checkpoints. ```Default : ien_training```<br>
> ```training``` : ```True``` or ```False```.<br>

If ```heterogeneous``` is set to ```True```:
> For R8-4-2-1, Set ```ratio : [32,8,2,1]``` and ```mplc_kernel_size : [8,4,2,1]``` <br>
> For R8-4-2, Set ```ratio : [8,2,1]``` and ```mplc_kernel_size : [8,4,2]``` <br>
> For R4-2-1, Set ```ratio : [8,2,1]``` and ```mplc_kernel_size : [4,2,1]``` <br>

Adjusting hyper-parameters:
> ```beta_weight``` : weightage for equivariance loss i.e. extent of equivariance required. For training equivalent Standard CNN i.e. without equivariance loss, Use ```beta_weight : 0```.<br>
> ```tot_epochs``` : Total Epochs to train. ```Default : 90```<br>
> ```batch_size``` : ```Default: 32```. <br>
> ```weight_decay_rate``` : Weight Decay Rate for Adam optimizer. ```Default : 1e-7```. <br>

Hyper-parameters for Cyclic LR:
> ```max_lr``` : Highest LR for the cycle. ```Default : 5e-3```<br>
> ```low_lr``` : Lowest LR for the cycle. ```Default : 1e-5```<br>
> ```cyc_size``` : Number of epochs for one cycle. ```Default : 70```<br>
> ```dec_scale``` : Factor by which peak LR decreases for next cycle. ```Default : 1```<br>

After adjusting parameters and hyperparameters, run ```main.py```
```bash
python main.py
```

#### 5. Any further questions?
Please contact <b>Naman Khetan</b> (namankhetan10@gmail.com) and <b>Tushar Arora</b> (tushararora1410@gmail.com) for any query or information.
