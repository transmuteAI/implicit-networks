# Implicit Equivariance in Convolutional Networks
Link to the paper- https://arxiv.org/pdf/2111.14157.pdf

# Abstract
Convolutional Neural Networks (CNN) are inherently equivariant under translations, however, they do not have an equivalent embedded mechanism to handle other transformations such as rotations
and change in scale. Several approaches exist that
make CNNs equivariant under other transformation
groups by design. Among these, steerable CNNs
have been especially effective. However, these approaches require redesigning standard networks with
filters mapped from combinations of predefined basis
involving complex analytical functions. We experimentally demonstrate that these restrictions in the
choice of basis can lead to model weights that are suboptimal for the primary deep learning task (e.g. classification). Moreover, such hard-baked explicit formulations make it difficult to design composite networks comprising heterogeneous feature groups. To
circumvent such issues, we propose Implicitly Equivariant Networks (IEN) which induce equivariance in
the different layers of a standard CNN model by
optimizing a multi-objective loss function that combines the primary loss with an equivariance loss term.
Through experiments with VGG and ResNet models
on Rot-MNIST , Rot-TinyImageNet, Scale-MNIST
and STL-10 datasets, we show that IEN, even with
its simple formulation, performs better than steerable
networks. Also, IEN facilitates construction of heterogeneous filter groups allowing reduction in number of channels in CNNs by a factor of over 30%
while maintaining performance on par with baselines.
The efficacy of IEN is further validated on the hard
problem of visual object tracking. We show that
IEN outperforms the state-of-the-art rotation equivariant tracking method while providing faster inference speed.

Code files for all E2CNN and IEN experiments are provided. 
All codes are written in python 3 and can be run simply after downloading the dataset.
To run any file: Use command "python filename.py" in terminal/command prompt

Some important points:

-  For running E2CNN experiments, make sure to install E2CNN library using:
       pip install git+https://github.com/QUVA-Lab/e2cnn@c77faef49fd1bf12ccf538a63cac201a89f16c6
       
-  Datasets can be downloaded from the following sites:
       For Rot-MNIST experiments: 
            http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip

       For R4 and R8 experiments on Rot-TIM dataset:
            https://drive.google.com/file/d/1bkFvJQNNM1kczghmkHIcDQ2aO_rqh_3p/view?usp=sharing
       
       For R4R experiments on R2-TIM dataset:
            https://drive.google.com/file/d/1fkGRK0MQrEHxq4VTAYUxhVvkZNieDF-a/view?usp=sharing

-  For standard CNN experiments, use beta_weight param as 0 in IEN implementations
