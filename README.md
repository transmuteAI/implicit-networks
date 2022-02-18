# Implicit-Networks
Code will be uploaded soon

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
