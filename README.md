# Cifar10-CNN

Convolutional Neural Network project based on Cifar-10 dataset from [Udacity Deep Learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101). 
Depending on the exercise in CNN part I made a model to work on the project. With the simple architecture and carefully chosen hyperparameters, the accuracy has increased to 84%

## About the project
The project shows classifying of CIFAR-10 database. The database is consisted of images that are small color that fall into one of ten classes. Below is the samples from the database.


<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/cifar_data.png">

### Prerequisites

Things you have to install or installed on your working machine:

* Python 3.8.5
* Numpy (win-64 v1.191.2)
* Pandas (win-64 v1.1.3)
* Matplotlib (win-64 v3.3.2)
* Jupyter Notebook
* Torchvision (win-64 v0.8.2)
* PyTorch (win-64 v1.7.1)

### Environment:

* [Anaconda](https://www.anaconda.com/)

## Jupyter Notebook

* `cifar10_cnn.ipynb`

The jupyter notebook describes the whole project from Udacity. You can find the whole description inside the notebook.

### Visualize a Batch of Training Data¶

We perform some simple data augmentation by randomly flipping and rotating the given image data. We do this by defining a torchvision transform.
This type of data augmentation should add some positional variety to these images, so that when we train a model on this data, it will be robust in the face of geometric changes (i.e. it will recognize a ship, no matter which direction it is facing).
Here is the samples after augmenting the data

<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/samples.PNG">

### An Image in More Detail

we look at the normalized red, green, and blue (RGB) color channels as three separate, grayscale intensity images.

<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/RGB.PNG">

### Define the Network Architecture

We defined a CNN architecture. Instead of an MLP, which used linear, fully-connected layers, you'll use the following:

* Convolutional layers, which can be thought of as stack of filtered images.
* Maxpooling layers, which reduce the x-y size of an input, keeping only the most active pixels from the previous layer.
* The usual Linear + Dropout layers to avoid overfitting and produce a 10-dim output.

We used CNN Layer blocks. Here is the summary of our model

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]          18,496
              ReLU-5           [-1, 64, 32, 32]               0
         MaxPool2d-6           [-1, 64, 16, 16]               0
            Conv2d-7          [-1, 128, 16, 16]          73,856
       BatchNorm2d-8          [-1, 128, 16, 16]             256
              ReLU-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
             ReLU-11          [-1, 128, 16, 16]               0
        MaxPool2d-12            [-1, 128, 8, 8]               0
        Dropout2d-13            [-1, 128, 8, 8]               0
           Conv2d-14            [-1, 256, 8, 8]         295,168
      BatchNorm2d-15            [-1, 256, 8, 8]             512
             ReLU-16            [-1, 256, 8, 8]               0
           Conv2d-17            [-1, 256, 8, 8]         590,080
             ReLU-18            [-1, 256, 8, 8]               0
        MaxPool2d-19            [-1, 256, 4, 4]               0
          Dropout-20                 [-1, 4096]               0
           Linear-21                 [-1, 1024]       4,195,328
             ReLU-22                 [-1, 1024]               0
           Linear-23                  [-1, 512]         524,800
             ReLU-24                  [-1, 512]               0
          Dropout-25                  [-1, 512]               0
           Linear-26                   [-1, 10]           5,130
================================================================
Total params: 5,852,170
Trainable params: 5,852,170
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.96
Params size (MB): 22.32
Estimated Total Size (MB): 26.30
----------------------------------------------------------------
```

## Training results

```
Epoch: 30 	Training Loss: 0.539699 	Validation Loss: 0.163262
```
See the visual result over iteration:

<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/vall-traingraph.PNG">

## Accuracy

The test data showing how well the neural network is modeling the data

```
Test Loss: 0.530732

Test Accuracy of airplane: 85% (850/1000)
Test Accuracy of automobile: 92% (924/1000)
Test Accuracy of  bird: 76% (768/1000)
Test Accuracy of   cat: 69% (691/1000)
Test Accuracy of  deer: 83% (839/1000)
Test Accuracy of   dog: 76% (765/1000)
Test Accuracy of  frog: 87% (879/1000)
Test Accuracy of horse: 89% (896/1000)
Test Accuracy of  ship: 91% (911/1000)
Test Accuracy of truck: 92% (923/1000)

Test Accuracy (Overall): 84% (8446/10000)
```

## Author 

* Shahrullohon Lutfillohonov


## License
[MIT](https://choosealicense.com/licenses/mit/)
