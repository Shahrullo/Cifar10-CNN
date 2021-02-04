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

<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/summary.PNG">

## Training results

```
Epoch: 30 	Training Loss: 0.539699 	Validation Loss: 0.163262
```
See the visual result over iteration:

<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/vall-traingraph.PNG">

## Accuracy

The test data showing how well the neural network is modeling the data

<img src="https://github.com/Shahrullo/Cifar10-CNN/blob/main/notebook_ims/accuracy.PNG">


## Author 

* Shahrullohon Lutfillohonov


## License
[MIT](https://choosealicense.com/licenses/mit/)
