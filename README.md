# fsl-dogs-breed-classification

The code shown is an excerpt from my engineering thesis, the abstract of which is below. It is an implementation of a prototype network in terms of the Stanford Dogs Dataset. The additional functionality that enables data retrieval and dataset preparation can be easily adapted to your own dataset with any structure.

### Table of contents
 - [Abstract](#abstract)
 - [Project Structure](#project-structure)
 - []()

## Abstract

The classical approach to training a neural network requires a large number of training data. This
is a major problem of deep learning, which significantly limits the ability to develop models. Scien-
tists inspired by the functioning of the human brain have created a new branch of deep learning:
meta learning. This is a way of training an artificial neural network in the regime of a small number
of data representing each class, so that it can quickly adapt to a specific classification task. In this
work, based on the architecture of the convolutional network, we discuss exactly how the artificial
neural network works. We then define the few shot learning problem. We discuss other appro-
aches to this problem, and describe transfer learning with EfficientNet architecture and Siamese
network architecture. We describe in detail the architecture that creates the designed predictive
system - prototype networks. We show how the selected network learns to create a multidimen-
sional representation of the features of the objects in the images, describe the process of creating
prototypes, how the objects are classified, and how the loss is updated in a cost function. As part
of our description of the prototype networks architecture, we present the Densenet model that we
use to encode features of the images in the latent space. In the next section, we describe working
methodology for testing model. We describe used metrics and the most relevant hyperparameters
of the network. We present the results of experiments studying the effect of hyperparameters on
the model’s behavior. In the last chapter, we describe conclusions of experiments, make an overall
review of the model, and present possible directions for future work.

## Project structure

1. [*run.sh*](src/run.sh) -  Linux shell script to download and initialize the dataset Stanford Dogs Dataset, from the official site.
2. [*training_summary.py*](src/training_summary.py) - allows you to locally analyze your network training with the *tensorboard* tool. As an argument, you can specify a path with training logs.
3. [*train.py*](src/train.py) - The training script allows you to adjust parameters:
   - *l_r*: learning rate
   - *epochs*: number of training epochs
   - *K* factor
   - *M* factor
4. [*test.py*](src/test.py) - A script for testing trained models. Arguments:
   - *path*: path to a specific model
   - *conf*: whether to show the confusion matrix
   - *report*: whether to show the report 
5. [*single_prediction.py*](src/single_prediction.py) - This script is used to perform a single prediction using any previously trained model. It is based on a very trivial GUI that allows you to select an image from a catalog, perform a classification and provide the results of the experiment.
   - *path*: path to a specific model
   - *K*: factor


<img src="https://github.com/Tymass/fsl-dogs-breed-classification/assets/83314524/a7beff30-e4c5-4e04-abc2-82d6850e92e1" width="400">
<img src="https://github.com/Tymass/fsl-dogs-breed-classification/assets/83314524/c6cbcddd-c5ad-491f-ac03-2b6838b26eca" width="600">


