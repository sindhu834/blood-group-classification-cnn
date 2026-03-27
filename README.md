# Blood Group Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify blood groups based on images of blood samples. The goal is to develop a system that can assist in quick and accurate blood group identification.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Usage](#usage)
- [Results](#results)
- [Conclusions](#conclusions)
- [License](#license)

## Introduction
Blood group classification is critical in healthcare, especially during blood transfusions. This project leverages deep learning techniques to automate this process, offering a potential solution for rapid diagnostic tools.

## Dataset
The dataset consists of images of blood samples categorized into different blood groups: A, B, AB, and O. Each class contains a diverse set of images to improve model generalization.

## Model Architecture
The model is designed using several convolutional layers followed by pooling layers. It employs:
- Convolutional layers for feature extraction
- Dropout layers to prevent overfitting
- Fully connected layers for final classification

## Training Process
- Training was performed using a TensorFlow/Keras framework.
- The dataset was divided into training, validation, and test sets to evaluate performance.
- Data augmentation techniques were applied to enhance model robustness.

## Usage
To use the trained model for prediction, follow these steps:
1. Load the model.
2. Preprocess the input image.
3. Pass the image through the model to get predictions.

## Results
Results should include metrics such as accuracy, precision, and recall to indicate model performance.

## Conclusions
The proposed model shows promise for blood group classification, with potential applications in clinical settings. Future work could involve using larger datasets and exploring other architectures for improved accuracy.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.