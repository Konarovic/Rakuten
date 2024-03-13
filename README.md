# RakutenTeam
# Rakuten Challenge

This repository features machine learning models developed for a data project at **datascientest focusing on the Rakuten Challenge. 
The challenge consists in predicting product categories based on text descriptions and images. 
This repository includes implementations for handling text data, image data, combined multimodal data (text and image), 
and ensemble models to enhance prediction accuracy in the context of this challenge.

## Overview

There are five classes for different aspects of the multimodal classification task:

- `TFbertClassifier`: For text classification using BERT models.
- `MLClassifier`: For classification tasks using traditional machine learning algorithms.
- `ImgClassifier`: For image classification tasks using pre-trained models like Vision Transformer (ViT), EfficientNet, ResNet, VGG16, and VGG19.
- `TFmultiClassifier`: For multimodal deep networks that combines text and image data.
- `MetaClassifier`: For applying ensemble methods (voting, stacking, bagging, and boosting) to improve model performance by combining multiple of the above classifiers.

## Quick Start
