# Distracted Driver Detection

This GitHub repository contains code for building a distracted driver detection model using the State Farm Distracted Driver Detection dataset. The project involves data preprocessing, model creation, training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Model](#pretrained-model)

## Introduction

Distracted driver detection is a crucial task for improving road safety. This project aims to develop a deep learning model capable of classifying images of drivers into different distraction categories. The code provided here includes data preprocessing, model construction, training, and evaluation.

## Prerequisites

Before you begin, ensure you have the following requirements met:

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- PIL (Python Imaging Library)
- Kaggle API key (for downloading the dataset from Kaggle)

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/distracted-driver-detection.git
cd distracted-driver-detection
```

2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle using your Kaggle API key:

```bash
kaggle datasets download -d your-username/state-farm-distracted-driver-detection
unzip state-farm-distracted-driver-detection.zip
```

## Data Preparation

The project starts by preprocessing the dataset, including:

- Organizing images into training and testing directories
- Creating labels and class directories
- Splitting the data into training and validation sets

The data preprocessing code can be found in `data_preparation.ipynb`. You can modify the parameters in this notebook to customize the data splitting ratio.

## Model Architecture

This project offers two different model architectures for distracted driver detection:

1. Custom Convolutional Neural Network (CNN)
2. Pretrained VGG16 model

The custom CNN architecture is defined in `custom_model.ipynb`, while the pretrained VGG16 model is defined in `pretrained_model.ipynb`.

## Training

You can train the selected model by running the appropriate notebook. Make sure to adjust hyperparameters like batch size, learning rate, and training epochs according to your needs.

## Evaluation

The trained models can be evaluated using various metrics such as accuracy and loss. The evaluation results are visualized using Matplotlib.

## Pretrained Model

If you prefer to use a pretrained model for inference, the VGG16-based model with trained weights is available for download.

Feel free to explore and contribute to this project to improve distracted driver detection and road safety!

**Note:** Make sure to replace `your-username` with your actual GitHub username when cloning the repository and `your-username` in the Kaggle dataset download command with your Kaggle username.
