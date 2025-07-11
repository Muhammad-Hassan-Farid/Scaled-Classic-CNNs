# Scaled Classic CNNs for Brain Tumor Classification


## Overview

This repository explores the classification of brain tumors from MRI scans using several classic Convolutional Neural Network (CNN) architectures. It provides a comparative analysis between the standard implementations of these models and versions that have been scaled up in terms of resolution and width (number of channels/neurons). This approach is inspired by model scaling principles, such as those used in EfficientNet, to evaluate whether systematically enhancing classic CNNs can improve their performance on this medical imaging task.

## Dataset

The project utilizes a brain MRI dataset organized into four distinct classes:
*   `glioma`
*   `meningioma`
*   `notumor`
*   `pituitary`

The dataset is structured with each class having its own subdirectory containing the respective MRI images.

## Models Explored

Each Jupyter notebook in this repository implements and evaluates a pair of models: a classic CNN and its scaled counterpart.

*   **VGG**
    *   `paper-brain-mri-vgg-16.ipynb`: VGG-16 vs. Scaled VGG-16
    *   `paper-brain-mri-vgg-19.ipynb`: VGG-19 vs. Scaled VGG-19

*   **ResNet**
    *   `paper-brain-mri-resnet-50.ipynb`: ResNet-50 vs. Scaled ResNet-50
    *   `paper-brain-mri-resnet-101.ipynb`: ResNet-101 vs. Scaled ResNet-101

*   **DenseNet**
    *   `paper-brain-mri-densenet-desnet-121.ipynb`: DenseNet-121 vs. Scaled DenseNet-121
    *   `paper-brain-mri-densenet-desnet-169.ipynb`: DenseNet-169 vs. Scaled DenseNet-169

## Methodology

### 1. Data Preprocessing

*   Images are loaded in grayscale.
*   For classic models, images are resized to `130x130` pixels.
*   For scaled models, a higher resolution of `156x156` pixels is used.
*   Pixel values are normalized to a [0, 1] range.
*   The dataset is split into a 70% training set and a 30% testing set.

### 2. Model Scaling

The scaled models are enhanced using a compound scaling approach:
*   **Resolution:** Image resolution is increased from `130x130` to `156x156`.
*   **Width:** The number of filters in convolutional layers and neurons in fully connected layers are increased by a factor of approximately 1.2. The fundamental block structure and layer count of the original architecture are maintained.

For example, in the scaled VGG-16 model, the first convolutional block uses 77 filters instead of 64, and the fully connected layers use 4920 neurons instead of 4096.

### 3. Training

*   **Optimizer:** `Adam` optimizer is used with a low learning rate.
*   **Loss Function:** `sparse_categorical_crossentropy` is employed for multi-class classification.
*   **Callbacks:**
    *   `EarlyStopping`: Halts training if validation accuracy does not improve for 5 consecutive epochs, preventing overfitting.
    *   `ModelCheckpoint`: Saves the model weights that achieve the best validation accuracy.

## Evaluation

Model performance is assessed on the held-out test set using a comprehensive set of metrics:
*   Accuracy
*   Precision (Weighted)
*   Recall (Weighted)
*   F1-Score (Weighted)
*   Specificity and Sensitivity (per class)

The notebooks generate detailed classification reports, confusion matrices, and plots of training/validation accuracy and loss curves to facilitate a thorough comparison between the classic and scaled models.

## How to Run

### Prerequisites

Ensure you have Python installed, along with the following libraries:
```
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/Muhammad-Hassan-Farid/Scaled-Classic-CNNs.git
    cd Scaled-Classic-CNNs
    ```
2.  Create a directory named `Datasest` in the root of the project.
3.  Inside `Datasest`, create subdirectories for each class and place the corresponding images inside:
    ```
    .
    ├── Datasest/
    │   ├── glioma/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    ├── paper-brain-mri-vgg-16.ipynb
    └── ...
    ```

### Execution

Open and execute the cells in any of the `.ipynb` files using Jupyter Notebook or a compatible IDE to train and evaluate the corresponding models.

## Results

Each notebook provides a direct comparison between a classic CNN architecture and its scaled version. The evaluation metrics and visualizations (confusion matrices, accuracy/loss plots) generated at the end of each notebook demonstrate the impact of scaling on model performance. These results allow for a clear assessment of improvements in learning efficiency and final predictive accuracy for the brain tumor classification task.
