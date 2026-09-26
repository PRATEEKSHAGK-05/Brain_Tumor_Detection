# Brain Tumor Detection and Classification from MRI

A deep learning-based system for **automatic brain tumor detection and classification from MRI images** using a **Convolutional Neural Network (CNN)**.

The model classifies MRI brain images into four categories:

* **Glioma**
* **Meningioma**
* **No Tumor**
* **Pituitary**

## Project Overview

Manual identification of brain tumors from MRI scans can be time-consuming, especially when different tumor types have similar visual characteristics. This project uses a CNN-based deep learning approach to automatically analyze MRI images and classify them into the appropriate category.

The images are resized and normalized before being passed to the CNN model. The model automatically extracts relevant image features and performs classification using fully connected layers.

## Problem Statement

Identifying brain tumors manually from MRI scans requires significant time and effort. Different tumor types may have similar visual characteristics, making classification difficult.

This project aims to develop an AI-based system that can:

* Automatically analyze MRI brain images
* Extract important visual features
* Classify MRI images into four tumor categories
* Provide prediction probabilities and confidence scores
* Reduce the time and effort required for manual classification

## Solution

A **Convolutional Neural Network (CNN)** is developed to analyze MRI brain images.

The overall approach includes:

1. Loading MRI images from the dataset
2. Resizing images to **128 × 128 pixels**
3. Normalizing pixel values
4. Extracting features using convolution and pooling layers
5. Flattening extracted features
6. Performing classification using dense layers
7. Applying dropout to reduce overfitting
8. Generating class probabilities using Softmax
9. Selecting the predicted class using the highest probability

## Dataset

The dataset contains MRI brain images organized into four classes.

| Dataset   | Number of Images |
| --------- | ---------------: |
| Training  |              800 |
| Testing   |               80 |
| **Total** |          **880** |

### Training Dataset

* Glioma – 200 images
* Meningioma – 200 images
* No Tumor – 200 images
* Pituitary – 200 images

### Testing Dataset

* 80 images
* 20 images from each class

### Image Processing

* Image size: **128 × 128 pixels**
* Batch size: **16**
* Pixel normalization: `Rescaling(1./255)`

These dataset details are taken from the case study presentation.

## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Convolutional Neural Network (CNN)**
* **NumPy**
* **Google Colab**
* **MRI Image Dataset**

## CNN Architecture

The model uses the following major components:

```text
Input MRI Image
       ↓
Resize to 128 × 128
       ↓
Rescaling / Normalization
       ↓
Conv2D
       ↓
MaxPooling
       ↓
Conv2D
       ↓
MaxPooling
       ↓
Conv2D
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense Layer
       ↓
Dropout
       ↓
Output Layer
       ↓
Softmax
       ↓
4 Classes
```

### Classification Classes

```text
0 → Glioma
1 → Meningioma
2 → No Tumor
3 → Pituitary
```

## Model Training

The CNN model is trained using:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Epochs:** 15
* **Batch Size:** 16
* **Activation:** Softmax for final classification layer

The presentation specifies Adam optimization, Sparse Categorical Crossentropy, and 15 training epochs.

## Workflow

```text
MRI Brain Image
       ↓
Dataset Loading
       ↓
Image Resizing
       ↓
Image Normalization
       ↓
CNN Feature Extraction
       ↓
Flatten Features
       ↓
Dense Layer
       ↓
Dropout
       ↓
Softmax Classification
       ↓
Prediction + Confidence Score
```

## Prediction

After training, the model predicts the class of a given MRI image.

The Softmax layer produces probabilities for all four classes, and `np.argmax()` selects the class with the highest probability.

Example:

```text
Glioma       : 0.01
Meningioma   : 0.9776
No Tumor     : 0.005
Pituitary    : 0.0064
```

**Predicted Class: Meningioma**

**Confidence: 97.76%**

The case study reports a tested image being predicted as **Meningioma with 97.76% confidence**.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

### 3. Prepare the Dataset

Organize the dataset in the following structure:

```text
Dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

### 4. Run the Model

Open the notebook or Python file in **Google Colab** and execute the cells sequentially.

The dataset can be loaded using:

```python
tf.keras.utils.image_dataset_from_directory()
```

## Results

The CNN successfully classifies MRI brain images into four categories:

* Glioma
* Meningioma
* No Tumor
* Pituitary

The system also generates prediction probabilities and confidence scores for the input image.

## Applications

This project demonstrates how deep learning can be applied to medical image classification for:

* Automated MRI image analysis
* Brain tumor classification
* Computer-aided medical image analysis
* Educational and research applications

> **Note:** This project is intended for educational and research purposes. It is not a substitute for professional medical diagnosis.

## Future Enhancements

* Increase the size and diversity of the dataset
* Apply data augmentation
* Perform hyperparameter tuning
* Compare CNN with transfer learning models such as ResNet, VGG, and EfficientNet
* Add model evaluation metrics such as precision, recall, F1-score, and confusion matrix
* Develop a web-based interface for uploading MRI images
* Deploy the trained model as an API

## Team

**Prateeksha G K** – 23CSR165
**Praneesh C** – 23CSR163
**Prajit Pranav K** – 23CSR161

**Course:** 22CSC71 – Deep Learning

## Conclusion

This project demonstrates a CNN-based approach for automated brain tumor detection and classification from MRI images. The system processes MRI images, extracts relevant features using convolutional layers, and classifies them into four categories: **Glioma, Meningioma, No Tumor, and Pituitary**.

The model provides both the predicted tumor category and its confidence score, demonstrating the potential of deep learning for faster automated MRI image classification.
