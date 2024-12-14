# CNN for Image Classification

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras for image classification. The model is designed to classify images into predefined categories, leveraging deep learning techniques.

## Features
- Utilizes TensorFlow and Keras for model creation and training.
- Implements key CNN layers such as Conv2D, MaxPooling2D, and Dense.
- Includes dropout and batch normalization to improve model performance and prevent overfitting.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- scikit-learn (optional, for performance evaluation)

Install dependencies with:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```
if you are using mac, then -
```bash
!pip3 install tensorflow numpy matplotlib scikit-learn
```

## Dataset
The project expects an image dataset organized in a directory structure:
```
- dataset/
  - training/
    - class1/
    - class2/
  - testing/
    - class1/
    - class2/
  - single_prediction
    - image1
    - image 2

```


## Model Architecture
The CNN is built using the following layers:
1. Convolutional layers with ReLU activation
2. MaxPooling layers for downsampling
3. Flatten layer to convert 2D features into a 1D vector
4. Dense layers for classification
5. Dropout layers to reduce overfitting
6. Batch normalization for faster convergence

## Usage
1. Clone the repository and navigate to the project directory.
2. Prepare your dataset in the required format.
3. Open the Jupyter Notebook file `CNN for Image Classification.ipynb`.
4. Execute the cells sequentially to:
   - Load and preprocess the dataset.
   - Build and compile the CNN model.
   - Train the model and evaluate its performance.

## Results
- Training accuracy: 89%
- Test accuracy: 85%
- loss: 36%



## Acknowledgments
- TensorFlow/Keras for providing a robust deep learning framework.
- The dataset providers 


