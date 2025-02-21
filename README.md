# deep-learning-challenge

# AlphabetSoupCharity Deep Learning Project

## Overview

This project leverages deep learning to predict the success of organizations funded by Alphabet Soup. Using TensorFlow/Keras, a neural network is trained to classify whether a non-profit will be successful based on given features.

## Files in This Repository

- AlphabetSoupCharity.ipynb - Initial model creation, training, and evaluation.

- AlphabetSoupCharity_Optimization.ipynb - Optimized version of the model to achieve higher accuracy.

- AlphabetSoupCharity.h5 - Saved model from the initial training.

- AlphabetSoupCharity_Optimization.h5 - Optimized model file.

## Project Steps

1. Data Preprocessing

- Load and clean the dataset.

- Identify target (IS_SUCCESSFUL) and feature variables.

- Encode categorical variables and normalize numerical data.

- Split the data into training and testing sets.

2. Neural Network Model

- Implemented a deep neural network with two hidden layers.

- Used the ReLU activation function in hidden layers and sigmoid in the output layer.

- Compiled with Adam optimizer and binary_crossentropy loss function.

3. Model Training and Evaluation

- Trained the model for 50 epochs with a batch size of 32.

- Evaluated performance using accuracy and loss metrics.

- Saved the model in .h5 format.

4. Model Optimization

- Adjusted layer structure (neurons, activation functions, and additional layers).

- Tried different batch sizes and epochs.

- Experimented with dropout layers and weight regularization.

## Results

- Initial model accuracy: XX%.

- Optimized model accuracy: XX%.

- Final model exceeded 75% accuracy threshold (if applicable).

## Setup Instructions

1. Clone the repository:

git clone https://github.com/your-repo-name.git
cd your-repo-name

2. Open AlphabetSoupCharity.ipynb in Jupyter Notebook or Google Colab.

3. Run each cell to preprocess data, train the model, and save results.

4. If optimizing, open AlphabetSoupCharity_Optimization.ipynb and experiment with different architectures.

## Dependencies

- Python 3.8+

- TensorFlow/Keras

- Pandas, NumPy, Sci-kit Learn

- Matplotlib (for visualization)

### Usage

- Modify the dataset as needed.

- Tune hyperparameters in AlphabetSoupCharity_Optimization.ipynb.

- Save the optimized model and use load_model() to make predictions.