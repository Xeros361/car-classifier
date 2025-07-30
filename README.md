# Car Classifier - Neural Network from Scratch

This simple project is a small neural network written in Python, without using any machine learning libraries like TensorFlow or PyTorch. Its goal is to classify whether a car is sport or not, based on several parameters.

## What the Model Does

Given 4 input features  of a car:
- Max speed
- Weight
- Acceleration
- Number of seats

the model predicts whether the car is sport or non-sport, and provides a probability score.

---

## Project Structure

- main.py - lets the user input car features and get predictions from the model
- train.py - trains the model using the provided dataset ('cars.csv')
- model.py - 
- data.py - handles data loading, normalization, and processing
- metrics.py - 
- user_input.py - manages user interaction

---

## How It Works

### Architecture

- Input layer: 4 neurons (one for each feature)
- Hidden layer 1: 4 neurons, with ReLU activation
- Hidden layer 2: 4 neurons, with ReLU activation
- Output layer: 1 neuron, with Sigmoid activation (gives probability from 0 to 1)

### Training

- Loss function: Mean Squared Error (MSE)
- Backpropagation:
- Learning rate: adjustable, default is 0.005
- Epochs: customizable, default is 250

---

## Dataset

There are 2 types of dataset:
- 'cars.csv': is used for training the model, which includes:
  - 200 training examples: 100 sports cars and 100 non-sports
- 'test_cars.csv': is used for testing the model, which includes:
  - 50 test examples: 25 sports cars and 25 non-sports

### Normalization

Before training and testing, all data is normalized using Min-Max Scaling to ensure that features are on the same scale.

## How to run

Run the main script:

$ python main.py

## Requirements
- pandas 

## Limitations

- The training process is quite slow due to the lack of optimization.
- The model is trained on simplified, synthetic data and  may not generalize well to real world cars.
- The neural network architecture is very basic, which can lead to low accuracy and high MSE in some cases.
