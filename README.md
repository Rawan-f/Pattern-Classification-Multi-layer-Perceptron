# Pattern Classification with Multi-layer Perceptron

This Python code implements a Multilayer Perceptron (MLP) classifier using GloVe word embeddings for a classification task. 

## Introduction

Implement the MLP algorithm for supervised multi-class classification. The specific goal of this is to classify words into four distinct categories: animals, countries, fruits, and vegetables. To achieve this, we will utilize word embeddings that represent the meaning of the words.

## Prerequisites

Before running the code, ensure you have the following prerequisites:

- Python (>= 3.6)
- NumPy library (install with `pip install numpy`)
- scikit-learn library (install with `pip install scikit-learn`)
- Matplotlib library (install with `pip install matplotlib`)

## Installation

Follow these steps to set up the code:

1. Download the code files from [44580108.zip].

2. Install the required libraries listed in the "Prerequisites" section.

3. Download the GloVe word embeddings file (e.g., 'glove.6B.50d.txt') from [44580108.zip] and place it in the project directory.

4. Download the animals, countries, fruits
and veggies files from [44580108.zip] and place it in the project directory.

## Usage

To run the code, follow these instructions:

1. Open the Python Script: Open the Python script task2, task3, task4, task5, task6  in your preferred code editor or IDE.

2. Review the Code: Take a moment to review the code to understand its structure and functionality.

3. Run the script

## Results

After implementing and running the MLP classifier on the provided dataset, we obtained the following results:

### Training and Testing Accuracy

- Training Accuracy: [98.02%]
- Testing Accuracy: [92.19%]

### Performance vs. Number of Hidden Layers

We evaluated the performance of the MLP classifier by varying the number of hidden layers (k) and assessed how it impacted the accuracy. Here's a summary of the results:

- Number of Hidden Layers: [1]
- Corresponding Training Accuracies: [99.21%]
- Corresponding Testing Accuracies: [90.62%]

- Number of Hidden Layers: [2]
- Corresponding Training Accuracies: [99.21%]
- Corresponding Testing Accuracies: [95.31%]

- Number of Hidden Layers: [3]
- Corresponding Training Accuracies: [15.02%]
- Corresponding Testing Accuracies: [15.62%]
