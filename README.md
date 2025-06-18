# Artwork Classification

## Description
Artwork Classification is a Python library designed for classifying artwork images using deep learning techniques. This project leverages convolutional neural networks (CNNs) to accurately identify and categorize various artworks based on their visual features. The library provides tools for data loading, preprocessing, model training, and evaluation, making it easy to implement and extend for various artwork classification tasks.

## Dataset

**Note:**  
This project originally used the [Painter by Numbers Kaggle competition dataset](https://www.kaggle.com/competitions/painter-by-numbers).  
As of 2025, this competition and its data are no longer available for download via the Kaggle API. If you wish to reproduce the results or run the examples, please use an alternative publicly available artwork dataset, such as [WikiArt](https://www.kaggle.com/datasets/crawford/wikiart).

## Features

- Modular Python library for data loading, model training, and evaluation
- Example Jupyter notebook for end-to-end workflow
- Unit tests and extensible code structure

## Installation
To install the required dependencies, you can use pip. Clone the repository and navigate to the project directory, then run:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the package directly using:

```bash
pip install .
```

## Usage
Here is a simple example of how to use the `artwork_classification` library in a Jupyter notebook:

```python
import torch
from artwork_classification.data import load_data
from artwork_classification.model import ArtworkClassifier
from artwork_classification.train import train_model
from artwork_classification.evaluate import evaluate_model

# Load the dataset
train_loader, test_loader = load_data('path/to/data')

# Initialize the model
model = ArtworkClassifier()

# Train the model
train_model(model, train_loader)

# Evaluate the model
accuracy = evaluate_model(model, test_loader)
print(f'Model accuracy: {accuracy:.2f}%')
```

## Running Tests

This project includes unit tests located in the `tests/` directory.  
To run all tests, use the following command from the project root:

```bash
pytest tests
```
