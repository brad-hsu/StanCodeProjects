# StanCodeProjects
My Collections of StanCodeProjects
## Coursework
Completed StanCode SC001,SC101 & SC201 Machine Learning Bootcamp.
Certificate included in this repository.


# Sentiment Analysis (StanCode SC201)

A lightweight sentiment classifier for movie reviews using sparse word features and gradient-based optimization.

## What it does
- Converts a review into a sparse bag-of-words feature vector (word counts)
- Trains a linear classifier with logistic loss via (stochastic) gradient descent
- Evaluates training/validation error per epoch
- Provides an interactive console demo where users can type reviews and get predictions

## Files
- `submission.py` — feature extraction + training (logistic loss, SGD) + optional character n-gram features
- `util.py` — sparse vector utilities (`dotProduct`, `increment`) and helper functions
- `interactive.py` — trains the model and launches interactive prediction

## How to run (example)
```bash
python interactive.py


## Methods

- Feature extraction: sparse bag-of-words (word counts)
- Model: logistic regression trained with gradient descent
- Evaluation: monitored training and validation error across epochs
