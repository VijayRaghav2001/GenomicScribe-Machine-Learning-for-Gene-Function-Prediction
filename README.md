# GenomicScribe – Machine Learning for Gene Function Prediction

## Project Description

This project implements a machine learning pipeline for predicting gene functions from genomic sequences. The system processes DNA sequences, extracts relevant features, and applies various ML models to classify genes into functional categories based on the Gene Ontology (GO) framework.

## Features

- Data preprocessing for genomic files (FASTA, GenBank)
- Feature extraction including k-mer frequencies, GC content, and sequence length
- Multiple machine learning models (Random Forest, SVM, XGBoost, Logistic Regression)
- Hyperparameter tuning with cross-validation
- Comprehensive model evaluation and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genome-function-prediction.git
cd genome-function-prediction
