
# Dry Bean Classification

Welcome to the Dry Bean Classification project repository. This project focuses on building and evaluating various machine learning models to classify dry beans into distinct categories based on a given dataset with seven unique classes.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Results and Analysis](#results-and-analysis)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Introduction
This project addresses the problem of classifying dry beans into distinct categories based on a given dataset with seven unique classes. The dataset contained 16 features, including traditional metrics like area, perimeter, and compactness, as well as shape-related factors such as eccentricity and solidity. The main goal was to build and evaluate various machine learning models to identify the most accurate classifier for distinguishing among the different types of dry beans.

## Project Structure
The repository contains the following files and directories:
```
├── dry_bean_classification_train.csv  # Training dataset
├── dry_bean_classification_test.csv   # Testing dataset
├── FinalProjectSwe.ipynb              # Jupyter notebook with project code
├── FinalProjectSwe.py                 # Python script with project code
├── Report.pdf                         # Detailed project report
└── README.md                          # This readme file
```

## Setup and Installation
To run this project, ensure you have the following dependencies installed:
- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn

You can install the required Python packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Usage
1. **Jupyter Notebook:**
   - Open `FinalProjectSwe.ipynb` in Jupyter Notebook.
   - Run the cells sequentially to preprocess the data, train the models, and evaluate their performance.

2. **Python Script:**
   - Alternatively, you can run the Python script `FinalProjectSwe.py` directly:
     ```bash
     python FinalProjectSwe.py
     ```
   - Ensure the datasets (`dry_bean_classification_train.csv` and `dry_bean_classification_test.csv`) are in the same directory as the script.

## Results and Analysis
The project employed various machine learning models, including:
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Multilayer Perceptron (MLP)
- Nearest Means Classifier
- Random Classifier (baseline)

Dimensionality reduction techniques like Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) were used to improve model performance.

Key findings:
- **MLP with PCA** achieved the highest test accuracy of 90.78%.
- **SVM with PCA** and **KNN with LDA** also showed promising results with test accuracies of 91.25% and 90.33%, respectively.
- The trivial random classifier achieved a test accuracy of approximately 14%.

For detailed results and analysis, please refer to the `Report.pdf` file.

---
