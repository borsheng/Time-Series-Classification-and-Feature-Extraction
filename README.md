# Time Series Classification and Feature Extraction

## Project Overview

This project is focusing on **Time Series Classification**. We use the AReM (Activity Recognition system based on Multisensor data fusion) dataset to classify human activities based on time series data obtained from a Wireless Sensor Network (WSN). The goal is to perform **feature extraction** from time series data, followed by **binary and multiclass classification** using logistic regression and L1-penalized logistic regression.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Classification](#classification)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Dataset

The AReM dataset contains time series data from seven types of human activities:
- Bending 1
- Bending 2
- Cycling
- Lying
- Sitting
- Standing
- Walking

Each folder contains files representing different instances of human activities. Each instance has six time series:
- `avg rss12`
- `var rss12`
- `avg rss13`
- `var rss13`
- `vg rss23`
- `ar rss23`

In total, there are 88 instances, each containing 480 consecutive data points for each time series.

## Feature Extraction

To classify time series data, it is essential to extract meaningful features. In this project, we extract the following **time-domain features** for each of the six time series in each instance:
- Minimum
- Maximum
- Mean
- Median
- Standard deviation
- First quartile
- Third quartile

Additionally:
- The **standard deviation** of each feature was estimated.
- A **90% bootstrap confidence interval** for the standard deviation was built.

We then selected the three most important features based on their relevance: **min**, **mean**, and **max**.

## Classification

### Binary Classification (Bending vs. Other Activities)

1. **Logistic Regression**:
   - Binary classification was performed to differentiate between "bending" and "other activities."
   - **Features**: The three selected features from time series 1, 2, and 6 were used for classification.
   - Scatterplots were created to visualize feature separation.

2. **Feature Splitting**:
   - Each time series was split into two equal parts, resulting in 12 time series per instance.
   - Logistic regression was performed again to check the effect of splitting on classification accuracy.

3. **Model Fitting**:
   - Logistic regression models were fitted with varying values of `l` (number of time series splits).
   - **Backward selection** using recursive feature elimination was employed to select the optimal set of features.
   - **Stratified cross-validation** was used to address class imbalance issues.

4. **Confusion Matrix and ROC Curves**:
   - The model's performance was evaluated using a confusion matrix, ROC curves, and AUC values.

5. **Test Set Evaluation**:
   - The classifier was tested on the test set, with time series split into the same number of parts as the training set.

6. **L1-Penalized Logistic Regression**:
   - The binary classification was repeated using L1-penalized logistic regression.
   - Cross-validation was performed for both `l` and the L1 penalty term (λ).
   - The results were compared with those obtained from p-value-based feature selection.

### Multiclass Classification

1. **Multinomial Regression**:
   - L1-penalized multinomial regression was used to classify all activities in the dataset.
   - The optimal value of `l` was determined using cross-validation.

2. **Naïve Bayes Classifier**:
   - The classification was repeated using a Naïve Bayes classifier with Gaussian and multinomial priors.
   - The results of multinomial regression and Naïve Bayes were compared to determine the better method for multiclass classification.

## How to Run

### Requirements

- Python 3.x
- Jupyter Notebook
- Required Python libraries (can be installed via `pip`):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Instructions

1. Clone the repository and navigate to the project folder.
2. Open the Jupyter Notebook (`Huang_Bor-Sheng.ipynb`).
3. Run the notebook cells to execute the feature extraction and classification tasks.

## Results

- **Binary Classification**:
  - Confusion matrix, ROC, and AUC were reported for both logistic regression and L1-penalized logistic regression.
  - [Add specific accuracy metrics and AUC values from your analysis.]

- **Multiclass Classification**:
  - Test error for multinomial regression and Naïve Bayes classifiers were compared.
  - [Add test error values and indicate which method performed better.]

## License

This project is intended for academic purposes and is based on the DSCI 552 course material.

