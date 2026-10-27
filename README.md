# Interactive Exploration of Ensemble Learning for Student Performance Prediction

This project presents an interactive application for investigating how individual machine learning models and their ensembles predict academic performance. It supports the full experimental workflow, from exploring educational data to comparing predictions, assessing uncertainty, and examining results for new student records.

The application compares two approaches to the same prediction task: **direct multiclass classification**, which predicts a performance category, and **regression followed by categorization**, which predicts a numerical grade and then assigns a corresponding category. Using a shared evaluation framework makes it possible to examine how the prediction strategy and ensemble method affect the results.

## Data and Prediction Task

The study uses the UCI Student Performance dataset, combining 395 mathematics records and 649 Portuguese-language records. The resulting dataset contains **1,044 student–subject records**; some students appear in both subjects.

Nine predictors describe age, parental education, travel time, study time, previous academic failures, absences, the first-period grade, and the subject studied. The second-period grade is excluded. The target, **Performance Index**, represents the final grade on the original **0–20 scale**.

For classification, grades are assigned to **Low**, **Medium**, and **High** categories using thresholds determined from the training data. Related records are kept together when forming the training and test subsets to reduce the risk of overlap between them.

## Experimental Workflow

### 1. Data Loading and Initial Analysis

The application presents the dataset structure, feature types, descriptive statistics, missing values, and repeated rows. This stage supports an initial assessment of the data and their suitability for prediction.

### 2. Data Preprocessing and Sample Formation

Data preparation includes handling missing values, scaling selected predictors, and forming training and test subsets. Preprocessing parameters and category thresholds are determined from the training data. The same split is used to compare both prediction strategies.

### 3. Data Visualization

Histograms, boxplots, and violin plots illustrate feature distributions before and after preprocessing. Association matrices and target-association charts help explore relationships among variables, with measures selected according to feature type.

### 4. First-Level Model Training

The application investigates Ridge regression or Ridge classification, Random Forest, and XGBoost. Cross-validation and hyperparameter selection support the comparison of regularized linear models with nonlinear tree-based methods. Bayesian optimization is available for the supported model searches.

### 5. Meta-Feature Generation

Out-of-fold predictions form the input representation for ensemble learning. Classification uses predicted class probabilities, while regression uses numerical predictions. This allows the second-level model to learn from predictions generated for records excluded from the corresponding first-level training fold.

### 6. Ensemble Construction

**Soft Voting** combines class probabilities, while regression **Voting** combines numerical predictions. Adjustable weights allow the contribution of the base models to be explored.

**Stacking** learns how to combine the first-level predictions through a separate meta-model: Logistic Regression for classification and Ridge for regression. The application therefore supports comparison between aggregation with specified weights and a learned combination of predictions.

### 7. Evaluation, Statistical Comparison, and Prediction

Classification is assessed using Accuracy, Balanced Accuracy, F1 macro, and F1 weighted. Regression is assessed using MAE, RMSE, and the coefficient of determination, alongside classification metrics after categorization.

Confusion matrices, actual-versus-predicted plots, residual plots, and class probabilities provide complementary views of prediction behavior. Bootstrap confidence intervals and paired differences describe uncertainty, while McNemar tests with Holm correction support comparisons of classification correctness. Group-based analysis is also available to examine the effect of related records.

Tables, figures, predictions, and experiment reports can be exported for further analysis and academic reporting. The application also supports predictions for individual new records and batches of observations.

## Research Purpose

The project allows users to investigate whether ensemble learning improves on individual models, how classification and regression-based categorization differ, and whether observed improvements are supported by statistical evidence. It also illustrates why model selection should consider uncertainty, class-specific errors, and the required form of prediction alongside numerical performance metrics.

## Data Source

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
