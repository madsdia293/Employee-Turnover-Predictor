# Employee-Turnover-Predictor

This project aims to predict employee turnover using machine learning techniques. It involves exploring and preprocessing the data, performing exploratory data analysis, training logistic regression and random forest models, evaluating their performance, and hyperparameter tuning. The project is implemented in Python.

## Overview

The Employee Turnover Predictor project follows the following steps:

### Data Loading: 
The HR data is loaded from a CSV file using the pandas library.
### Data Preprocessing: 
Duplicate rows are removed from the dataset to ensure data integrity.
### EDA: 
Exploratory data analysis is performed to gain insights into the data. Visualizations are created to analyze features such as monthly hours worked, number of projects, promotion history, work accidents, years at the company, department, and salary level.
### Data Preprocessing: 
Categorical variables (department and salary level) are encoded using one-hot encoding. Numerical features are scaled using min-max scaling.
### Data Split: 
The data is split into training and testing sets using the train_test_split function from scikit-learn.
### Model Training: 
Two models, logistic regression and random forest, are trained on the training data using the LogisticRegression and RandomForestClassifier classes from scikit-learn.
### Model Evaluation: 
The models are evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices are generated to visualize the performance of the models in predicting employee turnover.
### Hyperparameter Tuning: 
Grid search cross-validation is performed on the random forest model to find the best combination of hyperparameters that maximizes model performance.
### Model Saving: 
The best-performing model, determined by the grid search cross-validation, is saved as a pickle file for future use.

## Engineered Features
No additional features are engineered in this project. The focus is on analyzing the existing features to predict employee turnover.

## Models Created
The following models are created and evaluated:

### Logistic Regression: 
A logistic regression model is trained on the standardized training data using the LogisticRegression class from scikit-learn. The model predicts the probability of an employee leaving the company based on the input features.
### Random Forest: 
A random forest model is trained on the training data using the RandomForestClassifier class from scikit-learn. The model predicts employee turnover by aggregating predictions from multiple decision trees.
### Hyperparameter Tuning:
Grid search cross-validation is performed on the random forest model using the GridSearchCV class from scikit-learn to find the best combination of hyperparameters that maximizes model performance.

## Requirements
To run this project, you will need the following:

Python 3.x: The project is implemented in Python, so make sure you have Python 3.x installed on your system.

Jupyter Notebook: Jupyter Notebook is used for executing the project code. Install Jupyter Notebook by running the following command in your terminal: pip install jupyter

Required Python libraries: The project relies on various Python libraries such as pandas, numpy, seaborn, and scikit-learn. Install these libraries by running the following command: pip install pandas numpy seaborn scikit-learn matplotlib

## File Description
#### employee_turnover_predictor.py: 
The main Python script that performs data analysis, data preprocessing, model training, and model evaluation.
#### hr_data.csv: 
The HR data file containing the employee information and target variable (whether an employee left or stayed).
##### model.pkl: 
The trained model saved in pickle format.

Once you have fulfilled these requirements, you can proceed to execute the project code.
