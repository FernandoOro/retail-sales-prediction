# Technical Test: Data Scientist

## Business Context
You are working with a sales dataset from a retail chain. The dataset contains daily sales information from various stores, including data such as sale date, number of units sold, unit price, and product category.

**Your task is to build a prediction model to estimate the number of units that will be sold in the future based on past sales and other dataset characteristics.**

### Dataset
You have a file named `sales_data.csv` (provided as `sales_data.xlsx` in this repo) with the following columns:
- **Date**: The date of the sale.
- **Store**: The unique identifier of the store.
- **Category**: The product category.
- **Units_Sold**: The number of units sold.
- **Unit_Price**: The price per unit.

---

## Test Objectives

1.  **Exploratory Data Analysis (EDA)**: Analyze the data to identify patterns, trends, outliers, and potential relationships between variables.
2.  **Visualization**: Create visualizations that help better understand the data.
3.  **Data Cleaning**: Clean the data by removing null values and handling missing or duplicate data.
4.  **Normalization**: Normalize or standardize numeric features if necessary.
5.  **Encoding**: Encode categorical features using appropriate techniques.
6.  **Modeling**: Create a regression model to predict future sales based on historical data.
7.  **Evaluation**: Evaluate the model using appropriate metrics (e.g., RMSE, MAE).
8.  **Interpretation**: Interpret model results and provide recommendations on how to improve the model or adjust sales strategies based on the results.

---

## Deliverables

### 1. Exploratory Data Analysis (EDA) Report
A detailed report describing patterns, trends, and relationships identified in the dataset. This report must include:
*   Descriptive statistics (mean, median, standard deviation, etc.) for each variable.
*   Correlation analysis between variables.
*   Identification of outliers and their potential impact on the analysis.

### 2. Data Preprocessing Report
A report detailing the steps taken to clean and preprocess the data. This must include:
*   Methods used to handle null, missing, and duplicate values, with justification for each method.
*   Process of normalization or standardization of numeric features, if necessary.
*   Technique(s) used to encode categorical features (e.g., one-hot encoding, label encoding) and justification for the choice.

### 3. Predictive Model and Evaluation
A file containing the trained predictive model and a detailed description of the construction process, including:
*   Learning algorithm(s) used (e.g., linear regression, polynomial regression, etc.).
*   Feature selection and hyperparameter tuning processes.
*   Model evaluation using appropriate metrics (e.g., RMSE, MAE) and comparison between different models (if multiple were built).
*   Justification for the final model choice.

### 4. Results Interpretation and Recommendations Report
A final report interpreting the predictive model results and providing strategic recommendations. This must include:
*   Interpretation of model coefficients and how different features influence sales.
*   Identification of possible improvements in model accuracy (e.g., more data, different features, advanced modeling techniques).
*   Practical recommendations for the sales team based on model results (e.g., focus on specific products, pricing strategies).
*   Reflection on analysis limitations and suggestions for future work.
