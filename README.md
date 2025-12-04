# Retail Sales Prediction

## Project Overview
This project involves the analysis and predictive modeling of sales data for a retail chain. The goal is to estimate the number of units that will be sold in the future based on historical data.

This repository represents a refactored and professionalized version of a technical test, structured as a production-ready Python project.

> **[View Original Problem Statement](docs/PROBLEM_STATEMENT.md)**: See the full requirements and objectives of the technical test.

## Key Results & "Senior-Level" Analysis

The most significant finding of this analysis is the counter-intuitive selection of a **Linear Regression** model over more complex ensemble methods like Random Forest or Gradient Boosting.

### Winning Model: Linear Regression
- **R² Score**: 0.82 (Excellent for 55 records)
- **Performance**: Outperforms Random Forest by +30 percentage points and Gradient Boosting by +40 percentage points.

### Why Linear Regression? 
In a field often obsessed with complex algorithms, this project demonstrates the importance of the **Bias-Variance Trade-off**:

1.  **Small Dataset Context**: After rigorous data cleaning (removing 50% duplicates to prevent data leakage), we are left with only 55 unique records.
2.  **Underfitting in Complex Models**: Complex models like Random Forest require significantly more data to learn effective decision boundaries. With only 55 records, they fail to capture the underlying patterns (High Bias in this specific context due to lack of data density).
3.  **Robustness of Simplicity**: Linear Regression, being a high-bias/low-variance model, is far more robust to small datasets. It captures the general trend without overfitting or getting confused by the sparsity of the data.

### Data Integrity & Leakage Prevention
A critical decision was made to **remove duplicate records**, reducing the dataset from 110 to 55 rows.
- **With Duplicates**: R² ≈ 0.99 (Artificial inflation due to Data Leakage - the model "memorizes" rows present in both train and test sets).
- **Without Duplicates**: R² ≈ 0.82 (Honest, defensible metric).
- **Decision**: Prioritized data integrity and valid metrics over artificially high performance numbers.

## Project Structure

```
retail-sales-prediction/
├── data/
│   ├── raw/            # Original dataset
│   └── processed/      # Processed data (if any)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning and transformation pipeline
│   └── modeling.py            # Model training and evaluation pipeline
├── results/
│   └── figures/        # Generated plots
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd retail-sales-prediction
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the analysis**:
    You can run the Jupyter notebook located in `notebooks/01_exploratory_analysis.ipynb` to see the full analysis and visualizations.

    Alternatively, you can import the modules from `src` to run the pipelines programmatically.

## Model Comparison

| Model | Test R² | Status |
|-------|---------|--------|
| **Linear Regression** | **0.82** | **WINNER - OPTIMAL** |
| Ridge | 0.76 | Very Good |
| Lasso | 0.73 | Good |
| Random Forest | 0.52 | Underfitting |
| Gradient Boosting | 0.42 | Severe Underfitting |

## Conclusion
This project demonstrates that **understanding your data and the limitations of your models is more important than simply applying the most complex algorithm**. The choice of Linear Regression is technically justified, robust, and responsible given the constraints of the dataset.
