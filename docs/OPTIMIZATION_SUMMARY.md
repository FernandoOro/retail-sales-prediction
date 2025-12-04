# Model Optimization Summary

---

## Overview

This document summarizes the optimization steps applied to the sales prediction notebook and the rationale behind the final model selection.

## Applied Changes

### 1. Contextualization
A new section was added to the notebook to explicitly address the constraints of the small dataset (55 records). This sets the analytical expectation that simpler models may outperform complex ones due to the limited training data.

### 2. Analytical Depth
The modeling section was enhanced to include:
*   A detailed analysis of the **Bias-Variance Trade-off**.
*   A technical comparison between Linear Regression and Random Forest performance.
*   An explanation of the underfitting observed in ensemble methods.

### 3. Documentation
The final report (`FINAL_VALIDATION_REPORT.md`) was updated to include benchmarks for R² scores relative to dataset size, providing a frame of reference for evaluating the model's performance.

---

## Model Performance

The following table summarizes the performance of the evaluated models on the test set:

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| **Linear Regression** | **5.52** | **4.77** | **0.82** |
| Ridge | 6.23 | 5.43 | 0.76 |
| Lasso | 6.55 | 5.74 | 0.73 |
| Decision Tree | 8.92 | 7.13 | 0.53 |
| Random Forest | 8.99 | 7.48 | 0.52 |
| Gradient Boosting | 9.88 | 8.26 | 0.42 |

**Selected Model:** Linear Regression (R² = 0.82).

---

## Technical Justification

### Model Selection
Linear Regression was selected as the optimal model. Despite the popularity of ensemble methods like Random Forest, the dataset size (55 unique records) is insufficient to train complex models effectively.
*   **Random Forest:** With hundreds of trees, the model requires a higher density of data points to establish stable decision boundaries. In this context, it fails to capture the signal, resulting in underfitting (R² = 0.52).
*   **Linear Regression:** As a high-bias, low-variance model, it is better suited for small datasets. It captures the primary linear trends without being sensitive to the noise or sparsity that confuses more complex algorithms.

### Data Integrity
The decision to remove 50% of the dataset (duplicates) was critical.
*   **With Duplicates:** The model achieved an R² of ~0.99. However, this was due to data leakage, as identical rows appeared in both training and testing sets.
*   **Without Duplicates:** The R² of 0.82 represents the model's true ability to generalize to unseen data.

## Conclusion

The optimization process prioritized methodological rigor over raw performance metrics. The final model achieves an excellent balance of accuracy and interpretability given the constraints of the data.
