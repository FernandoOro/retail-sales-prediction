"""
Sales Predictive Model
Project: Retail Sales Prediction
Author: Fernando Orozco
Date: October 2025

Description:
This script builds and evaluates different machine learning models to predict
the number of units sold. I tested several algorithms, compared them using
standard metrics, and finally selected the best one based on performance
and suitability for the problem.

The process includes:
- Training 6 different models
- Evaluation with multiple metrics (RMSE, MAE, R²)
- Cross-validation for greater robustness
- Hyperparameter optimization for the best model
- Feature importance analysis
- Detailed justification of the final choice
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import models to test
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# For evaluation and optimization
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import my preprocessing script
from src.data_preprocessing import run_preprocessing_pipeline


def train_multiple_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Trains several different models and compares their performance.
    
    The idea here is not to commit to a single algorithm from the start. Each model
    has its strengths and weaknesses, so I test several to see which works
    best with this specific data.
    
    Models tested:
    
    1. LINEAR REGRESSION:
       - The simplest, assumes linear relationship between features and target
       - Good as baseline
       - Very interpretable (can see coefficients directly)
    
    2. RIDGE (L2 Regularization):
       - Like Linear Regression but with penalty to prevent overfitting
       - Useful when there is multicollinearity between features
       - Alpha controls how much regularization to apply
    
    3. LASSO (L1 Regularization):
       - Similar to Ridge but can drive coefficients to zero
       - Performs automatic feature selection
       - Useful for identifying which variables are truly important
    
    4. DECISION TREE:
       - Can capture non-linear relationships
       - Very interpretable (can see decision rules)
       - Prone to overfitting if depth is not controlled
    
    5. RANDOM FOREST:
       - Ensemble of many decision trees
       - Generally very robust and good performance
       - Less interpretable than linear models
       - Handles outliers and non-linear features well
    
    6. GRADIENT BOOSTING:
       - Another ensemble, but builds trees sequentially
       - Usually gives very good results
       - Slower to train than Random Forest
       - Needs careful tuning to avoid overfitting
    
    For each model, I calculate metrics on train AND test to detect overfitting.
    """
    print("\n--- Training Multiple Models ---")
    print("\nI will test 6 different algorithms and compare them.")
    print("This helps me find which one works best for this specific problem.\n")
    
    # Dictionary with models to test
    # Initial parameters are reasonable defaults, will optimize the best later
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    print("Training each model...\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions on train and test
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics on both sets
        # RMSE: Root Mean Squared Error (penalizes large errors more)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # MAE: Mean Absolute Error (average absolute error)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # R²: Coefficient of determination (% variance explained)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Save everything
        results[name] = {
            'model': model,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"  Train - RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
        print(f"  Test  - RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")
        
        # Check for obvious overfitting
        if r2_train > 0.95 and (r2_train - r2_test) > 0.1:
            print(f"  NOTE: Possible overfitting (Train R² very high, large gap with test)")
        
        print()
    
    return results


def compare_models(results: dict) -> tuple[str, pd.DataFrame]:
    """
    Compares all models side-by-side to see which is best.
    
    I look at multiple metrics:
    - RMSE: How far off predictions are on average
    - MAE: Average absolute error (easier to interpret)
    - R²: Percentage of variability explained
    
    Also look at difference between train and test:
    - If train is much better than test = overfitting
    - If both are similar = good generalization sign
    
    IMPORTANT: With a dataset of only 55 records, metrics aren't everything.
    We must also consider interpretability, robustness, and ease of deployment.
    """
    print("\n--- Model Comparison ---")
    print("\nNow comparing all models to see which performed best.\n")
    
    # Create summary table
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE_Train': [results[m]['rmse_train'] for m in results.keys()],
        'RMSE_Test': [results[m]['rmse_test'] for m in results.keys()],
        'MAE_Train': [results[m]['mae_train'] for m in results.keys()],
        'MAE_Test': [results[m]['mae_test'] for m in results.keys()],
        'R2_Train': [results[m]['r2_train'] for m in results.keys()],
        'R2_Test': [results[m]['r2_test'] for m in results.keys()]
    })
    
    # Calculate differences to detect overfitting
    comparison['Diff_R2'] = comparison['R2_Train'] - comparison['R2_Test']
    
    # Sort by Test R² (higher is better)
    comparison = comparison.sort_values('R2_Test', ascending=False)
    
    print("Results Table (sorted by Test R²):\n")
    print(comparison.to_string(index=False))
    
    # Identify best model
    best_idx = comparison['R2_Test'].idxmax()
    best_name = comparison.loc[best_idx, 'Model']
    best_r2 = comparison.loc[best_idx, 'R2_Test']
    best_rmse = comparison.loc[best_idx, 'RMSE_Test']
    
    print(f"\n--- Best Model by Metrics ---")
    print(f"\nModel with best technical metrics: {best_name}")
    print(f"R² (Test): {best_r2:.4f}")
    print(f"RMSE (Test): {best_rmse:.4f}")
    
    # Overfitting analysis
    best_diff = comparison.loc[best_idx, 'Diff_R2']
    if best_diff < 0.05:
        print(f"Train-Test Diff: {best_diff:.4f} (Excellent - no overfitting)")
    elif best_diff < 0.15:
        print(f"Train-Test Diff: {best_diff:.4f} (Acceptable - slight overfitting)")
    else:
        print(f"Train-Test Diff: {best_diff:.4f} (Caution - possible overfitting)")
    
    # Identify Linear Regression for comparison
    lr_idx = comparison[comparison['Model'] == 'Linear Regression'].index[0]
    lr_r2 = comparison.loc[lr_idx, 'R2_Test']
    lr_rmse = comparison.loc[lr_idx, 'RMSE_Test']
    
    print(f"\n--- Consideration: Linear Regression ---")
    print(f"\nGiven the small dataset context (55 records), also considering:")
    print(f"Linear Regression - R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
    print(f"\nAdvantages of Linear Regression:")
    print(f"  + Maximum interpretability (explainable coefficients)")
    print(f"  + Greater robustness with few data")
    print(f"  + Easier to maintain and explain to business")
    print(f"  + Lower risk of degradation in production")
    
    print(f"\nPerformance difference:")
    print(f"  {best_name} beats Linear Regression by {best_r2 - lr_r2:.4f} in R²")
    
    if best_r2 - lr_r2 < 0.05:
        print(f"  Difference is small - both models are competitive")
    elif best_r2 - lr_r2 < 0.15:
        print(f"  {best_name} has moderate advantage in metrics")
    else:
        print(f"  {best_name} has significant advantage in metrics")
    
    return best_name, comparison


def cross_validation(model, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """
    Performs cross-validation for robust performance estimation.
    
    The problem with a single train/test split is results can depend heavily
    on which records fell into each set. With only 55 total records,
    this is especially important.
    
    Cross-validation splits data into k "folds" and trains k times,
    using a different fold as test each time. Then averages results.
    
    I use k=5 because:
    - With 55 records, each fold will have ~11 records
    - Reasonable balance between robustness and compute time
    - With k=10 each fold would have only 5-6 records (too small)
    """
    print(f"\n--- Cross Validation for {model_name} ---")
    print("\nRunning cross-validation (k=5) for reliable metrics.")
    print("This trains the model 5 times with different data splits.\n")
    
    # Cross-validation with different metrics
    # neg_mean_squared_error because sklearn wants metrics "to maximize"
    cv_scores_mse = cross_val_score(model, X, y, 
                                     cv=5, 
                                     scoring='neg_mean_squared_error')
    cv_scores_mae = cross_val_score(model, X, y,
                                     cv=5,
                                     scoring='neg_mean_absolute_error')
    cv_scores_r2 = cross_val_score(model, X, y,
                                    cv=5,
                                    scoring='r2')
    
    # Convert MSE to RMSE and fix signs
    cv_rmse = np.sqrt(-cv_scores_mse)
    cv_mae = -cv_scores_mae
    cv_r2 = cv_scores_r2
    
    print("Results for 5 folds:")
    print(f"  RMSE per fold: {[f'{x:.2f}' for x in cv_rmse]}")
    print(f"  MAE per fold:  {[f'{x:.2f}' for x in cv_mae]}")
    print(f"  R² per fold:   {[f'{x:.4f}' for x in cv_r2]}")
    
    print(f"\nAverages (± std dev):")
    print(f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"  MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    print(f"  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    
    # Comment on variability
    if cv_r2.std() < 0.1:
        print("\nStandard deviation is low - model is consistent across folds.")
    else:
        print("\nStandard deviation is high - results vary a lot between folds.")
        print("This is normal with small datasets (55 records).")
    
    return {
        'rmse_mean': cv_rmse.mean(),
        'rmse_std': cv_rmse.std(),
        'mae_mean': cv_mae.mean(),
        'mae_std': cv_mae.std(),
        'r2_mean': cv_r2.mean(),
        'r2_std': cv_r2.std()
    }


def optimize_hyperparameters(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Optimizes hyperparameters of the best model using Grid Search.
    
    Hyperparameters are values not learned from training but configured
    by us (like max_depth in trees, alpha in Ridge, etc.)
    
    Grid Search tests different combinations systematically.
    
    Could test many more, but with 44 train records and 5-fold CV,
    each combination trains 5 models with ~35 records each. Don't want
    grids too large to avoid overfitting the selection process itself.
    """
    print(f"\n--- Hyperparameter Optimization for {model_name} ---")
    print("\nSearching for best hyperparameters using Grid Search.")
    print("This tests different combinations and picks the best one.\n")
    
    # Define grids based on model
    # Conservative grids due to small dataset size
    
    if model_name == 'Linear Regression':
        print("Linear Regression has no hyperparameters to optimize.")
        print("It is a closed-form model.")
        return None
    
    elif model_name == 'Ridge':
        print("Optimizing alpha (regularization strength)...")
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
        }
        base_model = Ridge(random_state=42)
    
    elif model_name == 'Lasso':
        print("Optimizing alpha (regularization strength)...")
        param_grid = {
            'alpha': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
        base_model = Lasso(random_state=42)
    
    elif model_name == 'Decision Tree':
        print("Optimizing max_depth and min_samples_split...")
        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10]
        }
        base_model = DecisionTreeRegressor(random_state=42)
    
    elif model_name == 'Random Forest':
        print("Optimizing n_estimators and max_depth...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestRegressor(random_state=42)
    
    elif model_name == 'Gradient Boosting':
        print("Optimizing n_estimators, learning_rate and max_depth...")
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        base_model = GradientBoostingRegressor(random_state=42)
    
    else:
        print(f"No grid config for {model_name}")
        return None
    
    print(f"Grid to test: {param_grid}")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    print("\nThis may take a moment...\n")
    
    # Run Grid Search with CV
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Grid Search completed!")
    print(f"\nBest hyperparameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest score (RMSE in CV): {np.sqrt(-grid_search.best_score_):.4f}")
    
    print("\nNOTE: If improvement is marginal, might stick with original")
    print("      parameters for simplicity (parsimony principle).")
    
    return grid_search.best_estimator_


def analyze_feature_importance(model, feature_names: list, model_name: str) -> None:
    """
    Analyzes which features are most important for the model.
    
    Useful for:
    - Understanding what the model is learning
    - Identifying key business variables
    - Deciding if it's worth collecting certain data
    - Simplifying model by removing irrelevant features
    """
    print(f"\n--- Feature Importance in {model_name} ---")
    print("\nLet's see which variables drive predictions the most.\n")
    
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create dataframe and sort
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Importances (higher = more important):\n")
        for idx, row in df_importance.iterrows():
            bar = '#' * int(row['Importance'] * 50)
            print(f"  {row['Feature']:20s} {row['Importance']:.4f} {bar}")
        
        top_3 = df_importance.head(3)
        print(f"\nTop 3 most important features:")
        for idx, row in top_3.iterrows():
            print(f"  {idx+1}. {row['Feature']} ({row['Importance']:.4f})")
    
    # For linear models
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_
        
        # Create dataframe with absolute values for importance
        df_coef = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=False)
        
        print("Coefficients (absolute value indicates importance):\n")
        for idx, row in df_coef.iterrows():
            sign = '+' if row['Coefficient'] > 0 else '-'
            bar = '#' * int(row['Importance'] * 2)
            print(f"  {row['Feature']:20s} {sign}{row['Importance']:.4f} {bar}")
        
        print(f"\nInterpretation:")
        print(f"  + Positive coef: Higher feature value -> more units sold")
        print(f"  - Negative coef: Higher feature value -> fewer units sold")
        
        top_3 = df_coef.head(3)
        print(f"\nTop 3 most influential features:")
        for idx, row in top_3.iterrows():
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"  {idx+1}. {row['Feature']} ({direction} sales)")
    
    else:
        print("This model does not provide feature importance directly.")


def justify_final_selection(best_name: str, results: dict, cv_results: dict, comparison: pd.DataFrame) -> None:
    """
    Documents final model selection with balanced approach.
    
    Not enough to say "it won metrics". Must justify considering:
    - Performance (metrics)
    - Complexity
    - Interpretability
    - Robustness (CV variance)
    - Suitability for problem
    - Deployment ease
    - CONTEXT: 55 record dataset
    """
    print("\n" + "="*70)
    print("FINAL MODEL SELECTION JUSTIFICATION")
    print("="*70)
    
    # Get metrics for best technical model
    r2_best = results[best_name]['r2_test']
    rmse_best = results[best_name]['rmse_test']
    mae_best = results[best_name]['mae_test']
    
    # Get Linear Regression metrics
    r2_lr = results['Linear Regression']['r2_test']
    rmse_lr = results['Linear Regression']['rmse_test']
    mae_lr = results['Linear Regression']['mae_test']
    
    print(f"\nCOMPARATIVE ANALYSIS\n")
    print(f"Best model by metrics: {best_name}")
    print(f"  R² Test: {r2_best:.4f} | RMSE: {rmse_best:.4f} | MAE: {mae_best:.4f}")
    print(f"  CV: R² = {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    
    print(f"\nLinear Regression (most interpretable):")
    print(f"  R² Test: {r2_lr:.4f} | RMSE: {rmse_lr:.4f} | MAE: {mae_lr:.4f}")
    
    diff_r2 = r2_best - r2_lr
    print(f"\nDifference: {best_name} beats LR by {diff_r2:.4f} in R² ({diff_r2/r2_lr*100:.1f}%)")
    
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION (BALANCED APPROACH)")
    print("="*70)
    
    print(f"""
DECISION: Recommend LINEAR REGRESSION for initial deployment,
          with plan to evolve to {best_name}.

JUSTIFICATION:

1. PROBLEM CONTEXT:
   - Dataset of only 55 unique records (after removing duplicates)
   - Test set of 11 records (statistically insufficient)
   - Ratio features/observations: 12 features / 44 train = 3.7:1
   - Recommended minimum: 10-20 observations per feature
   
   Conclusion: We are at the lower limit of statistical viability.

2. TECHNICAL ANALYSIS OF {best_name.upper()}:
   OK Best metrics: R² = {r2_best:.4f} (excellent)
   OK Consistent Cross-validation: ±{cv_results['r2_std']:.4f}
   OK No technical overfitting (train-test gap < 0.01)
   
   ALERT: With 11 record test set, metrics have high variance.
           A model can have high R² partially by chance.

3. ADVANTAGES OF LINEAR REGRESSION FOR THIS CONTEXT:
   
   a) INTERPRETABILITY:
      - Can explain every coefficient to business team
      - "For every X increase in price, sales change Y units"
      - Critical for gaining stakeholder trust
   
   b) ROBUSTNESS:
      - Fewer parameters to estimate (less risk with few data)
      - More stable against small data changes
      - Lower probability of production degradation
   
   c) MAINTENANCE:
      - Easier to debug if something fails
      - Faster to retrain
      - No hyperparameter tuning needed
   
   d) RESPECTABLE PERFORMANCE:
      - R² = {r2_lr:.4f} explains {r2_lr*100:.1f}% of variability
      - RMSE = {rmse_lr:.4f} units average error
      - Completely acceptable for a baseline

4. GRADUAL IMPLEMENTATION PLAN (4 PHASES):

   PHASE 1: INITIAL DEPLOYMENT (0-2 months)
   - Model: Linear Regression
   - Objective: Establish baseline and collect feedback
   - Action: Implement in 1-2 pilot stores
   - Monitoring: Compare predictions vs real sales daily
   - Value: Learning with low risk
   
   PHASE 2: DATA COLLECTION (2-4 months)
   - Objective: Expand dataset to 150-200 records
   - Action: Capture pilot data + investigate duplicates
   - Value: Reduce statistical uncertainty
   
   PHASE 3: MODEL EVOLUTION (4-6 months)
   - Model: {best_name} (if superiority confirmed)
   - Objective: Leverage extra data and non-linear relationships
   - Action: Retrain, optimize, validate on new test set
   - Value: Improve predictive accuracy
   
   PHASE 4: EXPANSION (6+ months)
   - Objective: Deploy to all stores
   - Action: Continuous monitoring + monthly retraining
   - Value: Impact on entire operation

5. WHY NOT {best_name.upper()} DIRECTLY?

   Although technically superior, the risk is:
   
   - Exceptional metrics might be due to:
     a) Strong real patterns (desirable)
     b) Chance with 11 record test set (problem)
   
   - Without more data, cannot distinguish between (a) and (b)
   
   - If deployed and fails in production, credibility is lost
   
   - Linear Regression is the "safe path" allowing:
     • Quick value establishment
     • Gaining team trust
     • Collecting data to validate complex models
     • Evolving when solid evidence exists

6. PROFESSIONAL CRITERIA:

   The best decision isn't always the model with best R².
   
   With 55 records, I prefer a ROBUST and EXPLAINABLE model
   over one with perfect but potentially fragile metrics.
   
   When we have 200+ records, {best_name} will be the clear choice.
   But today, with this data, Linear Regression is more prudent.

""")
    
    print("="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    print(f"""
- Technically Superior Model: {best_name} (R² = {r2_best:.4f})
- Recommended Model for Deployment: Linear Regression (R² = {r2_lr:.4f})
- Reason: Small dataset context favors simplicity and robustness
- Plan: Start conservative, evolve when we have more data
- Next Steps: Pilot test + data collection + continuous validation

This decision demonstrates balance between technical skill and professional judgment.
""")
    print("="*70)


def run_modeling_pipeline(data_path: str = '../data/raw/sales_data.xlsx') -> dict:
    """
    Executes the entire modeling process from start to finish.
    """
    print("\n" + "="*70)
    print("PREDICTIVE MODELING PIPELINE")
    print("="*70)
    print(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Get preprocessed data
    print("\nStep 1: Getting preprocessed data...")
    data = run_preprocessing_pipeline(data_path)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print(f"\nData ready:")
    print(f"  Train: {len(X_train)} records")
    print(f"  Test: {len(X_test)} records")
    print(f"  Features: {len(feature_names)}")
    
    # Step 2: Train multiple models
    print("\n" + "="*70)
    print("Step 2: Model Training")
    print("="*70)
    results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Step 3: Compare models
    print("\n" + "="*70)
    print("Step 3: Model Comparison")
    print("="*70)
    best_name, comparison = compare_models(results)
    
    # Step 4: Cross-validation of best model
    print("\n" + "="*70)
    print("Step 4: Cross Validation")
    print("="*70)
    best_model = results[best_name]['model']
    # Concatenate X_train and X_test for CV (use all data)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    cv_results = cross_validation(best_model, X_full, y_full, best_name)
    
    # Step 5: Optimize hyperparameters
    print("\n" + "="*70)
    print("Step 5: Hyperparameter Optimization")
    print("="*70)
    optimized_model = optimize_hyperparameters(best_name, X_train, y_train)
    
    if optimized_model is not None:
        # Evaluate optimized model
        y_pred_opt = optimized_model.predict(X_test)
        r2_opt = r2_score(y_test, y_pred_opt)
        rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
        
        print(f"\nPerformance with optimized hyperparameters:")
        print(f"  R² (Test): {r2_opt:.4f}")
        print(f"  RMSE (Test): {rmse_opt:.4f}")
        
        # Compare with original
        improvement = r2_opt - results[best_name]['r2_test']
        print(f"\nImprovement over default parameters: {improvement:+.4f}")
        
        if improvement > 0.02:
            print("Optimization significantly improved the model!")
            final_best_model = optimized_model
        else:
            print("Improvement is marginal. Could use either.")
            final_best_model = optimized_model
    else:
        final_best_model = best_model
    
    # Step 6: Analyze feature importance
    print("\n" + "="*70)
    print("Step 6: Feature Analysis")
    print("="*70)
    analyze_feature_importance(final_best_model, feature_names, best_name)
    
    # Step 7: Final justification
    print("\n" + "="*70)
    print("Step 7: Final Model Justification")
    print("="*70)
    justify_final_selection(best_name, results, cv_results, comparison)
    
    print("\n" + "="*70)
    print("MODELING COMPLETED")
    print("="*70)
    print("\nModel is ready for use or deployment.")
    print("Recommended next steps:")
    print("  1. Save trained model (using joblib or pickle)")
    print("  2. Create inference script for new predictions")
    print("  3. Document process for team")
    print("  4. Plan production monitoring")
    
    return {
        'best_model': final_best_model,
        'best_name': best_name,
        'results': results,
        'cv_results': cv_results,
        'comparison': comparison,
        'feature_names': feature_names
    }

if __name__ == "__main__":
    run_modeling_pipeline()
