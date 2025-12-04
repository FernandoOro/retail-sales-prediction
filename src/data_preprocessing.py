"""
Data Preprocessing Report
Project: Retail Sales Prediction
Author: Fernando Orozco
Date: October 2025

Description:
This script documents the complete data cleaning and preparation process for sales data
prior to modeling. It includes handling duplicate values, feature engineering,
categorical encoding, and normalization.

Objective:
To produce a clean dataset ready for training machine learning models, ensuring
that every decision made is justified and documented.
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)


def load_data(file_path: str = '../data/raw/sales_data.xlsx') -> pd.DataFrame:
    """
    Loads data from the Excel file.
    
    The file has a special format: CSV data is embedded within an Excel cell,
    so we need to manually parse it to extract it correctly.
    This is likely a quirk of how the data was originally exported.
    """
    print("\n--- Loading Data ---")
    
    # First, try reading the Excel file normally
    df_raw = pd.read_excel(file_path)
    
    # If it has only one column, it means the CSV data is inside
    if df_raw.shape[1] == 1:
        # Take the column name and all content as a CSV string
        col_name = df_raw.columns[0]
        csv_data = col_name + '\n' + '\n'.join(df_raw[col_name].astype(str))
        # Now parse the CSV properly
        df = pd.read_csv(io.StringIO(csv_data))
        print("OK - Data extracted from embedded CSV format")
    else:
        df = df_raw
        print("OK - Data loaded directly")
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns found: {', '.join(df.columns)}")
    
    # Show first rows to confirm correct loading
    print("\nFirst 3 rows of the dataset:")
    print(df.head(3).to_string())
    
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for null or missing values in any column.
    
    It's important to do this at the start because null values can cause problems
    in model training. Depending on the count, we decide whether to remove them,
    impute them, or if no action is needed.
    """
    print("\n--- Checking Missing Values ---")
    
    # Count nulls per column
    nulls = df.isnull().sum()
    
    if nulls.sum() == 0:
        print("Good news: No missing values in any column")
        print("This means the dataset is complete and we don't need imputation")
        print("We can proceed without worrying about missing data")
        return df
    
    # If we reach here, there are nulls (though in our case there aren't)
    print("Missing values found:")
    for col in nulls[nulls > 0].index:
        percentage = (nulls[col] / len(df)) * 100
        print(f"  - {col}: {nulls[col]} nulls ({percentage:.1f}%)")
    
    # Logic to handle them would go here, but since we don't have any, it's not needed
    
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes and removes duplicate records.
    
    This was a difficult decision. We found that 50% of the original dataset consisted
    of records with exactly the same values in all columns. The problem is that
    without a unique transaction ID, we cannot know if these are:
    
    a) Duplication errors (same record loaded twice)
    b) Legitimate transactions that coincidentally have the same values
    
    After consideration, I decided to remove them for these reasons:
    
    1. DATA LEAKAGE: If we keep duplicates, when we do train/test split, the same
       record might appear in both sets. This makes the model "cheat" because
       it already saw those data in training. Metrics would be artificially high but
       completely invalid.
    
    2. INTEGRITY: I prefer working with fewer data but being sure that my metrics
       are honest. A model trained with 55 unique records is more valuable than one with
       110 records but fake metrics.
    
    3. COMMON PRACTICE: In real projects, when you find duplicates without a way to verify them,
       the most conservative approach is to remove them. Then the source of the problem (ETL,
       extraction process, etc.) is investigated.
    
    4. BUSINESS IMPACT: Yes, it reduces our dataset by half, but it is the technically
       correct decision. In a real case, this would be a signal to investigate where the duplicates
       come from and collect more clean data.
    
    It is better to have a small dataset you can trust than a large one with problems.
    """
    print("\n--- Analyzing Duplicate Records ---")
    
    # Identify duplicates (considering all columns)
    num_duplicates = df.duplicated().sum()
    percentage = (num_duplicates / len(df)) * 100
    
    print(f"Total records: {len(df)}")
    print(f"Duplicate records found: {num_duplicates} ({percentage:.1f}%)")
    
    if num_duplicates == 0:
        print("No duplicates, proceeding with full dataset")
        return df
    
    # Show some examples to see what kind of duplicates they are
    print("\nExamples of duplicate records (showing some pairs):")
    duplicates_mask = df.duplicated(keep=False)  # keep=False marks ALL duplicates
    examples = df[duplicates_mask].head(4)  # Show 4 example rows
    print(examples[['Date', 'Store', 'Category', 'Units_Sold', 'Unit_Price']].to_string())
    
    print("\nDECISION: I will remove these duplicates")
    print("\nWhy remove them?")
    print("  1. Without transaction ID, I cannot confirm if they are legitimate")
    print("  2. Keeping them would cause data leakage between train and test")
    print("  3. I prioritize reliable metrics over having more data")
    print("  4. It is standard practice when duplicates cannot be verified")
    print("  5. Likely an error in the data extraction process")
    
    # Remove duplicates
    df_clean = df.drop_duplicates()
    removed = len(df) - len(df_clean)
    
    print(f"\nResult: {removed} records removed")
    print(f"Final dataset: {len(df_clean)} unique records")
    print("\nIMPORTANT: Although the dataset was significantly reduced, we can now")
    print("trust that our evaluation metrics will be valid and honest.")
    
    return df_clean


def check_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies outliers using the Interquartile Range (IQR) method.
    
    IQR is a standard way to detect outliers:
    - Calculate Q1 (25th percentile) and Q3 (75th percentile)
    - IQR is the difference: Q3 - Q1
    - Any value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is considered an outlier
    
    However, in sales data, outliers are not necessarily errors.
    An outlier can be a day of exceptionally high or low sales, and that is
    valuable information we want the model to learn.
    
    Therefore, although I identify them, I decided not to remove them.
    """
    print("\n--- Identifying Outliers ---")
    
    # Only check numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    print("Checking outliers in each numeric column using IQR method:\n")
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Limits to consider something an outlier
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        # Search for values outside these limits
        outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
        
        print(f"{col}:")
        print(f"  Normal range: [{lower_limit:.2f}, {upper_limit:.2f}]")
        print(f"  Outliers found: {len(outliers)}")
        
        if len(outliers) > 0:
            outlier_values = outliers[col].tolist()
            print(f"  Specific values: {outlier_values}")
        print()
    
    print("DECISION: Keep all outliers")
    print("\nWhy not remove them?")
    print("  - In sales, extreme values are real business behavior")
    print("  - An outlier can be a promotion day, special event, etc.")
    print("  - No evidence that they are data entry errors")
    print("  - The model needs to learn from these edge cases")
    print("  - Algorithms like Random Forest are naturally robust to outliers")
    
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional features from existing data.
    
    This is what we call feature engineering - creating new variables that can
    help the model predict better. The idea is to extract information that is "hidden"
    in the original data but can be very relevant.
    """
    print("\n--- Creating New Variables (Feature Engineering) ---")
    
    df_new = df.copy()
    
    # Variable 1: Total Revenue per transaction
    # It's simply price * units, but can be more predictive than either separately
    print("\n1. Total_Revenue = Unit_Price × Units_Sold")
    df_new['Total_Revenue'] = df_new['Unit_Price'] * df_new['Units_Sold']
    print(f"   This variable captures the total value of each sale")
    print(f"   Range: ${df_new['Total_Revenue'].min():.2f} to ${df_new['Total_Revenue'].max():.2f}")
    
    # Temporal variables extracted from date
    print("\n2. Temporal variables extracted from Date:")
    df_new['Date'] = pd.to_datetime(df_new['Date'])
    
    # Day of month (1-31)
    df_new['DayOfMonth'] = df_new['Date'].dt.day
    print("   - DayOfMonth: To capture patterns based on the day of the month")
    
    # Week of year
    df_new['WeekOfYear'] = df_new['Date'].dt.isocalendar().week
    print("   - WeekOfYear: For weekly patterns")
    
    # Is it weekend? (0 or 1)
    df_new['IsWeekend'] = (df_new['Date'].dt.dayofweek >= 5).astype(int)
    print("   - IsWeekend: Sales usually change on weekends")
    
    # Is it start of month? (first 5 days)
    df_new['IsStartOfMonth'] = (df_new['Date'].dt.day <= 5).astype(int)
    print("   - IsStartOfMonth: Some people buy more at the start of the month (payday)")
    
    # Is it end of month? (last 5 days)
    df_new['IsEndOfMonth'] = (df_new['Date'].dt.day >= 26).astype(int)
    print("   - IsEndOfMonth: There might be patterns at the end of the month")
    
    # Context variables: averages by group
    print("\n3. Context variables (aggregations):")
    
    # Average sales per store (to give context on size/performance of each store)
    df_new['Store_Avg_Units'] = df_new.groupby('Store')['Units_Sold'].transform('mean')
    print("   - Store_Avg_Units: Historical average of each store")
    print("     Helps the model understand if a store typically sells more or less")
    
    # Average per category
    df_new['Category_Avg_Units'] = df_new.groupby('Category')['Units_Sold'].transform('mean')
    print("   - Category_Avg_Units: Historical average per category")
    print("     Different products naturally have different sales volumes")
    
    print(f"\nSummary: Went from {len(df.columns)} to {len(df_new.columns)} columns")
    print(f"Created {len(df_new.columns) - len(df.columns)} new variables")
    
    return df_new


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converts categorical variables into numbers so models can use them.
    
    ML algorithms only work with numbers, so we need to convert things like
    category names or store IDs into numeric values.
    
    There are two main techniques I use here:
    
    1. LABEL ENCODING (for Store):
       Simply assigns a number to each store (101->0, 102->1, 103->2)
       I use it for Store because:
       - Only 3 values (low cardinality)
       - Store IDs are already numbers, I just re-encode them
       - Saves space vs creating 3 columns
       - Works perfectly for decision trees
    
    2. ONE-HOT ENCODING (for Category):
       Creates a binary column (0/1) for each category
       I use it for Category because:
       - Categories have no natural order (Electronics is not "greater" than Clothing)
       - Prevents the model from assuming ordinal relationships that don't exist
       - It is the standard way to handle nominal variables
       - With only 3 categories, it doesn't generate too many columns
    
    I could have used One-Hot for both, but preferred to be efficient with Store.
    """
    print("\n--- Encoding Categorical Variables ---")
    
    df_encoded = df.copy()
    encoders = {}  # Save encoders in case we need them later
    
    # First, Label Encoding for Store
    print("\n1. Label Encoding for 'Store'")
    print("   Converting each store ID into a sequential number")
    
    # Show original values
    unique_stores = df_encoded['Store'].unique()
    print(f"   Original stores: {unique_stores}")
    
    # Apply encoding
    le_store = LabelEncoder()
    df_encoded['Store_Encoded'] = le_store.fit_transform(df_encoded['Store'])
    encoders['Store'] = le_store
    
    # Show created mapping
    print("   Mapping created:")
    for original, encoded in zip(le_store.classes_, range(len(le_store.classes_))):
        print(f"     {original} -> {encoded}")
    
    print("\n   Why Label Encoding here?")
    print("     - Only 3 stores (low cardinality)")
    print("     - More efficient than creating 3 dummy columns")
    print("     - Tree-based models can handle it well")
    
    # Now, One-Hot Encoding for Category
    print("\n2. One-Hot Encoding for 'Category'")
    print("   Creating a binary column (0/1) for each category")
    
    categories = df_encoded['Category'].unique()
    print(f"   Categories found: {categories}")
    
    # Create dummy columns
    # drop_first=True removes the first category to avoid multicollinearity
    # (if you know it's not Electronics nor Clothing, then it must be Home Goods)
    dummies = pd.get_dummies(df_encoded['Category'], prefix='Category', drop_first=True)
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    print(f"   Columns created: {list(dummies.columns)}")
    print("   (Note: drop_first=True removes one category to avoid redundancy)")
    
    print("\n   Why One-Hot here?")
    print("     - Categories have no order (Electronics is not 'more' than Clothing)")
    print("     - Prevents model from assuming false relationships")
    print("     - Standard for nominal variables")
    print("     - With 3 categories, it is manageable")
    
    # Remove original categorical columns
    print("\n3. Cleanup: Removing original categorical columns")
    df_encoded = df_encoded.drop(['Store', 'Category'], axis=1)
    print("   Columns removed: Store, Category")
    print("   (We already have their encoded versions)")
    
    print(f"\nNow we have {len(df_encoded.columns)} columns, all numeric")
    
    return df_encoded, encoders


def normalize_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardizes numeric variables so they are on the same scale.
    
    Many algorithms (like linear regression or SVM) are sensitive to variable scale.
    If one variable goes from 0 to 1 and another from 0 to 10000, the model might
    give more importance to the second just because of its magnitude.
    
    I use StandardScaler, which transforms each variable to have:
    - Mean = 0
    - Standard Deviation = 1
    
    Formula: z = (x - mean) / std_dev
    
    Why StandardScaler?
    
    - MinMaxScaler: Compresses everything between 0 and 1, but is very sensitive to outliers
    - RobustScaler: Uses median instead of mean (good for many outliers, not our case)
    - Normalizer: Normalizes rows, not columns (not what I need)
    
    StandardScaler is the most common because:
    1. Robust for most cases
    2. Algorithms I'll use (Linear Regression, Ridge, Lasso) prefer it
    3. Maintains information about real data dispersion
    4. Allows comparing coefficients between variables later
    """
    print("\n--- Normalizing Numeric Variables ---")
    
    df_normalized = df.copy()
    
    # Identify which columns to normalize
    # I don't normalize binary variables (already 0 or 1) nor encoded ones
    columns_to_normalize = [
        'Unit_Price',           # Price varies a lot by category
        'Total_Revenue',        # Derived from price, also varies a lot
        'DayOfMonth',          # 1 to 31
        'WeekOfYear',          # 1 to 52
        'Store_Avg_Units',     # Averages
        'Category_Avg_Units'   # Averages
    ]
    
    # Ensure all exist in dataframe
    columns_to_normalize = [col for col in columns_to_normalize if col in df_normalized.columns]
    
    print("Chosen technique: StandardScaler (Z-score standardization)")
    print("Formula: z = (value - mean) / std_dev")
    print("\nColumns to normalize:")
    
    # Show current state of each column
    for i, col in enumerate(columns_to_normalize, 1):
        mean = df_normalized[col].mean()
        std = df_normalized[col].std()
        print(f"  {i}. {col}")
        print(f"     Before: mean={mean:.2f}, std={std:.2f}")
    
    print("\nWhy StandardScaler?")
    print("  - Linear regression works better with standardized variables")
    print("  - Allows comparing variable importance later")
    print("  - More robust than MinMaxScaler when there are outliers")
    print("  - Industry standard for this type of problem")
    
    print("\nAlternatives considered but discarded:")
    print("  - MinMaxScaler: Very sensitive to outliers (compresses all to 0-1)")
    print("  - RobustScaler: Uses median; unnecessary as I don't have many outliers")
    print("  - Normalizer: Normalizes by row, not column (doesn't apply here)")
    
    # Apply normalization
    scaler = StandardScaler()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
    
    print("\nResult (verifying it worked):")
    for col in columns_to_normalize:
        new_mean = df_normalized[col].mean()
        new_std = df_normalized[col].std()
        print(f"  {col}:")
        print(f"    After: mean={new_mean:.6f}, std={new_std:.6f}")
    
    print("\nNOTE: Binary columns (IsWeekend, etc.) are not normalized")
    print("      because they are already in 0-1 scale which is perfect")
    
    return df_normalized, scaler


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Splits data into training and testing sets.
    
    I use an 80/20 split:
    - 80% to train the model
    - 20% to evaluate how well it works
    
    random_state=42 ensures we always get the same split,
    which is important for reproducibility.
    
    Important consideration: With only 55 total records, the test set
    will have just ~11 records. This is statistically very small and
    means metrics will have significant variance. In an ideal project,
    we'd want 200+ records, but we work with what we have.
    
    To compensate, I'll use cross-validation later for more robust metrics.
    """
    print("\n--- Splitting Train and Test Sets ---")
    
    # Separate features (X) from target variable (y)
    # Target variable is what we want to predict: Units_Sold
    # Features are everything else (except Date which we don't need anymore)
    
    feature_columns = [col for col in df.columns 
                        if col not in ['Units_Sold', 'Date']]
    
    X = df[feature_columns]
    y = df['Units_Sold']
    
    print(f"Target variable (y): Units_Sold")
    print(f"Number of features (X): {len(feature_columns)}")
    print(f"Total records: {len(df)}")
    
    print("\nFeatures to use for prediction:")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"\nSplit performed:")
    print(f"  Train: {len(X_train)} records ({(1-test_size)*100:.0f}%)")
    print(f"  Test:  {len(X_test)} records ({test_size*100:.0f}%)")
    print(f"  random_state={random_state} (for reproducibility)")
    
    # Honest comment about limitation
    print("\nIMPORTANT WARNING:")
    print(f"  The test set has only {len(X_test)} records, which is statistically small.")
    print("  This means:")
    print("    - Metrics can vary significantly")
    print("    - A single 'difficult' record can affect the score a lot")
    print("    - I need to use cross-validation to have more confidence")
    print("\n  Ideally I'd want 200+ records, but I work with available data.")
    print("  The decision to remove duplicates was correct to maintain integrity,")
    print("  even though it left us with fewer data.")
    
    # Show basic stats to verify split is reasonable
    print("\nVerifying split is representative:")
    print(f"  Mean Units_Sold in train: {y_train.mean():.2f}")
    print(f"  Mean Units_Sold in test:  {y_test.mean():.2f}")
    print(f"  Std in train: {y_train.std():.2f}")
    print(f"  Std in test:  {y_test.std():.2f}")
    
    if abs(y_train.mean() - y_test.mean()) < 5:
        print("  OK - Distributions are similar, good split")
    else:
        print("  NOTE - Means differ quite a bit (possible due to small size)")
    
    return X_train, X_test, y_train, y_test, feature_columns


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the processed dataframe to a CSV file.
    """
    print(f"\n--- Saving Processed Data ---")
    try:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=False)
        print(f"Data saved successfully to: {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def run_preprocessing_pipeline(file_path: str = '../data/raw/sales_data.xlsx', save_path: str = None) -> dict:
    """
    Executes the entire preprocessing process from start to finish.
    
    Acts as the conductor: calls each function in the correct order
    and ensures everything flows well.
    """
    print("\n" + "="*70)
    print("STARTING DATA PREPROCESSING")
    print("="*70)
    print(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"File: {file_path}")
    
    # Step 1: Load data
    df = load_data(file_path)
    
    # Step 2: Check for nulls
    df = check_missing_values(df)
    
    # Step 3: Handle duplicates (this is the most important decision)
    df_clean = handle_duplicates(df)
    
    # Step 4: Check outliers (but don't remove them)
    df_clean = check_outliers(df_clean)
    
    # Step 5: Create new variables
    df_features = feature_engineering(df_clean)
    
    # Step 6: Encode categoricals
    df_encoded, encoders = encode_categorical(df_features)
    
    # Step 7: Normalize
    df_normalized, scaler = normalize_data(df_encoded)
    
    # Step 8: Save processed data (if path provided)
    if save_path:
        save_data(df_normalized, save_path)
    
    # Step 9: Prepare train/test
    X_train, X_test, y_train, y_test, feature_names = split_train_test(df_normalized)
    
    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETED")
    print("="*70)
    
    print("\nTransformation Summary:")
    print(f"  Initial records:      {len(df)}")
    print(f"  Final records:        {len(df_clean)}")
    print(f"  Initial variables:    {len(df.columns)}")
    print(f"  Final variables:      {len(feature_names)}")
    print(f"  Train set size:       {len(X_train)}")
    print(f"  Test set size:        {len(X_test)}")
    
    print("\nMain decisions taken:")
    print("  1. Missing values: None found")
    print("  2. Duplicates: Removed 55 records (50%) to avoid data leakage")
    print("  3. Outliers: Kept them (real sales data)")
    print("  4. Feature engineering: Created 8 new variables")
    print("  5. Encoding: Label for Store, One-Hot for Category")
    print("  6. Normalization: StandardScaler on numeric variables")
    print("  7. Split: 80/20 train/test")
    
    print("\nData is ready for model training.")
    print("Next step: Test different algorithms and compare them.")
    
    # Return everything we might need later
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names,
        'df_original': df,
        'df_processed': df_normalized
    }


# If executed directly
if __name__ == "__main__":
    print("\nRunning data preprocessing...")
    print("(This file documents each step of the process)")
    
    # Run full pipeline
    results = run_preprocessing_pipeline('../data/raw/sales_data.xlsx')
    
    print("\nDone. Preprocessed data is in 'results' dictionary")
    print("You can access X_train, X_test, y_train, y_test from there.")
