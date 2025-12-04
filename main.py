"""
Retail Sales Prediction - Main Execution Script

This script runs the entire data analysis pipeline:
1. Data Preprocessing
2. Exploratory Data Analysis (Visualization)
3. Predictive Modeling

Usage:
    python main.py
"""

import os
import sys
from src.data_preprocessing import run_preprocessing_pipeline
from src.modeling import run_modeling_pipeline
from src.visualization import generate_eda_figures

def main():
    print("="*80)
    print("RETAIL SALES PREDICTION - PROJECT PIPELINE")
    print("="*80)
    
    # Define paths
    raw_data_path = 'data/raw/sales_data.xlsx'
    figures_dir = 'results/figures'
    
    # Check if data exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Data file not found at {raw_data_path}")
        print("Please ensure the 'data/raw' directory contains 'sales_data.xlsx'")
        return

    # 1. Preprocessing
    print("\n" + "-"*40)
    print("PHASE 1: DATA PREPROCESSING")
    print("-" * 40)
    # Save processed data to data/processed/
    processed_data_path = 'data/processed/sales_data_processed.csv'
    data_artifacts = run_preprocessing_pipeline(raw_data_path, save_path=processed_data_path)
    
    # 2. Visualization
    print("\n" + "-"*40)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (VISUALIZATION)")
    print("-" * 40)
    # We use the original dataframe for EDA as it has the raw values before normalization
    df_original = data_artifacts['df_original']
    # But we need to handle duplicates again for visualization consistency if not done in load
    # (The pipeline does it, but returns processed objects. Let's use the processed df but denormalized? 
    # Actually, visualization module handles raw data usually. Let's pass the cleaned dataframe from preprocessing if available, 
    # or just let visualization module load it. 
    # Better: The visualization module I wrote takes a dataframe. Let's pass the one from preprocessing BEFORE normalization if possible.
    # Looking at run_preprocessing_pipeline, it returns 'df_original' (raw loaded) and 'df_processed' (normalized).
    # We should probably visualize the cleaned data (no duplicates) but with original scales.
    # The 'handle_duplicates' function was called on df. 
    # Let's see... run_preprocessing_pipeline returns 'df_original' which is the result of load_data.
    # It does NOT return the intermediate 'df_clean' (deduplicated but not normalized).
    # I should probably update visualization.py to accept the file path or just use the df_original and let it handle duplicates?
    # Visualization.py has a main block that does: load -> handle_duplicates -> generate.
    # So I can just call generate_eda_figures with a dataframe.
    # I will re-apply handle_duplicates to df_original to be safe and consistent with the analysis.
    
    from src.data_preprocessing import handle_duplicates
    df_clean = handle_duplicates(data_artifacts['df_original'])
    generate_eda_figures(df_clean, output_dir=figures_dir)
    
    # 3. Modeling
    print("\n" + "-"*40)
    print("PHASE 3: PREDICTIVE MODELING")
    print("-" * 40)
    # The modeling pipeline calls preprocessing internally, which is redundant but ensures isolation.
    # We pass the raw data path so it works correctly from root.
    run_modeling_pipeline(raw_data_path)
    
    print("\n" + "="*80)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"1. Data processed and cleaned")
    print(f"2. Figures generated in {figures_dir}/")
    print(f"3. Models trained and evaluated")
    print("="*80)

if __name__ == "__main__":
    main()
