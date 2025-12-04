import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import handle_duplicates, create_new_features

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe for testing
        self.df_duplicates = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'Store': [1, 1, 2],
            'Category': ['A', 'A', 'B'],
            'Units_Sold': [10, 10, 20],
            'Unit_Price': [5.0, 5.0, 10.0]
        })

        self.df_features = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'Units_Sold': [10, 20],
            'Unit_Price': [5.0, 10.0]
        })

    def test_handle_duplicates(self):
        # Should remove one duplicate row
        df_clean = handle_duplicates(self.df_duplicates)
        self.assertEqual(len(df_clean), 2)
        self.assertEqual(len(df_clean.drop_duplicates()), 2)

    def test_create_new_features(self):
        # Should create Total_Revenue and date features
        df_new = create_new_features(self.df_features)
        
        # Check Total_Revenue
        self.assertIn('Total_Revenue', df_new.columns)
        self.assertEqual(df_new['Total_Revenue'].iloc[0], 50.0)
        
        # Check Date features
        self.assertIn('Month', df_new.columns)
        self.assertIn('DayOfWeek', df_new.columns)
        self.assertIn('Is_Weekend', df_new.columns)
        
        # Check values
        self.assertEqual(df_new['Month'].iloc[0], 1)

if __name__ == '__main__':
    unittest.main()
