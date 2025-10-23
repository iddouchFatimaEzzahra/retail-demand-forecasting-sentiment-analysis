import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta
from model_prophet import (preparer_donnees_prophet, 
                          grid_search_prophet, 
                          evaluer_modele,
                          sauvegarder_metriques_mlflow)
import numpy as np
#import pytest

def test_prophet_updates():
    """Comprehensive smoke test for Prophet model pipeline"""
    print("\n=== STARTING PROPHET SMOKE TEST ===")
    
    # 1. Create realistic test data with all required columns
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    test_df = pd.DataFrame({
        'Date': dates,
        'Product_ID': 'P001',
        'Store_ID': 'S001',
        'Total_Quantity_Sold': [100 + i*2 + np.random.randint(-10, 10) for i in range(30)],
        'Holiday_Flag': [1 if i in [5, 12, 19, 26] else 0 for i in range(30)],
        'Promotional_Flag': [1 if i % 7 == 0 else 0 for i in range(30)]
    })
    
    # 2. Test data preparation
    try:
        print("\nTesting data preparation...")
        prepared_data = preparer_donnees_prophet(test_df)
        assert 'ds' in prepared_data.columns, "Missing 'ds' column"
        assert 'y' in prepared_data.columns, "Missing 'y' column"
        assert not prepared_data['y'].isnull().any(), "Null values in target"
        print("✅ Data preparation successful")
        print(f"Prepared data shape: {prepared_data.shape}")
    except Exception as e:
        print(f"❌ Data preparation failed: {str(e)}")
        raise
    
    # 3. Test grid search with minimal parameters
    try:
        print("\nTesting grid search...")
        best_params = grid_search_prophet(
            prepared_data,
            max_evals=2,  # Minimal testing
            timeout_minutes=1
        )
        required_params = ['changepoint_prior_scale', 'seasonality_prior_scale', 
                         'holidays_prior_scale', 'seasonality_mode']
        for param in required_params:
            assert param in best_params, f"Missing {param} in best_params"
        print("✅ Grid search completed successfully")
        print(f"Best parameters: {best_params}")
    except Exception as e:
        print(f"❌ Grid search failed: {str(e)}")
        raise
    
    # 4. Test model evaluation
    try:
        print("\nTesting model evaluation...")
        y_true = prepared_data['y'].values[-10:]  # Last 10 points as test
        y_pred = y_true + np.random.normal(0, 5, size=len(y_true))  # Simulate predictions
        
        metrics = evaluer_modele(y_true, y_pred, 'P001', 'S001')
        required_metrics = ['rmse', 'mae', 'mape', 'r2']
        for metric in required_metrics:
            assert metric in metrics, f"Missing {metric} in results"
        print("✅ Model evaluation successful")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"❌ Model evaluation failed: {str(e)}")
        raise
    
    # 5. Test MLflow metrics saving (mock if needed)
    try:
        print("\nTesting MLflow metrics saving...")
        sauvegarder_metriques_mlflow(metrics, 'P001', 'S001')
        print("✅ MLflow metrics saving test passed (check MLflow UI for actual results)")
    except Exception as e:
        print(f"❌ MLflow metrics saving failed: {str(e)}")
        raise
    
    print("\n=== ALL SMOKE TESTS PASSED ===")

if __name__ == "__main__":
    test_prophet_updates()