"""
Unit tests for ML model
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import AttritionPredictor


class TestAttritionPredictor:
    """Test suite for attrition prediction model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample HR data for testing"""
        return pd.DataFrame({
            'Age': [25, 35, 45, 30, 50],
            'MonthlyIncome': [3000, 5000, 7000, 4000, 9000],
            'YearsAtCompany': [2, 5, 10, 3, 15],
            'YearsInCurrentRole': [1, 3, 8, 2, 12],
            'YearsSinceLastPromotion': [1, 2, 5, 1, 8],
            'YearsWithCurrManager': [1, 3, 7, 2, 10],
            'NumCompaniesWorked': [2, 3, 1, 2, 1],
            'DistanceFromHome': [10, 5, 20, 8, 3],
            'PercentSalaryHike': [15, 12, 10, 18, 11],
            'TrainingTimesLastYear': [2, 3, 1, 4, 2],
            'EnvironmentSatisfaction': [3, 4, 2, 4, 3],
            'JobSatisfaction': [3, 4, 2, 3, 4],
            'WorkLifeBalance': [3, 4, 2, 3, 4],
            'JobInvolvement': [3, 4, 2, 3, 4],
            'PerformanceRating': [3, 4, 3, 4, 4],
            'OverTime': [1, 0, 1, 0, 0],
            'Attrition': [1, 0, 1, 0, 0]
        })
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        predictor = AttritionPredictor()
        assert predictor.model is None
        assert predictor.feature_columns is None
    
    def test_feature_preparation(self, sample_data):
        """Test feature preparation"""
        predictor = AttritionPredictor()
        X, y = predictor.prepare_features(sample_data)
        
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert predictor.feature_columns is not None
    
    def test_risk_categorization(self):
        """Test risk score categorization"""
        predictor = AttritionPredictor()
        
        assert predictor._categorize_risk(85) == 'HIGH'
        assert predictor._categorize_risk(55) == 'MEDIUM'
        assert predictor._categorize_risk(25) == 'LOW'
    
    def test_prediction_output_format(self, sample_data):
        """Test prediction returns correct format"""
        predictor = AttritionPredictor()
        predictor.train(sample_data)
        
        predictions = predictor.predict(sample_data)
        
        # Check required columns exist
        required_cols = ['PredictedAttrition', 'AttritionProbability', 
                        'RiskScore', 'RiskLevel']
        for col in required_cols:
            assert col in predictions.columns
        
        # Check value ranges
        assert predictions['AttritionProbability'].between(0, 1).all()
        assert predictions['RiskScore'].between(0, 100).all()
        assert predictions['RiskLevel'].isin(['LOW', 'MEDIUM', 'HIGH']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
