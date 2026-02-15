"""
Machine Learning Model for Attrition Prediction
Omnistal v1.5
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import (
    MODEL_FILE,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    XGBOOST_PARAMS,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM
)


class AttritionPredictor:
    """
    XGBoost-based employee attrition prediction model
    
    Features:
    - Handles class imbalance with SMOTE
    - Provides risk scores and categorization
    - Calculates feature importance
    - Supports model persistence
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (loads if exists)
        """
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.training_metrics = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for training/prediction
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y) where X is features and y is target (None if no Attrition column)
        """
        # Use configured feature columns that exist in df
        available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        self.feature_columns = available_features
        
        X = df[self.feature_columns].copy()
        
        # Handle target variable if exists
        y = None
        if 'Attrition' in df.columns:
            y = df['Attrition']
        
        return X, y
    
    def train(
        self, 
        df: pd.DataFrame, 
        use_smote: bool = True,
        cross_validate: bool = True
    ) -> Dict:
        """
        Train the attrition prediction model
        
        Args:
            df: Training DataFrame
            use_smote: Whether to use SMOTE for class balancing
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        print("🎯 TRAINING ATTRITION PREDICTION MODEL")
        print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_features(df)
        
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Samples: {len(X)}")
        print(f"   Attrition rate: {y.mean():.1%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y
        )
        
        print(f"\n   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        # Handle class imbalance
        if use_smote:
            print("\n   Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   Balanced samples: {len(X_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train model
        print("\n   Training XGBoost model...")
        self.model = XGBClassifier(**XGBOOST_PARAMS)
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Cross-validation
        if cross_validate:
            print("\n   Running 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train_balanced, y_train_balanced, 
                cv=5, scoring='roc_auc'
            )
            print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Evaluate on test set
        print("\n   Evaluating on test set...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store metrics
        self.training_metrics = {
            'roc_auc': roc_auc,
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'confusion_matrix': conf_matrix.tolist(),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"   ROC-AUC Score: {roc_auc:.4f}")
        print(f"   Accuracy: {class_report['accuracy']:.4f}")
        print(f"   Precision: {class_report['1']['precision']:.4f}")
        print(f"   Recall: {class_report['1']['recall']:.4f}")
        print(f"   F1-Score: {class_report['1']['f1-score']:.4f}")
        
        print("\n   Confusion Matrix:")
        print(f"   TN: {conf_matrix[0][0]:4d}  FP: {conf_matrix[0][1]:4d}")
        print(f"   FN: {conf_matrix[1][0]:4d}  TP: {conf_matrix[1][1]:4d}")
        
        print("\n   Top 5 Important Features:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        print("=" * 60)
        
        return self.training_metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for employees
        
        Args:
            df: DataFrame with employee data
            
        Returns:
            DataFrame with predictions and risk scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        X, _ = self.prepare_features(df)
        
        # Generate predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = df.copy()
        results['PredictedAttrition'] = predictions
        results['AttritionProbability'] = probabilities
        results['RiskScore'] = (probabilities * 100).round(0).astype(int)
        results['RiskLevel'] = results['RiskScore'].apply(self._categorize_risk)
        
        return results
    
    def _categorize_risk(self, score: int) -> str:
        """
        Categorize risk score into levels
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            Risk level (HIGH/MEDIUM/LOW)
        """
        if score >= RISK_THRESHOLD_HIGH:
            return 'HIGH'
        elif score >= RISK_THRESHOLD_MEDIUM:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            top_n: Number of features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: Optional[Path] = None) -> None:
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model (default: from config)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = filepath or MODEL_FILE
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, save_path)
        print(f"\n💾 Model saved to: {save_path}")
    
    def load_model(self, filepath: Optional[Path] = None) -> None:
        """
        Load trained model from file
        
        Args:
            filepath: Path to model file (default: from config)
        """
        load_path = filepath or MODEL_FILE
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        print(f"📥 Loading model from: {load_path}")
        
        model_data = joblib.load(load_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance')
        self.training_metrics = model_data.get('training_metrics')
        
        print("   ✅ Model loaded successfully")


# Example usage
if __name__ == "__main__":
    import sqlite3
    from config import DB_FILE, DB_TABLE_NAME
    
    # Load data from database
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {DB_TABLE_NAME}", conn)
    conn.close()
    
    # Train model
    predictor = AttritionPredictor()
    metrics = predictor.train(df)
    
    # Save model
    predictor.save_model()
    
    # Generate predictions
    predictions = predictor.predict(df)
    print(f"\nGenerated predictions for {len(predictions)} employees")
    print(f"High risk: {(predictions['RiskLevel'] == 'HIGH').sum()}")
    print(f"Medium risk: {(predictions['RiskLevel'] == 'MEDIUM').sum()}")
    print(f"Low risk: {(predictions['RiskLevel'] == 'LOW').sum()}")
