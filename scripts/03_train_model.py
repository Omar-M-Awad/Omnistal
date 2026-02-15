"""
Train Machine Learning Model
Omnistal v1.5
"""

import sys
from pathlib import Path

# Add src to path - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import sqlite3
import pandas as pd
from model import AttritionPredictor
from config import DB_FILE, DB_TABLE_NAME


def main():
    """Train and save attrition prediction model"""
    
    print("\n" + "=" * 60)
    print("OMNISTAL v1.5 - MODEL TRAINING")
    print("=" * 60)
    
    # Load data from database
    print("\n📥 Loading data from database...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {DB_TABLE_NAME}", conn)
    conn.close()
    
    print(f"   Loaded {len(df):,} employee records")
    
    # Initialize predictor
    predictor = AttritionPredictor()
    
    # Train model
    metrics = predictor.train(df, use_smote=True, cross_validate=True)
    
    # Save model
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED")
    print("=" * 60)
    
    print("\nModel Performance Summary:")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Precision: {metrics['precision']:.1%}")
    print(f"   Recall: {metrics['recall']:.1%}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\nNext step:")
    print("  Run: python scripts/04_generate_predictions.py")


if __name__ == "__main__":
    main()
