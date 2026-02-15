"""
Generate Predictions for All Employees
Omnistal v1.5
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import sqlite3
import pandas as pd
from model import AttritionPredictor
from config import DB_FILE, DB_TABLE_NAME, MODEL_FILE


def main():
    """Generate and save predictions for all employees"""
    
    print("\n" + "=" * 60)
    print("OMNISTAL v1.5 - GENERATE PREDICTIONS")
    print("=" * 60)
    
    # Check if model exists
    if not MODEL_FILE.exists():
        print(f"\n❌ Model file not found: {MODEL_FILE}")
        print("   Please run: python scripts/03_train_model.py")
        return
    
    # Load model
    print("\n📥 Loading trained model...")
    predictor = AttritionPredictor(MODEL_FILE)
    
    # Load employee data
    print("📥 Loading employee data...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {DB_TABLE_NAME}", conn)
    
    print(f"   Loaded {len(df):,} employees")
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = predictor.predict(df)
    
    # Prepare predictions table
    today = datetime.now().strftime('%Y-%m-%d')
    predictions_df = pd.DataFrame({
        'employee_id': range(1, len(predictions) + 1),
        'prediction_date': today,
        'predicted_attrition': predictions['PredictedAttrition'],
        'attrition_probability': predictions['AttritionProbability'],
        'risk_score': predictions['RiskScore'],
        'risk_level': predictions['RiskLevel'],
        'model_version': 'v1.5'
    })
    
    # Save predictions to database
    print("💾 Saving predictions to database...")
    predictions_df.to_sql('predictions', conn, if_exists='replace', index=False)
    
    # Also update the main employees table with predictions
    predictions[['PredictedAttrition', 'AttritionProbability', 'RiskScore', 'RiskLevel']].to_sql(
        'employee_predictions',
        conn,
        if_exists='replace',
        index=False
    )
    
    conn.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    risk_counts = predictions['RiskLevel'].value_counts()
    total = len(predictions)
    
    print(f"\nTotal Employees: {total:,}")
    print(f"\nRisk Distribution:")
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        count = risk_counts.get(level, 0)
        pct = (count / total * 100)
        print(f"   {level:8s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nAverage Risk Score: {predictions['RiskScore'].mean():.1f}")
    print(f"Predicted Attrition Rate: {predictions['PredictedAttrition'].mean():.1%}")
    
    # High risk employees
    high_risk = predictions[predictions['RiskLevel'] == 'HIGH']
    if len(high_risk) > 0:
        print(f"\n⚠️  {len(high_risk)} HIGH-RISK EMPLOYEES identified")
        print("\nTop 5 Highest Risk:")
        top_risk = predictions.nlargest(5, 'RiskScore')[
            ['RiskScore', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany']
        ]
        print(top_risk.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ PREDICTIONS GENERATED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"\nPredictions saved to: {DB_FILE}")
    print("\nNext step:")
    print("  Open Power BI dashboard: powerbi/Omnistal_Dashboard.pbix")


if __name__ == "__main__":
    main()
