"""
ETL Pipeline for Omnistal v1.5
Extracts, transforms, and loads HR data
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional

from config import (
    RAW_DATA_FILE,
    DB_FILE,
    DB_TABLE_NAME,
    CATEGORICAL_COLUMNS
)
from data_quality import validate_data


class HRDataPipeline:
    """
    Complete ETL pipeline for HR analytics data
    
    Workflow:
    1. Extract: Load raw CSV data
    2. Transform: Clean, encode, engineer features
    3. Load: Save to SQLite database
    """
    
    def __init__(self, raw_file: Optional[Path] = None, db_file: Optional[Path] = None):
        """
        Initialize pipeline
        
        Args:
            raw_file: Path to raw CSV file (default: from config)
            db_file: Path to SQLite database (default: from config)
        """
        self.raw_file = raw_file or RAW_DATA_FILE
        self.db_file = db_file or DB_FILE
        self.df = None
        
    def extract(self) -> pd.DataFrame:
        """
        Extract data from CSV file
        
        Returns:
            Raw DataFrame
        """
        print("📥 EXTRACTING DATA")
        print(f"   Loading from: {self.raw_file}")
        
        if not self.raw_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.raw_file}")
        
        self.df = pd.read_csv(self.raw_file)
        print(f"   ✅ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        return self.df
    
    def transform(self) -> pd.DataFrame:
        """
        Transform data: clean, encode, and engineer features
        
        Returns:
            Transformed DataFrame
        """
        print("\n🔄 TRANSFORMING DATA")
        
        # 1. Validate data quality
        print("   Step 1: Validating data quality...")
        if not validate_data(self.df, verbose=False):
            print("   ⚠️  Quality issues detected - proceeding with caution")
        
        # 2. Remove duplicates
        print("   Step 2: Removing duplicates...")
        original_count = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = original_count - len(self.df)
        if duplicates_removed > 0:
            print(f"   Removed {duplicates_removed} duplicate rows")
        
        # 3. Encode binary variables
        print("   Step 3: Encoding binary variables...")
        binary_mappings = {
            'Attrition': {'Yes': 1, 'No': 0},
            'OverTime': {'Yes': 1, 'No': 0},
            'Gender': {'Male': 1, 'Female': 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
        
        # 4. Encode categorical variables
        print("   Step 4: Encoding categorical variables...")
        for col in CATEGORICAL_COLUMNS:
            if col in self.df.columns and col not in binary_mappings:
                # Simple label encoding for categories
                self.df[f'{col}_Encoded'] = pd.Categorical(self.df[col]).codes
        
        # 5. Feature engineering
        print("   Step 5: Engineering new features...")
        self.df = self._create_features(self.df)
        
        print(f"   ✅ Transformed to {len(self.df.columns)} total features")
        
        return self.df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Tenure-based features
        df['TenureToPromotionRatio'] = (
            df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
        )
        
        df['TenureToRoleRatio'] = (
            df['YearsAtCompany'] / (df['YearsInCurrentRole'] + 1)
        )
        
        # Income-based features
        df['IncomePerYear'] = (
            df['MonthlyIncome'] * 12 / df['YearsAtCompany'].replace(0, 1)
        )
        
        df['IncomeToAgeRatio'] = df['MonthlyIncome'] / df['Age']
        
        # Work-life balance features
        df['BurnoutScore'] = df['OverTime'] * (5 - df['WorkLifeBalance'])
        
        df['SatisfactionAverage'] = (
            df['EnvironmentSatisfaction'] + 
            df['JobSatisfaction'] + 
            df['WorkLifeBalance']
        ) / 3
        
        # Career progression
        df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
        
        # Job Hopping Tendency (FIXED - prevent division by zero and infinity)
        # Use max of (Age - 18) or 1 to avoid division by zero
        df['JobHoppingTendency'] = df['NumCompaniesWorked'] / df['Age'].apply(
            lambda age: max(age - 18, 1)
        )
        
        # Replace any remaining inf/nan with 0 for safety
        df['JobHoppingTendency'] = df['JobHoppingTendency'].replace([float('inf'), float('-inf')], 0)
        df['JobHoppingTendency'] = df['JobHoppingTendency'].fillna(0)
        
        return df
    
    def load(self) -> None:
        """
        Load data into SQLite database
        """
        print("\n💾 LOADING DATA")
        print(f"   Saving to: {self.db_file}")
        
        # Create database directory if it doesn't exist
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_file)
        
        # Save DataFrame to SQL
        self.df.to_sql(
            DB_TABLE_NAME,
            conn,
            if_exists='replace',
            index=False
        )
        
        # Create indexes for better query performance
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_attrition 
            ON {DB_TABLE_NAME}(Attrition)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_department 
            ON {DB_TABLE_NAME}(Department)
        """)
        
        conn.commit()
        conn.close()
        
        print(f"   ✅ Saved {len(self.df):,} rows to database")
    
    def run(self) -> None:
        """
        Execute complete ETL pipeline
        """
        print("=" * 60)
        print("OMNISTAL v1.5 - ETL PIPELINE")
        print("=" * 60)
        
        try:
            # Extract
            self.extract()
            
            # Transform
            self.transform()
            
            # Load
            self.load()
            
            print("\n" + "=" * 60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
        except Exception as e:
            print("\n" + "=" * 60)
            print(f"❌ PIPELINE FAILED: {str(e)}")
            print("=" * 60)
            raise
    
    def get_summary(self) -> dict:
        """
        Get pipeline execution summary
        
        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            return {"status": "Not executed"}
        
        return {
            "total_records": len(self.df),
            "total_features": len(self.df.columns),
            "attrition_rate": f"{(self.df['Attrition'].mean() * 100):.1f}%",
            "missing_values": self.df.isnull().sum().sum(),
            "database_file": str(self.db_file)
        }


# Example usage
if __name__ == "__main__":
    pipeline = HRDataPipeline()
    pipeline.run()
    
    # Print summary
    summary = pipeline.get_summary()
    print("\nPipeline Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
