"""
Data quality validation for Omnistal v1.5
Ensures data meets quality standards before processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataQualityChecker:
    """Validates data quality and returns detailed report"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        
    def check_missing_values(self) -> bool:
        """Check for missing values in critical columns"""
        missing = self.df.isnull().sum()
        if missing.any():
            self.issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
            return False
        return True
    
    def check_duplicates(self) -> bool:
        """Check for duplicate employee records"""
        if 'EmployeeNumber' in self.df.columns:
            duplicates = self.df['EmployeeNumber'].duplicated().sum()
            if duplicates > 0:
                self.issues.append(f"Found {duplicates} duplicate employee records")
                return False
        return True
    
    def check_data_types(self) -> bool:
        """Validate data types for key columns"""
        type_checks = {
            'Age': (int, float),
            'MonthlyIncome': (int, float),
            'Attrition': (str, object)
        }
        
        for col, expected_types in type_checks.items():
            if col in self.df.columns:
                if not self.df[col].dtype in expected_types:
                    self.issues.append(f"Column {col} has unexpected type: {self.df[col].dtype}")
                    return False
        return True
    
    def check_value_ranges(self) -> bool:
        """Check if values are within expected ranges"""
        checks_passed = True
        
        # Age should be between 18 and 100
        if 'Age' in self.df.columns:
            if not self.df['Age'].between(18, 100).all():
                self.issues.append("Age values outside expected range (18-100)")
                checks_passed = False
        
        # MonthlyIncome should be positive
        if 'MonthlyIncome' in self.df.columns:
            if not (self.df['MonthlyIncome'] > 0).all():
                self.issues.append("MonthlyIncome contains non-positive values")
                checks_passed = False
        
        # YearsAtCompany should be non-negative
        if 'YearsAtCompany' in self.df.columns:
            if not (self.df['YearsAtCompany'] >= 0).all():
                self.issues.append("YearsAtCompany contains negative values")
                checks_passed = False
        
        return checks_passed
    
    def check_categorical_values(self) -> bool:
        """Validate categorical column values"""
        checks_passed = True
        
        # Attrition should only be Yes/No
        if 'Attrition' in self.df.columns:
            valid_values = {'Yes', 'No'}
            if not set(self.df['Attrition'].unique()).issubset(valid_values):
                self.issues.append(f"Attrition has invalid values: {self.df['Attrition'].unique()}")
                checks_passed = False
        
        # Gender should only be Male/Female
        if 'Gender' in self.df.columns:
            valid_values = {'Male', 'Female'}
            if not set(self.df['Gender'].unique()).issubset(valid_values):
                self.issues.append(f"Gender has invalid values: {self.df['Gender'].unique()}")
                checks_passed = False
        
        return checks_passed
    
    def check_logical_consistency(self) -> bool:
        """Check for logical inconsistencies in data"""
        checks_passed = True
        
        # YearsInCurrentRole should not exceed YearsAtCompany
        if all(col in self.df.columns for col in ['YearsInCurrentRole', 'YearsAtCompany']):
            inconsistent = self.df['YearsInCurrentRole'] > self.df['YearsAtCompany']
            if inconsistent.any():
                count = inconsistent.sum()
                self.issues.append(f"{count} records have YearsInCurrentRole > YearsAtCompany")
                checks_passed = False
        
        return checks_passed
    
    def run_all_checks(self) -> Tuple[bool, List[str]]:
        """
        Run all data quality checks
        
        Returns:
            Tuple of (all_passed, list_of_issues)
        """
        checks = [
            self.check_missing_values(),
            self.check_duplicates(),
            self.check_data_types(),
            self.check_value_ranges(),
            self.check_categorical_values(),
            self.check_logical_consistency()
        ]
        
        all_passed = all(checks)
        
        return all_passed, self.issues
    
    def get_quality_report(self) -> Dict:
        """
        Generate comprehensive quality report
        
        Returns:
            Dictionary with quality metrics
        """
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }


def validate_data(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Convenience function to validate data quality
    
    Args:
        df: DataFrame to validate
        verbose: Print detailed report
        
    Returns:
        True if all checks pass, False otherwise
    """
    checker = DataQualityChecker(df)
    passed, issues = checker.run_all_checks()
    
    if verbose:
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        report = checker.get_quality_report()
        print(f"Total Rows: {report['total_rows']:,}")
        print(f"Total Columns: {report['total_columns']}")
        print(f"Missing Values: {report['missing_values']}")
        print(f"Duplicate Rows: {report['duplicate_rows']}")
        print(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")
        print(f"Numeric Columns: {report['numeric_columns']}")
        print(f"Categorical Columns: {report['categorical_columns']}")
        
        print("\n" + "=" * 60)
        if passed:
            print("✅ ALL DATA QUALITY CHECKS PASSED")
        else:
            print("❌ DATA QUALITY ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        print("=" * 60)
    
    return passed


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'EmployeeNumber': [1, 2, 3],
        'Age': [25, 35, 45],
        'MonthlyIncome': [3000, 5000, 7000],
        'YearsAtCompany': [2, 5, 10],
        'YearsInCurrentRole': [1, 3, 8],
        'Attrition': ['No', 'No', 'Yes'],
        'Gender': ['Male', 'Female', 'Male']
    })
    
    validate_data(sample_data)
