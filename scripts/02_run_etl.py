"""
Run ETL Pipeline
Omnistal v1.5
"""

import sys
from pathlib import Path

# Add src to path - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from etl_pipeline import HRDataPipeline


def main():
    """Execute ETL pipeline"""
    
    pipeline = HRDataPipeline()
    pipeline.run()
    
    # Print summary
    print("\n📊 PIPELINE SUMMARY")
    print("=" * 60)
    summary = pipeline.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    print("\nNext step:")
    print("  Run: python scripts/03_train_model.py")


if __name__ == "__main__":
    main()
