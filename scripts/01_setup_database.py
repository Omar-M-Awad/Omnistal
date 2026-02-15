"""
Setup SQLite database for Omnistal v1.5
Creates database schema and initializes tables
"""

import sys
from pathlib import Path

# Add src to path - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import sqlite3
from config import DB_FILE, DB_TABLE_NAME, PROCESSED_DATA_DIR


def setup_database():
    """Initialize SQLite database with required schema"""
    
    print("=" * 60)
    print("OMNISTAL v1.5 - DATABASE SETUP")
    print("=" * 60)
    
    # Create directory if needed
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Database location: {DB_FILE}")
    
    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create predictions table for storing model outputs
    print("\n🔧 Creating predictions table...")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            employee_id INTEGER,
            prediction_date TEXT,
            predicted_attrition INTEGER,
            attrition_probability REAL,
            risk_score INTEGER,
            risk_level TEXT,
            model_version TEXT,
            PRIMARY KEY (employee_id, prediction_date)
        )
    """)
    
    conn.commit()
    
    print("   ✅ Predictions table created")
    
    # Create indexes
    print("\n🔧 Creating indexes...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_risk_level 
        ON predictions(risk_level)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_date 
        ON predictions(prediction_date)
    """)
    
    conn.commit()
    print("   ✅ Indexes created")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ DATABASE SETUP COMPLETED")
    print("=" * 60)
    print(f"\nDatabase ready at: {DB_FILE}")
    print("\nNext steps:")
    print("  1. Run: python scripts/02_run_etl.py")
    print("  2. Run: python scripts/03_train_model.py")


if __name__ == "__main__":
    setup_database()
