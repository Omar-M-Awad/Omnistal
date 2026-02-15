"""
Export SQLite Database to Excel
Omnistal v1.5
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import sqlite3
import pandas as pd
from config import DB_FILE, PROCESSED_DATA_DIR


def export_to_excel():
    """Export all database tables to Excel file"""
    
    print("\n" + "=" * 60)
    print("OMNISTAL v1.5 - EXPORT TO EXCEL")
    print("=" * 60)
    
    # Check if database exists
    if not DB_FILE.exists():
        print(f"\n❌ Database file not found: {DB_FILE}")
        print("   Please run scripts 01-04 first")
        return
    
    # Connect to database
    print(f"\n📥 Connecting to: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    
    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"   Found {len(tables)} tables: {', '.join(tables)}")
    
    # Output Excel file path
    excel_file = PROCESSED_DATA_DIR / "omnistal_data.xlsx"
    
    print(f"\n📊 Exporting to Excel: {excel_file}")
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        for table_name in tables:
            print(f"   → Exporting '{table_name}' sheet...")
            
            # Read table
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            
            # Write to Excel sheet
            df.to_excel(writer, sheet_name=table_name[:31], index=False)  # Excel sheet name max 31 chars
            
            print(f"      ✅ {len(df):,} rows exported")
    
    conn.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ EXCEL EXPORT COMPLETED")
    print("=" * 60)
    
    print(f"\nExcel file created: {excel_file}")
    print(f"File size: {excel_file.stat().st_size / 1024:.1f} KB")
    
    print("\n📊 Sheets in Excel file:")
    for table_name in tables:
        conn_temp = sqlite3.connect(DB_FILE)
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn_temp).iloc[0]['cnt']
        conn_temp.close()
        print(f"   • {table_name[:31]:30s}: {count:,} rows")
    
    print("\n🎯 Next steps:")
    print("   1. Open Power BI Desktop")
    print("   2. Get Data → Excel")
    print(f"   3. Browse to: {excel_file}")
    print("   4. Select 'employees' sheet")
    print("   5. Load data")
    
    print("\n💡 TIP: The 'employees' sheet contains all employee data")
    print("   including predictions (RiskScore, RiskLevel, etc.)")


if __name__ == "__main__":
    export_to_excel()
