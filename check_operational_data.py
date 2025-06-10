import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def check_operational_data():
    """Check what operational data files exist and what date ranges they cover"""
    
    print("🔍 OPERATIONAL DATA AVAILABILITY CHECK")
    print("="*60)
    
    # Find all operational data files
    operative_files = glob.glob("tables/**/daily_operative.csv", recursive=True)
    
    if not operative_files:
        print("❌ No operational data files found")
        return
    
    print(f"✅ Found {len(operative_files)} operational data files")
    print()
    
    target_dates = ['2025-01-15', '2025-01-20']
    
    for file_path in sorted(operative_files):
        print(f"📁 File: {file_path}")
        
        # Extract node path from file structure
        path_parts = Path(file_path).parts
        if len(path_parts) >= 3:
            node_folder = path_parts[-2]  # e.g., "Global_LH"
            # Convert underscores to slashes for node path
            node_path = node_folder.replace("_", "/")
            print(f"   🏷️  Node: {node_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the header is malformed (like in the example)
            if df.columns[0].startswith('Date_Master['):
                print(f"   ⚠️  Malformed header detected: {df.columns[0]}")
                # Try to fix the header
                corrected_columns = df.columns[0].replace('Date_Master[', '').split(',')
                corrected_columns[0] = 'Date_Master'
                if len(corrected_columns) == len(df.columns):
                    df.columns = corrected_columns
                    print(f"   🔧 Fixed columns: {list(df.columns)}")
            
            # Convert date column to datetime
            if 'Date_Master' in df.columns:
                df['Date_Master'] = pd.to_datetime(df['Date_Master']).dt.date
                
                # Get date range
                min_date = df['Date_Master'].min()
                max_date = df['Date_Master'].max()
                total_days = len(df)
                
                print(f"   📅 Date range: {min_date} to {max_date} ({total_days} days)")
                
                # Check for our target dates
                target_found = []
                for target_date in target_dates:
                    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                    if target_date_obj in df['Date_Master'].values:
                        target_found.append(target_date)
                
                if target_found:
                    print(f"   ✅ Target dates found: {', '.join(target_found)}")
                else:
                    print(f"   ❌ Target dates NOT found: {', '.join(target_dates)}")
                
                # Show available metrics
                metric_columns = [col for col in df.columns if col != 'Date_Master']
                non_empty_metrics = []
                for col in metric_columns:
                    non_null_count = df[col].notna().sum()
                    if non_null_count > 0:
                        non_empty_metrics.append(f"{col}({non_null_count})")
                
                if non_empty_metrics:
                    print(f"   📊 Available metrics: {', '.join(non_empty_metrics)}")
                else:
                    print(f"   ⚠️  No metric data available")
            else:
                print(f"   ❌ No Date_Master column found. Columns: {list(df.columns)}")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {str(e)}")
        
        print()

def check_path_mapping():
    """Check how the current path mapping works vs actual file structure"""
    print("🗂️  PATH MAPPING CHECK")
    print("="*60)
    
    # Example node paths the interpreter expects
    expected_nodes = [
        "Global/LH",
        "Global/SH", 
        "Global/LH/Business",
        "Global/SH/Economy/YW"
    ]
    
    # Show what the current interpreter expects vs what exists
    for node_path in expected_nodes:
        # Current interpreter path logic
        folder_name = "2025-01-20"  # Example date folder
        expected_path = f"tables/{folder_name}/{node_path}/daily_operative.csv"
        
        # Actual file structure (with underscores)
        actual_folder_pattern = "tables/flight_local_daily_2025_01_20_flexible_1d_*/Global*/daily_operative.csv"
        actual_files = glob.glob(actual_folder_pattern)
        
        underscore_path = node_path.replace("/", "_")
        matching_files = [f for f in actual_files if underscore_path in f]
        
        print(f"🎯 Node: {node_path}")
        print(f"   Expected: {expected_path}")
        print(f"   Actual:   {matching_files[0] if matching_files else 'NOT FOUND'}")
        print()

if __name__ == "__main__":
    check_operational_data()
    print()
    check_path_mapping() 