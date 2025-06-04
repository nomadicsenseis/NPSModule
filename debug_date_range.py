#!/usr/bin/env python3
"""
Debug script to test date range parsing for Period 1
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def test_date_range_parsing():
    # Test the exact same logic from the interpreter
    data_folder = "tables/flexible_7d_04_06_2025"
    target_period = 1
    aggregation_days = 7
    
    # Load one of the CSV files to get the period mapping
    data_folder_path = Path(data_folder)
    global_folder = data_folder_path / "Global"
    flexible_file = global_folder / f'flexible_NPS_{aggregation_days}d.csv'
    
    if not flexible_file.exists():
        print(f"❌ File not found: {flexible_file}")
        return None
        
    print(f"📂 Loading: {flexible_file}")
    df = pd.read_csv(flexible_file)
    
    # Show the raw data for period 1
    period_data = df[df['Period_Group'] == target_period]
    if period_data.empty:
        print(f"❌ No data found for period {target_period}")
        return None
    
    print("\n📊 Raw data for Period 1:")
    print(period_data[['Period_Group', 'Min_Date', 'Max_Date', 'Responses']].to_string())
    
    # Parse dates
    df['Min_Date'] = pd.to_datetime(df['Min_Date'])
    df['Max_Date'] = pd.to_datetime(df['Max_Date'])
    
    # Get dates for period 1
    period_data = df[df['Period_Group'] == target_period]
    start_date = period_data.iloc[0]['Min_Date']
    end_date = period_data.iloc[0]['Max_Date']
    
    print(f"\n📅 Parsed dates:")
    print(f"   Start: {start_date} (type: {type(start_date)})")
    print(f"   End: {end_date} (type: {type(end_date)})")
    print(f"   Formatted: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Duration: {(end_date - start_date).days + 1} days")
    
    return start_date, end_date

if __name__ == "__main__":
    test_date_range_parsing() 