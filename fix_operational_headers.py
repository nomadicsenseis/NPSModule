import pandas as pd
import glob
from datetime import datetime

def fix_and_check_operational_data():
    """Fix malformed headers and check what dates are actually available"""
    
    print("🔧 FIXING OPERATIONAL DATA HEADERS AND CHECKING DATES")
    print("="*60)
    
    operative_files = glob.glob("tables/**/daily_operative.csv", recursive=True)
    target_dates = ['2025-01-15', '2025-01-20']
    
    # Check one file first to understand the data structure
    if operative_files:
        file_path = operative_files[0]
        print(f"📁 Analyzing first file: {file_path}")
        
        try:
            # Read the raw file to see the actual structure
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                print(f"   📝 Raw header: {first_line}")
            
            # Try to read with pandas
            df = pd.read_csv(file_path)
            print(f"   📊 Pandas columns: {list(df.columns)}")
            
            # Fix the malformed header
            if df.columns[0].startswith('Date_Master['):
                # The header appears to be: "Date_Master[Date,Load_Factor,OTP15_adjusted,Misconex,Mishandling"
                # Missing the closing bracket and the columns are concatenated
                
                # Try different parsing approaches
                print("   🔧 Attempting to fix header...")
                
                # Re-read with custom header
                corrected_columns = ['Date_Master', 'Load_Factor', 'OTP15_adjusted', 'Misconex', 'Mishandling']
                df_fixed = pd.read_csv(file_path, names=corrected_columns, skiprows=1)
                
                print(f"   ✅ Fixed columns: {list(df_fixed.columns)}")
                print(f"   📊 Data shape: {df_fixed.shape}")
                
                # Convert date column
                df_fixed['Date_Master'] = pd.to_datetime(df_fixed['Date_Master']).dt.date
                
                # Check date range
                min_date = df_fixed['Date_Master'].min()
                max_date = df_fixed['Date_Master'].max()
                print(f"   📅 Date range: {min_date} to {max_date}")
                
                # Check for target dates
                for target_date in target_dates:
                    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                    if target_date_obj in df_fixed['Date_Master'].values:
                        print(f"   ✅ Found target date: {target_date}")
                        
                        # Show data for this date
                        day_data = df_fixed[df_fixed['Date_Master'] == target_date_obj]
                        print(f"      📊 Data for {target_date}:")
                        for col in ['Load_Factor', 'OTP15_adjusted', 'Misconex', 'Mishandling']:
                            value = day_data[col].iloc[0] if not day_data[col].empty else 'NaN'
                            print(f"         {col}: {value}")
                    else:
                        print(f"   ❌ Target date NOT found: {target_date}")
                
                # Show sample of available dates
                sample_dates = sorted(df_fixed['Date_Master'].unique())[:10]
                print(f"   📅 Sample available dates: {sample_dates}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    print("1. All operational files have malformed headers that need fixing")
    print("2. Need to update the data loading logic to handle underscore paths")
    print("3. Need to check if target dates (2025-01-15, 2025-01-20) exist in data")
    print("4. May need to adjust date ranges in operational data collection")

if __name__ == "__main__":
    fix_and_check_operational_data() 