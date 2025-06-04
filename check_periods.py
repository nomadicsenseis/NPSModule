#!/usr/bin/env python3
"""Check which periods have valid NPS_2025 data"""

import pandas as pd

def check_periods():
    df = pd.read_csv('tables/flexible_7d_03_06_2025/Global/flexible_NPS_7d.csv')
    
    print('📊 PERÍODOS CON DATOS NPS_2025:')
    nps_2025_data = df[df['NPS_2025'].notna()].sort_values('Period_Group')
    print(nps_2025_data[['Period_Group', 'Min_Date', 'Max_Date', 'NPS_2025']].to_string())
    
    print(f'\n📈 Rango de períodos con NPS_2025: {nps_2025_data["Period_Group"].min()} a {nps_2025_data["Period_Group"].max()}')
    
    print('\n📊 PERÍODOS CON DATOS NPS_2024:')
    nps_2024_data = df[df['NPS_2024'].notna()].sort_values('Period_Group')
    print(f'Rango de períodos con NPS_2024: {nps_2024_data["Period_Group"].min()} a {nps_2024_data["Period_Group"].max()}')
    
    print('\n🔍 PERÍODO MÁS RECIENTE CON DATOS:')
    most_recent_with_data = df[(df['NPS_2025'].notna()) | (df['NPS_2024'].notna())]['Period_Group'].min()
    print(f'Período más reciente con datos: {most_recent_with_data}')

if __name__ == "__main__":
    check_periods() 