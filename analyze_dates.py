#!/usr/bin/env python3
"""
Analyze dates in flexible data to understand the date range
"""

import pandas as pd

def analyze_dates():
    """Analyze date ranges in the flexible data"""
    df = pd.read_csv('test_flexible_output.csv')
    df['Min_Date'] = pd.to_datetime(df['Min_Date'])
    df['Max_Date'] = pd.to_datetime(df['Max_Date'])

    print('📅 FECHA ANALYSIS')
    print('='*40)
    print(f'Total períodos: {len(df)}')
    print(f'Rango completo: {df["Min_Date"].min()} - {df["Max_Date"].max()}')

    print('\n🔍 ÚLTIMOS 10 PERÍODOS (más recientes):')
    recent = df.head(10)[['Period_Group', 'Min_Date', 'Max_Date', 'NPS_2025', 'NPS_2024', 'Responses']]
    print(recent.to_string())

    print('\n🔍 PRIMEROS 10 PERÍODOS (más antiguos):')
    old = df.tail(10)[['Period_Group', 'Min_Date', 'Max_Date', 'NPS_2025', 'NPS_2024', 'Responses']]
    print(old.to_string())
    
    # Check for 2024-2025 data
    df_2024_2025 = df[(df['Min_Date'] >= '2024-01-01') | (df['Max_Date'] >= '2024-01-01')]
    print(f'\n📊 Períodos con fechas 2024+: {len(df_2024_2025)}')
    
    if len(df_2024_2025) > 0:
        print('Datos 2024+:')
        print(df_2024_2025[['Period_Group', 'Min_Date', 'Max_Date', 'NPS_2025', 'NPS_2024', 'Responses']].to_string())
    
    # Check for NPS_2025 data
    has_2025_data = df['NPS_2025'].notna().sum()
    has_2024_data = df['NPS_2024'].notna().sum()
    print(f'\n📈 Períodos con NPS_2025: {has_2025_data}')
    print(f'📈 Períodos con NPS_2024: {has_2024_data}')

if __name__ == "__main__":
    analyze_dates() 