#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')
from dashboard_analyzer.enhanced_flexible_main import show_all_anomaly_periods_with_explanations, run_flexible_analysis_silent

async def test_single_period():
    # First run the analysis silently to get data
    data_folder = 'tables/flight_local_daily_2025_01_20_flexible_1d_Global_LH_1413'
    analysis_data = await run_flexible_analysis_silent(data_folder)
    
    # Then show just the first period with explanations
    if analysis_data:
        print('=== Testing Single Period Analysis ===')
        # Modify to show only first period
        original_periods = analysis_data['anomaly_periods']
        analysis_data['anomaly_periods'] = original_periods[:1]  # Just first period
        
        await show_all_anomaly_periods_with_explanations(analysis_data, 'Global/LH')

if __name__ == "__main__":
    asyncio.run(test_single_period()) 