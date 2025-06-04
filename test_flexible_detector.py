#!/usr/bin/env python3
"""
Test FlexibleAnomalyDetector with real 2024-2025 data
"""

from dashboard_analyzer.anomaly_detection.flexible_detector import FlexibleAnomalyDetector
import os
import shutil

def test_flexible_detector():
    """Test the flexible detector with real data"""
    print('🧪 Testing FlexibleAnomalyDetector with Global data...')
    
    # Create test data folder structure
    os.makedirs('test_data/Global', exist_ok=True)
    
    # Copy the test data as Global flexible data
    shutil.copy('test_flexible_output.csv', 'test_data/Global/flexible_NPS_7d.csv')
    
    # Initialize detector
    detector = FlexibleAnomalyDetector(aggregation_days=7, threshold=10.0)
    
    # Run detection
    anomalies, deviations, periods = detector.analyze_flexible_anomalies('test_data')
    
    print(f'\n📊 RESULTS:')
    print(f'Found {len(periods)} periods: {periods[:5]}...')
    print(f'Anomalies: {anomalies}')
    print(f'Deviations: {deviations}')
    
    # Check data quality
    if len(periods) > 0:
        print(f'\n✅ Period range: {periods[-1]} to {periods[0]}')
        print(f'✅ Most recent period: {periods[0]}')
        print(f'✅ Data spans: {len(periods)} periods')
    
    # Cleanup
    shutil.rmtree('test_data', ignore_errors=True)

if __name__ == "__main__":
    test_flexible_detector() 