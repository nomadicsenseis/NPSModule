#!/usr/bin/env python3
"""
MAIN NPS ANOMALY DETECTION SYSTEM
=================================
Complete automated system that:
1. Downloads data from Power BI for all nodes
2. Saves data in folders named by current date
3. Runs anomaly detection analysis for last 7 days
4. Shows collapsed tree view for each day

Usage: python main.py [--date YYYY-MM-DD]
If no date specified, uses today's date
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import os

# Import our modules
from dashboard_analyzer.data_collection import PBIDataCollector
from dashboard_analyzer.anomaly_detection.anomaly_tree import AnomalyTree
from dashboard_analyzer.anomaly_detection.anomaly_interpreter import AnomalyInterpreter
from dashboard_analyzer.anomaly_explanation import OperationalDataAnalyzer

def create_date_folder(target_date: str = None) -> str:
    """Create folder name from date (DD_MM_YYYY format)"""
    if target_date:
        # Parse provided date and convert to folder format
        dt = datetime.strptime(target_date, '%Y-%m-%d')
        folder_name = dt.strftime('%d_%m_%Y')
    else:
        # Use today's date
        folder_name = datetime.now().strftime('%d_%m_%Y')
    
    return folder_name

def download_all_data(date_folder: str) -> bool:
    """Download all PBI data and save to date folder"""
    print('📡 DOWNLOADING DATA FROM POWER BI')
    print('=' * 60)
    
    try:
        # Initialize collector
        collector = PBIDataCollector()
        print(f'✅ Connected to Power BI (Tenant: {collector.tenant_id})')

        # Set output directory
        output_base = 'tables'
        base_path = Path(output_base) / date_folder
        
        # Create base directory
        if not base_path.exists():
            os.makedirs(base_path, exist_ok=True)
            print(f'📂 Created directory: {base_path}')
        
        print(f'📁 Output directory: {base_path}')

        # Define all node paths (correct structure from README)
        node_paths = [
            'Global',
            'Global/LH',
            'Global/LH/Economy',
            'Global/LH/Business', 
            'Global/LH/Premium',
            'Global/SH',
            'Global/SH/Economy',
            'Global/SH/Economy/IB',
            'Global/SH/Economy/YW',
            'Global/SH/Business',
            'Global/SH/Business/IB',
            'Global/SH/Business/YW'
        ]
        
        print(f'🌳 Downloading data for {len(node_paths)} nodes...')
        
        # Collect data for each node
        all_results = {}
        for node_path in node_paths:
            all_results[node_path] = collector.collect_node_data(node_path, base_path)

        # Summary report
        successful_files = sum(
            sum(1 for success in results.values() if success) 
            for results in all_results.values()
        )
        total_files = sum(len(results) for results in all_results.values())
        
        print(f'\n📊 DOWNLOAD SUMMARY:')
        print(f'   Files: {successful_files}/{total_files} successful')
        print(f'   Success Rate: {(successful_files/total_files)*100:.1f}%')
        
        return successful_files > 0
        
    except Exception as e:
        print(f'❌ Download Error: {e}')
        return False

def run_anomaly_analysis(date_folder: str) -> bool:
    """Run complete anomaly detection analysis"""
    print('\n🔍 RUNNING ANOMALY DETECTION ANALYSIS')
    print('=' * 60)
    
    try:
        # Create anomaly tree and interpreter
        tree = AnomalyTree(data_base_path="tables")
        interpreter = AnomalyInterpreter()
        
        # Run basic analysis
        tree.run_full_analysis(date_folder)
        
        if not tree.dates:
            print("❌ No data available for analysis")
            return False
            
        # Get dates with moving average data (day 7 onwards)
        dates_with_ma = []
        for date in tree.dates:
            if date in tree.daily_anomalies:
                has_ma_data = any(
                    state != "?" for state in tree.daily_anomalies[date].values()
                )
                if has_ma_data:
                    dates_with_ma.append(date)
        
        if not dates_with_ma:
            print("❌ No dates with sufficient data for moving average analysis")
            return False
            
        # Get last 7 days (or all available if less than 7)
        last_7_days = dates_with_ma[-7:] if len(dates_with_ma) >= 7 else dates_with_ma
        
        print(f'\n🤖 LAST {len(last_7_days)} DAYS ANALYSIS')
        print(f'📅 Period: {last_7_days[0]} to {last_7_days[-1]}')
        print('📊 Using 7-day moving average (trailing, not centered)')
        print("\n" + "="*80)
        
        # Initialize operational analyzer
        print('\n📊 INITIALIZING OPERATIONAL DATA ANALYZER')
        print('-' * 50)
        operational_analyzer = OperationalDataAnalyzer(data_base_path="tables")
        available_nodes = operational_analyzer.get_available_nodes(date_folder)
        operational_analyzer.load_operative_data(date_folder, available_nodes)
        
        # Show analysis for each day
        for i, date in enumerate(last_7_days, 1):
            print(f"\n🗓️  DAY {i}/{len(last_7_days)}: {date}")
            print("─" * 50)
            
            # Get anomaly counts
            anomalies = tree.daily_anomalies[date]
            plus_count = sum(1 for state in anomalies.values() if state == "+")
            minus_count = sum(1 for state in anomalies.values() if state == "-")
            normal_count = sum(1 for state in anomalies.values() if state == "N")
            
            status_emoji = "🚨" if (plus_count + minus_count) > 0 else "✅"
            print(f"{status_emoji} Summary: +{plus_count} anomalies, -{minus_count} anomalies, {normal_count} normal")
            
            # Print tree with explanation needed tags
            interpreter.print_interpreted_tree(tree, date)
            
            # Print tree with operational explanations  
            interpreter.print_tree_with_operational_explanations(tree, date, operational_analyzer)
            
            # Add separator
            if i < len(last_7_days):
                print("\n" + "▔" * 80)
        
        # Final summary table
        print(f"\n📈 WEEK SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Date':<12} {'Status':<8} {'+':<3} {'-':<3} {'N':<3} {'Total':<6}")
        print("─" * 80)
        
        for date in last_7_days:
            anomalies = tree.daily_anomalies[date]
            plus_count = sum(1 for state in anomalies.values() if state == "+")
            minus_count = sum(1 for state in anomalies.values() if state == "-")
            normal_count = sum(1 for state in anomalies.values() if state == "N")
            total_nodes = len([s for s in anomalies.values() if s != "?"])
            
            status = "🚨 Alert" if (plus_count + minus_count) > 0 else "✅ Normal"
            print(f"{date:<12} {status:<8} {plus_count:<3} {minus_count:<3} {normal_count:<3} {total_nodes:<6}")
        
        print("\n💡 Anomaly Legend:")
        print("   [+] = NPS above 7-day average by +10 points or more")
        print("   [-] = NPS below 7-day average by -10 points or more") 
        print("   [N] = NPS within ±10 points of 7-day average")
        
        print("\n🔍 Interpretation Guide:")
        print("   • Consistent: All children have same state as parent")
        print("   • Diluted: Anomalous children's impact reduced by normal children")
        print("   • Significant: Anomalous children's impact overcomes dilution")
        print("   • Cancelled: Positive and negative children cancel each other")
        print("   • Inconsistency: Parent-child relationship doesn't follow expected pattern")
        print("   • [Explanation needed]: Node requires individual anomaly explanation")
        
        return True
        
    except Exception as e:
        print(f'❌ Analysis Error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function - complete automated NPS anomaly detection system"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NPS Anomaly Detection System')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD). If not provided, uses today.')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download, only run analysis')
    args = parser.parse_args()
    
    # Create date folder name
    date_folder = create_date_folder(args.date)
    display_date = args.date if args.date else datetime.now().strftime('%Y-%m-%d')
    
    print('🚀 NPS ANOMALY DETECTION SYSTEM')
    print('=' * 60)
    print(f'📅 Target Date: {display_date}')
    print(f'📂 Data Folder: tables/{date_folder}')
    print('=' * 60)
    
    success = True
    
    # Step 1: Download data (unless skipped)
    if not args.skip_download:
        success = download_all_data(date_folder)
        if not success:
            print('❌ Data download failed. Exiting.')
            return 1
    else:
        print('⏭️  Skipping data download (--skip-download flag)')
        
        # Check if data folder exists
        data_path = Path('tables') / date_folder
        if not data_path.exists():
            print(f'❌ Data folder does not exist: {data_path}')
            print('💡 Remove --skip-download flag to download data first')
            return 1
    
    # Step 2: Run anomaly analysis
    success = run_anomaly_analysis(date_folder)
    if not success:
        print('❌ Anomaly analysis failed.')
        return 1
    
    print('\n✅ SYSTEM COMPLETED SUCCESSFULLY!')
    print(f'📁 Data saved in: tables/{date_folder}')
    print('🎉 Anomaly detection analysis completed!')
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 