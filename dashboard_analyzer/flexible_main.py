#!/usr/bin/env python3
"""
Flexible NPS Anomaly Detection System
Supports configurable temporal aggregation (7, 14, 30 days, etc.)
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_detection.flexible_detector import FlexibleAnomalyDetector
from dashboard_analyzer.anomaly_detection.anomaly_interpreter import AnomalyInterpreter

async def collect_flexible_data(aggregation_days: int, target_folder: str):
    """Collect data using flexible aggregation"""
    print(f"📥 STEP 1: Flexible Data Collection ({aggregation_days} days)")
    print("-" * 50)
    
    collector = PBIDataCollector()
    
    # Define all node paths
    node_paths = [
        "Global",
        "Global/LH",
        "Global/LH/Economy", 
        "Global/LH/Business", 
        "Global/LH/Premium",
        "Global/SH",
        "Global/SH/Economy", 
        "Global/SH/Business",
        "Global/SH/Economy/IB", 
        "Global/SH/Economy/YW",
        "Global/SH/Business/IB", 
        "Global/SH/Business/YW"
    ]
    
    print(f"Starting flexible data collection to: {target_folder}")
    
    # Collect data for all nodes
    total_success = 0
    total_attempted = 0
    
    for node_path in node_paths:
        try:
            results = await collector.collect_flexible_data_for_node(
                node_path, aggregation_days, target_folder
            )
            total_attempted += len(results)
            total_success += sum(results.values())
        except Exception as e:
            print(f"❌ Error collecting data for {node_path}: {e}")
    
    print(f"\n📊 Flexible Data Collection Summary:")
    print(f"   Total files attempted: {total_attempted}")
    print(f"   Successful files: {total_success}")
    if total_attempted > 0:
        print(f"   Success rate: {total_success/total_attempted*100:.1f}%")
    else:
        print(f"   Success rate: 0.0% (no files attempted)")
    
    if total_success > 0:
        print(f"✅ Flexible data collection completed: {total_success}/{total_attempted} successful")
        return True
    else:
        print("❌ No data collected successfully")
        return False

def detect_flexible_anomalies(aggregation_days: int, data_folder: str):
    """Detect anomalies using flexible aggregation"""
    print(f"\n🔍 STEP 2: Flexible Anomaly Detection ({aggregation_days} days)")
    print("-" * 50)
    
    detector = FlexibleAnomalyDetector(
        aggregation_days=aggregation_days,
        threshold=10.0,
        min_sample_size=5
    )
    
    anomalies, deviations, periods = detector.analyze_flexible_anomalies(data_folder)
    
    if not anomalies:
        print("❌ No anomalies detected or insufficient data")
        return None, None, None
    
    # Generate period summary
    summary_df = detector.get_period_summary(data_folder, periods)
    
    print(f"\n{'='*60}")
    print(f"📈 PERIOD SUMMARY TABLE ({aggregation_days} days aggregation)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("-" * 60)
    
    return anomalies, deviations, periods

def print_flexible_tree(anomalies: dict, deviations: dict, aggregation_days: int, target_period: int):
    """Print the anomaly tree for flexible analysis"""
    print(f"\n🌳 STEP 3: Flexible Tree Analysis")
    print("-" * 40)
    
    def get_nps_info(node_path):
        if node_path in deviations:
            deviation = deviations[node_path]
            return f"({deviation:+.1f} vs baseline)"
        return ""
    
    def needs_explanation_tag(node_path):
        # Simple logic for explanation flags (can be enhanced)
        state = anomalies.get(node_path, "?")
        if state in ["+", "-"]:
            return " [Explanation needed]"
        return ""
    
    print(f"\n🌳 Flexible Anomaly Tree - Period {target_period} ({aggregation_days}d aggregation)")
    print("-" * 60)
    
    # Global
    global_state = anomalies.get("Global", "?")
    global_info = get_nps_info("Global")
    print(f"🌍 Global [{global_state}]{needs_explanation_tag('Global')}")
    if global_info:
        print(f"    📊 {global_info}")
    
    # Long Haul
    lh_state = anomalies.get("Global/LH", "?")
    lh_info = get_nps_info("Global/LH")
    print(f"\n├─ 🛫 LONG HAUL (LH) [{lh_state}]{needs_explanation_tag('Global/LH')}")
    if lh_info:
        print(f"│    📊 {lh_info}")
    
    lh_cabins = ["Economy", "Business", "Premium"]
    for i, cabin in enumerate(lh_cabins):
        cabin_path = f"Global/LH/{cabin}"
        cabin_state = anomalies.get(cabin_path, "?")
        cabin_info = get_nps_info(cabin_path)
        
        connector = "├──" if i < len(lh_cabins) - 1 else "└──"
        print(f"│  {connector} 💺 {cabin} [{cabin_state}]{needs_explanation_tag(cabin_path)}")
        if cabin_info:
            print(f"│      📊 {cabin_info}")
    
    # Short Haul
    sh_state = anomalies.get("Global/SH", "?")
    sh_info = get_nps_info("Global/SH")
    print(f"\n└─ ✈️  SHORT HAUL (SH) [{sh_state}]{needs_explanation_tag('Global/SH')}")
    if sh_info:
        print(f"     📊 {sh_info}")
    
    sh_cabins = ["Economy", "Business"]
    for i, cabin in enumerate(sh_cabins):
        cabin_path = f"Global/SH/{cabin}"
        cabin_state = anomalies.get(cabin_path, "?")
        cabin_info = get_nps_info(cabin_path)
        
        connector = "├──" if i < len(sh_cabins) - 1 else "└──"
        print(f"   {connector} 💺 {cabin} [{cabin_state}]{needs_explanation_tag(cabin_path)}")
        if cabin_info:
            print(f"       📊 {cabin_info}")
        
        # Company subdivisions for SH
        companies = ["IB", "YW"]
        for j, company in enumerate(companies):
            company_path = f"Global/SH/{cabin}/{company}"
            company_state = anomalies.get(company_path, "?")
            company_info = get_nps_info(company_path)
            
            company_connector = "├────" if j < len(companies) - 1 else "└────"
            print(f"       {company_connector} 🏢 {company} [{company_state}]{needs_explanation_tag(company_path)}")
            if company_info:
                print(f"             📊 {company_info}")

async def main():
    """Main entry point for flexible anomaly detection"""
    parser = argparse.ArgumentParser(description='Flexible NPS Anomaly Detection System')
    parser.add_argument('--days', '-d', type=int, default=7, 
                       help='Aggregation period in days (default: 7)')
    parser.add_argument('--mode', '-m', choices=['collect', 'analyze', 'full'], default='full',
                       help='Mode: collect data only, analyze only, or full pipeline (default: full)')
    parser.add_argument('--folder', '-f', type=str, default=None,
                       help='Target folder for data (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Generate folder name if not provided
    if args.folder is None:
        current_date = datetime.now()
        date_str = current_date.strftime('%d_%m_%Y')
        args.folder = f"tables/flexible_{args.days}d_{date_str}"
    
    print("🚀 FLEXIBLE NPS ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    print(f"📅 Aggregation period: {args.days} days")
    print(f"📁 Data folder: {args.folder}")
    print(f"🔧 Mode: {args.mode}")
    
    # Step 1: Data Collection
    if args.mode in ['collect', 'full']:
        success = await collect_flexible_data(args.days, args.folder)
        if not success and args.mode == 'full':
            print("❌ Data collection failed, stopping pipeline")
            return
    
    # Step 2: Anomaly Detection  
    if args.mode in ['analyze', 'full']:
        anomalies, deviations, periods = detect_flexible_anomalies(args.days, args.folder)
        
        if anomalies and periods:
            # Step 3: Tree Visualization
            latest_period = periods[0]
            print_flexible_tree(anomalies, deviations, args.days, latest_period)
            
            # Summary
            anomaly_count = sum(1 for a in anomalies.values() if a in ['+', '-'])
            print(f"\n🎯 SUMMARY:")
            print(f"   📊 Period analyzed: {latest_period}")
            print(f"   🔍 Total anomalies: {anomaly_count}")
            print(f"   📈 Aggregation: {args.days} days")
            print(f"   💾 Data saved in: {args.folder}")
        else:
            print("❌ No anomaly analysis possible")

if __name__ == "__main__":
    asyncio.run(main()) 