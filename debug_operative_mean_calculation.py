#!/usr/bin/env python3
"""
Debug script detallado para verificar cálculos de medias en operative_data_tool modo single
"""

import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, '/app')

from dashboard_analyzer.anomaly_explanation.genai_core.agents.causal_explanation_agent import CausalExplanationAgent
from dashboard_analyzer.anomaly_explanation.data_analyzer import OperationalDataAnalyzer

async def debug_mean_calculations():
    """
    Debug detallado de los cálculos de medias
    """
    print("🔍 DEBUG MEAN CALCULATIONS - OPERATIVE DATA TOOL")
    print("=" * 70)
    
    try:
        # Create agent
        print("📝 Creating CausalExplanationAgent...")
        agent = CausalExplanationAgent()
        
        # Test parameters
        node_path = "Global"
        target_date_str = "2025-09-07"
        target_date_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        baseline_periods = 4
        anomaly_detection_mode = "mean"
        aggregation_days = 1
        
        print(f"🎯 Parameters:")
        print(f"   • node_path: {node_path}")
        print(f"   • target_date: {target_date_str}")
        print(f"   • baseline_periods: {baseline_periods}")
        print(f"   • anomaly_detection_mode: {anomaly_detection_mode}")
        print(f"   • aggregation_days: {aggregation_days}")
        print()
        
        # Step 1: Collect raw operative data
        print("📊 STEP 1: Collecting raw operative data...")
        operative_data = await agent._collect_operative_data_with_query_tracking(
            node_path=node_path,
            target_date=target_date_dt,
            comparison_days=14,  # Extended range to get enough baseline data
            use_flexible=True
        )
        
        if operative_data.empty:
            print("❌ No operative data collected")
            return
            
        print(f"✅ Collected {len(operative_data)} rows of operative data")
        print(f"🔢 Columns: {list(operative_data.columns)}")
        
        # Check what date columns we have
        date_cols = [col for col in operative_data.columns if 'date' in col.lower() or 'Date' in col]
        print(f"📅 Available date columns: {date_cols}")
        
        # Show date range based on available columns
        if 'Min_Date' in operative_data.columns and 'Max_Date' in operative_data.columns:
            print(f"📅 Date range: {operative_data['Min_Date'].min()} to {operative_data['Max_Date'].max()}")
        elif 'Date_Master' in operative_data.columns:
            print(f"📅 Date range: {operative_data['Date_Master'].min()} to {operative_data['Date_Master'].max()}")
        print()
        
        # Step 2: Show raw data sample
        print("📊 STEP 2: Raw data sample (sorted by Period_Group if available):")
        print("-" * 50)
        if 'Period_Group' in operative_data.columns:
            # Sort by period group for better visualization (flexible aggregated data)
            operative_data_sorted = operative_data.sort_values('Period_Group', ascending=True)
            display_cols = ['Period_Group', 'Min_Date', 'Max_Date', 'Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']
            available_cols = [col for col in display_cols if col in operative_data_sorted.columns]
            print(operative_data_sorted[available_cols].head(10).to_string(index=False))
        elif 'Date_Master' in operative_data.columns:
            # Sort by date for legacy data
            operative_data_sorted = operative_data.sort_values('Date_Master', ascending=False)
            display_cols = ['Date_Master', 'Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']
            available_cols = [col for col in display_cols if col in operative_data_sorted.columns]
            print(operative_data_sorted[available_cols].head(10).to_string(index=False))
        else:
            print(operative_data.head(10).to_string(index=False))
        print()
        
        # Step 3: Use OperationalDataAnalyzer to see detailed calculations
        print("📊 STEP 3: Creating OperationalDataAnalyzer for detailed analysis...")
        analyzer = OperationalDataAnalyzer(
            comparison_mode=anomaly_detection_mode,
            comparison_start_date=None,
            comparison_end_date=None,
            aggregation_days=aggregation_days,
            baseline_periods=baseline_periods
        )
        
        # Load data into analyzer
        analyzer.operative_data[node_path] = operative_data
        
        # Step 4: Manual calculation verification
        print("📊 STEP 4: Manual calculation verification...")
        print("-" * 50)
        
        if 'Period_Group' in operative_data.columns:
            # Working with flexible aggregated data (Period_Group format)
            print("📋 Working with flexible aggregated data (Period_Group format)")
            
            # Convert date columns to datetime for comparison
            operative_data['Min_Date'] = pd.to_datetime(operative_data['Min_Date'])
            operative_data['Max_Date'] = pd.to_datetime(operative_data['Max_Date'])
            target_dt = pd.to_datetime(target_date_str)
            
            # Find target period (Period_Group = 1 should be the most recent period)
            target_period = operative_data[operative_data['Period_Group'] == 1]
            print(f"🎯 Target period data (Period_Group = 1):")
            if not target_period.empty:
                row = target_period.iloc[0]
                print(f"   📅 Period: {row['Min_Date'].strftime('%Y-%m-%d')} to {row['Max_Date'].strftime('%Y-%m-%d')}")
                for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                    if col in target_period.columns:
                        value = row[col]
                        print(f"   • {col}: {value}")
            else:
                print("   ❌ No data found for target period")
            print()
            
            # Find baseline periods (including target period: 1 to baseline_periods)
            print(f"📈 Baseline data (Period_Group 1 to {baseline_periods}) - INCLUDING target period:")
            baseline_periods_data = operative_data[
                (operative_data['Period_Group'] >= 1) & 
                (operative_data['Period_Group'] <= baseline_periods)
            ].sort_values('Period_Group', ascending=True)
            
            print(f"   📊 Baseline periods: {len(baseline_periods_data)}")
            
            if not baseline_periods_data.empty:
                display_cols = ['Period_Group', 'Min_Date', 'Max_Date', 'Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']
                available_cols = [col for col in display_cols if col in baseline_periods_data.columns]
                print("   📋 Baseline periods:")
                for _, row in baseline_periods_data.iterrows():
                    period_str = f"Period {int(row['Period_Group'])}: {row['Min_Date'].strftime('%Y-%m-%d')} to {row['Max_Date'].strftime('%Y-%m-%d')}"
                    print(f"   {period_str}")
                    for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                        if col in baseline_periods_data.columns:
                            value = row[col]
                            print(f"     • {col}: {value}")
                
                print(f"\n   🧮 Baseline means:")
                for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                    if col in baseline_periods_data.columns:
                        mean_val = baseline_periods_data[col].mean()
                        print(f"   • {col}: {mean_val:.2f}")
                        
                        # Calculate difference with target (Target vs Mean of Baseline that includes Target)
                        if not target_period.empty and col in target_period.columns:
                            target_val = target_period[col].iloc[0]
                            diff = target_val - mean_val
                            print(f"     └─ Target vs Baseline Mean: {target_val:.2f} - {mean_val:.2f} = {diff:+.2f}")
            print()
            
        elif 'Date_Master' in operative_data.columns:
            # Working with legacy daily data
            print("📋 Working with legacy daily data format")
            # Convert to datetime for comparison
            operative_data['Date_Master'] = pd.to_datetime(operative_data['Date_Master'])
            target_dt = pd.to_datetime(target_date_str)
            
            # Find target date data
            target_data = operative_data[operative_data['Date_Master'] == target_dt.date()]
            print(f"🎯 Target date data ({target_date_str}):")
            if not target_data.empty:
                for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                    if col in target_data.columns:
                        value = target_data[col].iloc[0]
                        print(f"   • {col}: {value}")
            else:
                print("   ❌ No data found for target date")
            print()
            
            # Find baseline data (previous N periods)
            print(f"📈 Baseline data (previous {baseline_periods} days):")
            baseline_start = target_dt - timedelta(days=baseline_periods)
            baseline_data = operative_data[
                (operative_data['Date_Master'] >= baseline_start.date()) & 
                (operative_data['Date_Master'] < target_dt.date())
            ].sort_values('Date_Master', ascending=False)
            
            print(f"   📅 Baseline period: {baseline_start.strftime('%Y-%m-%d')} to {(target_dt - timedelta(days=1)).strftime('%Y-%m-%d')}")
            print(f"   📊 Baseline data points: {len(baseline_data)}")
            
            if not baseline_data.empty:
                display_cols = ['Date_Master', 'Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']
                available_cols = [col for col in display_cols if col in baseline_data.columns]
                print("   📋 Baseline data:")
                print("   " + baseline_data[available_cols].to_string(index=False).replace('\n', '\n   '))
                
                print(f"\n   🧮 Baseline means:")
                for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                    if col in baseline_data.columns:
                        mean_val = baseline_data[col].mean()
                        print(f"   • {col}: {mean_val:.2f}")
                        
                        # Calculate difference with target
                        if not target_data.empty and col in target_data.columns:
                            target_val = target_data[col].iloc[0]
                            diff = target_val - mean_val
                            print(f"     └─ Target vs Mean: {target_val:.2f} - {mean_val:.2f} = {diff:+.2f}")
            print()
        
        # Step 5: Run the analyzer and compare
        print("📊 STEP 5: Running OperationalDataAnalyzer...")
        analysis_result = analyzer.analyze_operative_metrics(node_path, target_date_str)
        
        if 'error' in analysis_result:
            print(f"❌ Analyzer error: {analysis_result['error']}")
        else:
            print("✅ Analyzer results:")
            if 'metrics' in analysis_result:
                for metric_name, metric_data in analysis_result['metrics'].items():
                    print(f"   • {metric_name}:")
                    print(f"     - Current: {metric_data.get('current', metric_data.get('current_value', 'N/A'))}")
                    print(f"     - Baseline: {metric_data.get('baseline', metric_data.get('historical_mean', 'N/A'))}")
                    print(f"     - Historical Periods: {metric_data.get('historical_periods', 'N/A')}")
                    print(f"     - Delta: {metric_data.get('delta', 'N/A')}")
                    print(f"     - Direction: {metric_data.get('direction', 'N/A')}")
                    print(f"     - Significant: {metric_data.get('is_significant', 'N/A')}")
            
            if 'summary' in analysis_result:
                print(f"\n   📋 Summary: {analysis_result['summary']}")
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_mean_calculations())
