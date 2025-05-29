#!/usr/bin/env python3
"""
NPS Anomaly Detection System - Main Pipeline

Complete pipeline for:
1. Data collection from Power BI
2. Anomaly detection with 7-day moving averages 
3. Tree interpretation with bottom-up analysis
4. Operational explanation correlation
5. AI-powered anomaly interpretation

Usage:
    python -m dashboard_analyzer.main
    python -m dashboard_analyzer.main --skip-download
    python -m dashboard_analyzer.main --date 2025-05-24
"""

import argparse
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .data_collection.pbi_collector import PBIDataCollector
from .anomaly_detection.anomaly_tree import AnomalyTree
from .anomaly_detection.anomaly_interpreter import AnomalyInterpreter
from .anomaly_explanation.data_analyzer import OperationalDataAnalyzer
from .anomaly_explanation.anomaly_interpreter_agent import AnomalyInterpreterAgent
from .anomaly_explanation.genai_core.utils.enums import LLMType

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def print_week_summary(tree, dates):
    """Print a summary table of the week's anomaly status"""
    print("\n" + "="*60)
    print("📈 WEEK SUMMARY TABLE")
    print("="*60)
    print(f"{'Date':<12} {'Status':<10} {'+':<3} {'-':<3} {'N':<3} {'Total':<5}")
    print("-" * 40)
    
    for date_str in dates:
        if date_str in tree.daily_anomalies:
            anomalies = tree.daily_anomalies[date_str]
            pos_count = sum(1 for state in anomalies.values() if state == '+')
            neg_count = sum(1 for state in anomalies.values() if state == '-')
            normal_count = sum(1 for state in anomalies.values() if state == 'N')
            total = len([s for s in anomalies.values() if s != "?"])
            
            # Determine status
            if pos_count > 0 or neg_count > 0:
                status = "🚨 Alert"
            else:
                status = "✅ Normal"
            
            print(f"{date_str:<12} {status:<10} {pos_count:<3} {neg_count:<3} {normal_count:<3} {total:<5}")
    
    print("-" * 40)

async def analyze_week_with_ai_interpretation(tree, interpreter, operational_analyzer, available_dates):
    """Analyze the last 7 days with AI interpretation for each day"""
    # Get last 7 days
    last_7_days = available_dates[-7:] if len(available_dates) >= 7 else available_dates
    
    print(f"\n🔍 ANALYZING LAST {len(last_7_days)} DAYS WITH AI INTERPRETATION")
    print("="*80)
    
    for i, date in enumerate(last_7_days, 1):
        print(f"\n📅 DAY {i}/{len(last_7_days)}: {date}")
        print("="*60)
        
        # Check if this date has any anomalies
        if date in tree.daily_anomalies:
            anomalies = tree.daily_anomalies[date]
            anomaly_count = sum(1 for state in anomalies.values() if state in ['+', '-'])
            
            if anomaly_count == 0:
                print(f"✅ No anomalies detected for {date} - skipping AI analysis")
                continue
            
            print(f"🚨 {anomaly_count} anomalies detected - performing analysis...")
        
        # Get interpretations for the date
        interpretations = interpreter.analyze_tree_for_date(tree, date)
        
        # Show tree with explanation flags
        print("🌳 Tree with [Explanation needed] flags:")
        print("-" * 40)
        interpreter.print_interpreted_tree(tree, date, interpretations)
        
        # Show tree with operational explanations
        print("\n🌳 Tree with Operational Explanations:")
        print("-" * 40)
        tree_with_explanations = interpreter.print_tree_with_operational_explanations(
            tree, date, operational_analyzer, interpretations
        )
        
        # AI Interpretation
        print(f"\n🤖 AI INTERPRETATION FOR {date}:")
        print("="*45)
        
        try:
            # Create AI agent
            ai_agent = AnomalyInterpreterAgent(
                llm_type=LLMType.CLAUDE_SONNET_4,  # Using AWS Claude Sonnet 4 with inference profile
                logger=logging.getLogger("ai_interpreter")
            )
            
            # Build AI input string from tree data
            ai_input = f"🌳 Anomaly Tree with Operational Explanations: {date}\n"
            
            if date in tree.daily_anomalies:
                anomalies = tree.daily_anomalies[date]
                
                # Build hierarchical string
                global_state = anomalies.get("Global", "?")
                ai_input += f"Global [{global_state}]\n"
                
                if "Global" in interpretations:
                    ai_input += f"  {interpretations['Global']}\n"
                
                # Add LH branch
                lh_state = anomalies.get("Global/LH", "?")
                ai_input += f"\n  LH [{lh_state}]\n"
                if "Global/LH" in interpretations:
                    ai_input += f"    {interpretations['Global/LH']}\n"
                    
                for cabin in ["Economy", "Business", "Premium"]:
                    cabin_path = f"Global/LH/{cabin}"
                    cabin_state = anomalies.get(cabin_path, "?")
                    ai_input += f"\n    {cabin} [{cabin_state}]\n"
                    if cabin_path in interpretations:
                        ai_input += f"      {interpretations[cabin_path]}\n"
                
                # Add SH branch
                sh_state = anomalies.get("Global/SH", "?")
                ai_input += f"\n  SH [{sh_state}]\n"
                if "Global/SH" in interpretations:
                    ai_input += f"    {interpretations['Global/SH']}\n"
                    
                for cabin in ["Economy", "Business"]:
                    cabin_path = f"Global/SH/{cabin}"
                    cabin_state = anomalies.get(cabin_path, "?")
                    ai_input += f"\n    {cabin} [{cabin_state}]\n"
                    if cabin_path in interpretations:
                        ai_input += f"      {interpretations[cabin_path]}\n"
                        
                    for company in ["IB", "YW"]:
                        company_path = f"Global/SH/{cabin}/{company}"
                        company_state = anomalies.get(company_path, "?")
                        ai_input += f"\n      {company} [{company_state}]\n"
                        if company_path in interpretations:
                            ai_input += f"        {interpretations[company_path]}\n"
            
            # Generate AI interpretation
            ai_interpretation = await ai_agent.interpret_anomaly_tree(
                tree_data=ai_input,
                date=date
            )
            
            print(ai_interpretation)
            
            # Show performance metrics
            metrics = ai_agent.get_performance_metrics()
            print(f"\n📊 AI Performance: {metrics['last_execution_time']:.2f}s, "
                  f"Tokens: {metrics['input_tokens']}+{metrics['output_tokens']}, "
                  f"Cost: ${metrics['money_spent']:.4f}")
            
        except Exception as e:
            print(f"❌ AI Interpretation failed for {date}: {e}")
            print("Continuing with next day...")
            import traceback
            traceback.print_exc()
        
        # Add separator between days
        if i < len(last_7_days):
            print("\n" + "="*80)

async def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description='NPS Anomaly Detection System')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip data download, use existing data')
    parser.add_argument('--date', type=str, 
                       help='Analyze specific date (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    print("🚀 NPS ANOMALY DETECTION SYSTEM")
    print("="*50)
    
    try:
        # Determine target date and folder
        if args.date:
            # Parse the specified date and convert to folder format
            target_analysis_date = args.date
            date_obj = datetime.strptime(args.date, '%Y-%m-%d')
            target_folder = date_obj.strftime('%d_%m_%Y')
        else:
            # Use today's date
            today = datetime.now()
            target_analysis_date = today.strftime('%Y-%m-%d')
            target_folder = today.strftime('%d_%m_%Y')
        
        print(f"🎯 Target analysis date: {target_analysis_date}")
        print(f"📁 Looking for data folder: {target_folder}")
        
        # Step 1: Data Collection (if not skipped)
        if not args.skip_download:
            print("\n📥 STEP 1: Data Collection")
            print("-" * 30)
            
            collector = PBIDataCollector()
            success_count, total_count = await collector.collect_all_data()
            
            print(f"✅ Data collection completed: {success_count}/{total_count} successful")
        else:
            print("\n⏭️  STEP 1: Data Collection (SKIPPED)")
        
        # Step 2: Anomaly Detection
        print("\n🔍 STEP 2: Anomaly Detection")
        print("-" * 30)
        
        tree = AnomalyTree()
        tree.build_tree_structure()
        
        # Check if the target folder exists
        tables_path = Path("tables")
        target_folder_path = tables_path / target_folder
        
        if not target_folder_path.exists():
            print(f"❌ Data folder not found: tables/{target_folder}")
            
            # Show available folders
            if tables_path.exists():
                available_folders = [d.name for d in tables_path.iterdir() if d.is_dir()]
                if available_folders:
                    available_folders.sort(key=lambda x: datetime.strptime(x, '%d_%m_%Y'))
                    print(f"📅 Available folders: {', '.join(available_folders)}")
                    latest_folder = available_folders[-1]
                    print(f"💡 Suggestion: Use --date {datetime.strptime(latest_folder, '%d_%m_%Y').strftime('%Y-%m-%d')}")
            else:
                    print("📅 No data folders found in tables/")
            return
        
        print(f"📊 Using data from: {target_folder}")
        tree.load_data(target_folder)
        tree.calculate_moving_averages()
        tree.detect_daily_anomalies()
        
        available_dates = tree.dates
        if not available_dates:
            print("❌ No data available for analysis")
            return
        
        print(f"📊 Loaded data for {len(available_dates)} days: {available_dates[0]} to {available_dates[-1]}")
        
        # Determine the actual analysis date from available data
        if args.date:
            # User specified a date, use it if available
            if target_analysis_date not in available_dates:
                print(f"⚠️  Specified date {target_analysis_date} not found in data")
                print(f"📅 Available dates: {available_dates[0]} to {available_dates[-1]}")
                # Use the closest available date
                target_analysis_date = available_dates[-1]
                print(f"🔄 Using most recent date instead: {target_analysis_date}")
        else:
            # No date specified, use most recent available date
            target_analysis_date = available_dates[-1]
            print(f"📅 Using most recent available date: {target_analysis_date}")
        
        # Show week summary
        print_week_summary(tree, available_dates[-7:])  # Last 7 days
        
        # Step 3: Tree Interpretation
        print("\n🌳 STEP 3: Tree Interpretation") 
        print("-" * 30)
        
        interpreter = AnomalyInterpreter()
        operational_analyzer = OperationalDataAnalyzer()
        
        # Load operational data for the same folder
        operational_analyzer.load_operative_data(target_folder, list(tree.nodes.keys()))
        
        # Analyze with AI interpretation
        await analyze_week_with_ai_interpretation(tree, interpreter, operational_analyzer, available_dates)
        
        print(f"\n✅ Analysis completed for {target_analysis_date}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 