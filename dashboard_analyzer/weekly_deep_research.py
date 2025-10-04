#!/usr/bin/env python3
"""
Weekly Deep Research - Automated NPS Analysis
Runs weekly comparative (7d) + daily single (1d x7) analysis automatically
Consolidates results, uploads to S3, and sends email notifications
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import sys

# Import execute_analysis_flow and helpers from deep_research_period
from dashboard_analyzer.deep_research_period import (
    execute_analysis_flow,
    determine_anomaly_mode_for_vslast
)

from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type


async def generate_consolidated_summary(agent, consolidated_data: List[Dict], date_flight_local: str = None) -> str:
    """Generate a consolidated summary from multiple analysis types including weekly comparative and daily single analyses."""
    
    # Check if it's the weekly format (with 'weekly_comparative' and 'daily_singles' keys)
    if consolidated_data and isinstance(consolidated_data[0], dict) and 'weekly_comparative' in consolidated_data[0]:
        # New format from weekly_deep_research.py
        data = consolidated_data[0]
        weekly_data = data.get('weekly_comparative', '')
        daily_data = data.get('daily_singles', [])
        
        # Format weekly data (it's a list of periods from execute_analysis_flow)
        if isinstance(weekly_data, list) and weekly_data:
            weekly_comparative_analysis = ""
            for period in weekly_data:
                date_range = period.get('date_range', 'Unknown')
                interpretation = period.get('ai_interpretation', '')
                weekly_comparative_analysis += f"{interpretation}\n\n"
        elif isinstance(weekly_data, str):
            weekly_comparative_analysis = weekly_data
        else:
            weekly_comparative_analysis = ""
        
        # Daily data is already formatted
        daily_single_analyses = daily_data
        
        # Call generate_comprehensive_summary
        try:
            comprehensive_summary = await asyncio.wait_for(
                agent.generate_comprehensive_summary(
                    weekly_comparative_analysis=weekly_comparative_analysis,
                    daily_single_analyses=daily_single_analyses,
                    date_flight_local=date_flight_local
                ),
                timeout=600.0
            )
            return comprehensive_summary
        except Exception as e:
            print(f"❌ Error in generate_comprehensive_summary: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error generating comprehensive summary: {str(e)}"
    
    else:
        # Unexpected format
        print(f"⚠️ Unexpected consolidated_data format")
        return f"❌ Error: Unexpected data format for consolidation"


async def run_weekly_comprehensive_analysis(
    analysis_date: datetime,
    date_parameter: str,
    segment: str = "Global",
    causal_filter: str = "vs L7d",
    comparison_start_date: Optional[datetime] = None,
    comparison_end_date: Optional[datetime] = None,
    date_flight_local: Optional[str] = None,
    # Parameters for daily analysis with defaults matching old hardcoded values
    daily_anomaly_detection_mode: str = 'mean',
    daily_baseline_periods: int = 7,
    daily_aggregation_days: int = 1,
    daily_periods: int = 7
):
    """
    Comprehensive weekly analysis orchestrator.
    Executes two distinct analysis flows:
    1. A weekly comparative analysis (7d).
    2. A daily single analysis for each of the last 7 days.
    Finally, it consolidates the results, uploads to S3, and sends email.
    """
    print("🚀 WEEKLY DEEP RESEARCH - Comprehensive NPS Analysis")
    print("=" * 80)
    print(f"📅 Analysis Date: {analysis_date.strftime('%Y-%m-%d')} ({date_parameter})")
    print(f"🎯 Segment Focus: {segment}")

    generated_reports = []

    # --- 1. Weekly Comparative Analysis ---
    print("\n" + "=" * 40)
    print("📊 STEP 1: Running Weekly Comparative Analysis")
    print("=" * 40)
    
    # Determine anomaly detection mode based on causal filter
    comp_start_str = comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date and hasattr(comparison_start_date, 'strftime') else str(comparison_start_date) if comparison_start_date else None
    comp_end_str = comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date and hasattr(comparison_end_date, 'strftime') else str(comparison_end_date) if comparison_end_date else None
    anomaly_mode, baseline_desc = determine_anomaly_mode_for_vslast(causal_filter, comp_start_str, comp_end_str)
    print(f"🎯 Anomaly Detection Mode: {anomaly_mode} (baseline: {baseline_desc})")
    
    try:
        weekly_report_path = await execute_analysis_flow(
            analysis_date=analysis_date,
            date_parameter=date_parameter,
            segment=segment,
            anomaly_detection_mode=anomaly_mode,
            baseline_periods=7,
            aggregation_days=7,
            periods=1,
            causal_filter=causal_filter,
            comparison_start_date=comparison_start_date,
            comparison_end_date=comparison_end_date,
            date_flight_local=date_flight_local,
            study_mode="comparative",
        )
        if weekly_report_path and "Error" not in str(weekly_report_path):
            generated_reports.append({
                'type': 'weekly',
                'data': weekly_report_path
            })
            print(f"✅ Weekly analysis completed. Data length: {len(str(weekly_report_path))} chars")
        else:
            print(f"⚠️ Weekly analysis completed but no anomalies found or insufficient data.")
            if weekly_report_path and "Error" not in str(weekly_report_path):
                generated_reports.append({
                    'type': 'weekly',
                    'data': weekly_report_path
                })

    except Exception as e:
        print(f"❌ CRITICAL ERROR during weekly analysis: {e}")
        import traceback
        traceback.print_exc()

    # --- 2. Daily Analysis ---
    print("\n" + "=" * 40)
    print("📊 STEP 2: Running Daily Analysis")
    print("=" * 40)
    try:
        daily_report_path = await execute_analysis_flow(
            analysis_date=analysis_date,
            date_parameter=date_parameter,
            segment=segment,
            anomaly_detection_mode=daily_anomaly_detection_mode,
            baseline_periods=daily_baseline_periods,
            aggregation_days=daily_aggregation_days,
            periods=daily_periods,
            causal_filter=None,
            comparison_start_date=None,
            comparison_end_date=None,
            date_flight_local=date_flight_local,
            study_mode="single",
        )
        if daily_report_path and "Error" not in str(daily_report_path):
            generated_reports.append({
                'type': 'daily',
                'data': daily_report_path
            })
            print(f"✅ Daily analysis completed. Data length: {len(str(daily_report_path))} chars")
        else:
            print(f"❌ Daily analysis failed. Reason: {daily_report_path}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during daily analysis: {e}")
        import traceback
        traceback.print_exc()

    # --- 3. Final Summary ---
    print("\n" + "=" * 40)
    print("📝 STEP 3: Consolidating All Reports")
    print("=" * 40)
    if generated_reports:
        print(f"✅ Found {len(generated_reports)} reports to summarize.")
        
        # Separate weekly and daily reports
        weekly_comparative_analysis = ""
        daily_single_analyses = []
        
        for report in generated_reports:
            if report['type'] == 'weekly':
                weekly_comparative_analysis = report['data']
                print(f"✅ Found weekly report: {len(weekly_comparative_analysis)} chars")
            elif report['type'] == 'daily':
                daily_single_analyses = report['data']
                print(f"✅ Found daily analysis data: {len(daily_single_analyses)} periods")
        
        # Convert daily data to the format expected by summary agent if needed
        if daily_single_analyses and isinstance(daily_single_analyses, list):
            formatted_daily_analyses = []
            for daily in daily_single_analyses:
                if isinstance(daily, dict) and 'ai_interpretation' in daily:
                    formatted_daily_analyses.append({
                        'date': daily.get('date_range', daily.get('period', 'Unknown')),
                        'analysis': daily.get('ai_interpretation', ''),
                        'anomalies': ['daily_analysis']
                    })
            daily_single_analyses = formatted_daily_analyses
            print(f"✅ Formatted daily analyses: {len(daily_single_analyses)} periods")

        # Use Summary Agent to consolidate
        try:
            print("\n🤖 Initializing Summary Agent...")
            summary_agent = AnomalySummaryAgent(
                llm_type=get_default_llm_type(),
                logger=logging.getLogger("summary_agent")
            )
            print("✅ Summary Agent initialized. Generating executive summary...")
            
            # Prepare metadata
            execution_metadata = {
                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                'segment': segment,
                'causal_filter': causal_filter
            }
            
            weekly_analysis_params = {
                'anomaly_detection_mode': 'vslast',
                'baseline_periods': 7,
                'aggregation_days': 7,
                'periods': 1,
                'study_mode': 'comparative'
            }
            
            daily_analysis_params = {
                'daily_anomaly_detection_mode': daily_anomaly_detection_mode,
                'daily_baseline_periods': daily_baseline_periods,
                'daily_aggregation_days': daily_aggregation_days,
                'daily_periods': daily_periods
            }
            
            consolidated_data = [{
                'weekly_comparative': weekly_comparative_analysis,
                'daily_singles': daily_single_analyses,
                'metadata': execution_metadata,
                'weekly_params': weekly_analysis_params,
                'daily_params': daily_analysis_params
            }]
            
            executive_summary = await generate_consolidated_summary(
                summary_agent, 
                consolidated_data, 
                date_flight_local
            )
            
            print("\n" + "=" * 80)
            print("📋 EXECUTIVE SUMMARY")
            print("=" * 80)
            print(executive_summary)
            print("=" * 80)
            
            # TODO: Upload to S3 and send email (if configured)
            print("\n📤 Upload to S3 and email notification: [TODO]")
            
            return executive_summary
            
        except Exception as e:
            print(f"❌ Error generating executive summary: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("⚠️ No reports generated. Skipping consolidation.")
        return None


async def main():
    """Main entry point for weekly deep research"""
    print("🚀 Weekly Deep Research System")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Weekly Deep Research - Automated NPS Analysis')
    
    # Date-related parameters
    parser.add_argument('--insert-date-ci', type=str,
                       help='Simulate today being this date (YYYY-MM-DD). Data available until this date - 4 days')
    parser.add_argument('--date-flight-local', type=str,
                       help='Use this date directly as available in dashboard (YYYY-MM-DD)')
    
    # Segment parameter
    parser.add_argument('--segment', type=str, default='Global',
                       help='Root segment to analyze (e.g., Global, SH, Global/SH/Economy, Global/LH). Default: Global (full tree)')
    
    # Causal comparison filter parameters
    parser.add_argument('--causal-filter-comparison',
                        type=str,
                        default="vs L7d",
                        help='Causal filter for comparative analysis (e.g., "vs L7d", "vs LM", "vs LY", "vs Target", "vs Sel. Period")')
    parser.add_argument('--comparison-start-date', type=str,
                        help='Start date for comparison period (YYYY-MM-DD) when using --causal-filter-comparison "vs Sel. Period"')
    parser.add_argument('--comparison-end-date', type=str,
                        help='End date for comparison period (YYYY-MM-DD) when using --causal-filter-comparison "vs Sel. Period"')
    
    # Daily analysis parameters (with defaults)
    parser.add_argument('--daily-anomaly-detection-mode', type=str, default='mean',
                       help='Anomaly detection mode for daily analysis. Default: mean')
    parser.add_argument('--daily-baseline-periods', type=int, default=7,
                       help='Number of baseline periods for daily analysis. Default: 7')
    
    args = parser.parse_args()
    
    # Add placeholders for arguments that might not be defined
    if not hasattr(args, 'comparison_start_date'):
        args.comparison_start_date = None
    if not hasattr(args, 'comparison_end_date'):
        args.comparison_end_date = None
    
    # Calculate the analysis start date based on date parameters
    today = datetime.now().date()
    pbi_lag_days = 4  # PBI dashboard has 4-day lag
    
    # Determine which date parameter was used
    if args.insert_date_ci and args.date_flight_local:
        print("❌ Error: Cannot specify both --insert-date-ci and --date-flight-local. Choose one.")
        return
    elif args.insert_date_ci:
        try:
            simulated_today = datetime.strptime(args.insert_date_ci, '%Y-%m-%d').date()
            analysis_date = datetime.combine(simulated_today - timedelta(days=pbi_lag_days), datetime.min.time())
            date_parameter = 'insert_ci'
            date_description = f"Simulating today as {simulated_today.strftime('%Y-%m-%d')}"
        except ValueError:
            print("❌ Error: --insert-date-ci must be in YYYY-MM-DD format")
            return
    elif args.date_flight_local:
        try:
            analysis_date = datetime.strptime(args.date_flight_local, '%Y-%m-%d')
            date_parameter = 'flight_local'
            date_description = "Using direct dashboard date"
        except ValueError:
            print("❌ Error: --date-flight-local must be in YYYY-MM-DD format")
            return
    else:
        # Use default: today's date minus PBI lag
        analysis_date = datetime.combine(today - timedelta(days=pbi_lag_days), datetime.min.time())
        date_parameter = 'default'
        date_description = f"Default (today - {pbi_lag_days} days lag)"
    
    print(f"\n📅 DATE CONFIGURATION:")
    print(f"   • {date_description}")
    print(f"   • Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
    if date_parameter == 'insert_ci':
        print(f"   • Note: Simulating CI run on {args.insert_date_ci}")
    elif date_parameter == 'flight_local':
        print(f"   • Note: Using date directly from dashboard without lag simulation")
    
    # Parse comparison dates if provided
    comparison_start_date = None
    comparison_end_date = None
    if args.comparison_start_date:
        try:
            comparison_start_date = datetime.strptime(args.comparison_start_date, '%Y-%m-%d')
        except ValueError:
            print("❌ Error: --comparison-start-date must be in YYYY-MM-DD format")
            return
    if args.comparison_end_date:
        try:
            comparison_end_date = datetime.strptime(args.comparison_end_date, '%Y-%m-%d')
        except ValueError:
            print("❌ Error: --comparison-end-date must be in YYYY-MM-DD format")
            return
    
    if args.causal_filter_comparison == "vs Sel. Period" and (not comparison_start_date or not comparison_end_date):
        print("❌ Error: When using 'vs Sel. Period', you must provide both --comparison-start-date and --comparison-end-date")
        return
    
    if comparison_start_date and comparison_end_date:
        print(f"   • Comparison period: {comparison_start_date.strftime('%Y-%m-%d')} to {comparison_end_date.strftime('%Y-%m-%d')}")
    
    # Execute weekly comprehensive analysis
    try:
        await run_weekly_comprehensive_analysis(
            analysis_date=analysis_date,
            date_parameter=date_parameter,
            segment=args.segment,
            causal_filter=args.causal_filter_comparison,
            comparison_start_date=comparison_start_date,
            comparison_end_date=comparison_end_date,
            date_flight_local=args.date_flight_local,
            daily_anomaly_detection_mode=args.daily_anomaly_detection_mode,
            daily_baseline_periods=args.daily_baseline_periods
        )
    except KeyboardInterrupt:
        print("\n⏸️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())

