#!/usr/bin/env python3
"""
Weekly Deep Research - Automated NPS Analysis

Pipeline (all three steps active):
  1. Weekly comparative analysis (7d) — interpreter uses CLAUDE_OPUS_4_6 (default).
  2. Daily single analyses (1d × 7) — interpreter uses CLAUDE_OPUS_4_5 to reduce cost.
  3. Executive summarizer — uses CLAUDE_OPUS_4_6; uploads comprehensive report to S3.
"""

import asyncio
import argparse
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import sys

import pandas as pd

# Import execute_analysis_flow and helpers from deep_research_period
from dashboard_analyzer.deep_research_period import (
    execute_analysis_flow,
    determine_anomaly_mode_for_vslast
)
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.data_collection.s3_report_uploader import S3ReportUploader
from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType, get_default_llm_type

# Cost-optimised model assignment:
#   - Daily interpreter  → CLAUDE_SONNET_4_5 (cheaper, high-throughput)
#   - Weekly interpreter → CLAUDE_OPUS_4_6 (default, highest quality)
#   - Summarizer         → CLAUDE_OPUS_4_6 (executive report, highest quality)
DAILY_LLM_TYPE = LLMType.CLAUDE_SONNET_4_5
SUMMARY_LLM_TYPE = LLMType.CLAUDE_OPUS_4_6


# Constants for data availability probing
MIN_PBI_LAG_DAYS = 4   # Minimum expected PBI lag
MAX_PBI_LAG_DAYS = 10  # Maximum lag to probe before giving up


async def find_latest_available_date(
    simulated_today: datetime,
    segment: str = "Global",
    environment: str = "prod"
) -> Tuple[datetime, int]:
    """
    Probe for the latest date with available data, starting from MIN_PBI_LAG_DAYS.
    
    This function handles variable PBI data pipeline delays by progressively
    checking older dates until data is found.
    
    Args:
        simulated_today: The date we're simulating as "today" (from --insert-date-ci)
        segment: Segment to check data for (default: "Global")
        environment: Environment setting ("local" or "prod")
        
    Returns:
        Tuple of (analysis_date, actual_lag_days_used)
        
    Raises:
        ValueError: If no data found within MAX_PBI_LAG_DAYS
        
    Example:
        If simulated_today is 2025-01-25 and data is available starting from 2025-01-19:
        - Tries 2025-01-21 (lag 4)... no data
        - Tries 2025-01-20 (lag 5)... no data
        - Tries 2025-01-19 (lag 6)... data found!
        - Returns (datetime(2025-01-19), 6)
    """
    print(f"\n🔍 PROBING DATA AVAILABILITY")
    print(f"   Starting from: {simulated_today.strftime('%Y-%m-%d')} - {MIN_PBI_LAG_DAYS} days")
    print(f"   Max lag to try: {MAX_PBI_LAG_DAYS} days")
    print("=" * 50)
    
    collector = PBIDataCollector(environment=environment)
    
    for lag in range(MIN_PBI_LAG_DAYS, MAX_PBI_LAG_DAYS + 1):
        candidate_date = simulated_today - timedelta(days=lag)
        print(f"\n   🔍 Checking {candidate_date.strftime('%Y-%m-%d')} (lag: -{lag} days)...")
        
        try:
            # Create a temporary directory for the probe
            with tempfile.TemporaryDirectory() as tmp_folder:
                # Lightweight probe: 1 day aggregation, Global node only
                # This is fast because we only need to check if ANY data exists
                results = await collector.collect_flexible_data_for_node(
                    "Global",  # Always check Global - if Global has data, segments will too
                    aggregation_days=1,
                    target_folder=tmp_folder,
                    analysis_date=candidate_date
                )
                
                # Check if NPS data was collected AND has data for Period 1 (the target date)
                if results.get('flexible_NPS', False):
                    # Read the CSV and check if Period 1 exists with valid NPS data
                    nps_file = Path(tmp_folder) / "Global" / "flexible_NPS_1d.csv"
                    if nps_file.exists():
                        df = pd.read_csv(nps_file)
                        
                        # Create Unified_NPS from year-specific columns (same logic as FlexibleAnomalyDetector)
                        year_cols = ['NPS_2026', 'NPS_2025', 'NPS_2024', 'NPS_2019']
                        df['Unified_NPS'] = pd.NA
                        for col in year_cols:
                            if col in df.columns:
                                df['Unified_NPS'] = df['Unified_NPS'].fillna(df[col])
                        
                        # Check if Period_Group 1 exists and has valid NPS
                        if 'Period_Group' in df.columns:
                            period_1_data = df[df['Period_Group'] == 1]
                            if not period_1_data.empty and period_1_data['Unified_NPS'].notna().any():
                                nps_value = period_1_data['Unified_NPS'].dropna().iloc[0] if not period_1_data['Unified_NPS'].dropna().empty else None
                                print(f"   ✅ DATA FOUND for {candidate_date.strftime('%Y-%m-%d')} (lag: {lag} days) - NPS: {nps_value:.1f}")
                                print("=" * 50)
                                return candidate_date, lag
                            else:
                                print(f"      ⚠️ No data for Period 1 (target date) - trying older date...")
                        else:
                            print(f"      ⚠️ Period_Group column not found - trying older date...")
                    else:
                        print(f"      ⚠️ NPS file not found - trying older date...")
                else:
                    print(f"      ⚠️ No NPS data collected - trying older date...")
                    
        except Exception as e:
            print(f"      ⚠️ Probe error: {str(e)[:100]} - trying older date...")
    
    # No data found within max lag
    error_msg = (
        f"❌ No data available within {MAX_PBI_LAG_DAYS} days of "
        f"{simulated_today.strftime('%Y-%m-%d')}. "
        f"Checked dates from {(simulated_today - timedelta(days=MIN_PBI_LAG_DAYS)).strftime('%Y-%m-%d')} "
        f"to {(simulated_today - timedelta(days=MAX_PBI_LAG_DAYS)).strftime('%Y-%m-%d')}."
    )
    print(f"\n{error_msg}")
    raise ValueError(error_msg)


async def generate_consolidated_summary(
    agent, 
    consolidated_data: List[Dict], 
    date_flight_local: str = None, 
    segment: str = 'Global',
    adaptive_card_weekly: str = None,
    adaptive_cards_daily: List[str] = None
) -> str:
    """Generate a consolidated summary from multiple analysis types including weekly comparative and daily single analyses.
    
    Args:
        agent: The AnomalySummaryAgent instance
        consolidated_data: List of consolidated analysis data
        date_flight_local: Local flight date for context
        segment: The root segment for hierarchical analysis (default: 'Global')
        adaptive_card_weekly: Optional Adaptive Card JSON for weekly analysis
        adaptive_cards_daily: Optional list of Adaptive Card JSONs for daily analyses
    """
    
    # If Adaptive Cards are provided directly, use the new method
    if adaptive_card_weekly or adaptive_cards_daily:
        print("📊 Using Adaptive Card-based summary generation...")
        
        # Prepare Adaptive Cards data for summary agent
        adaptive_cards_data = []
        
        # Add weekly Adaptive Card if available
        if adaptive_card_weekly:
            # Extract date range from consolidated_data if available
            date_range = "Unknown"
            if consolidated_data and isinstance(consolidated_data[0], dict):
                metadata = consolidated_data[0].get('metadata', {})
                date_range = metadata.get('analysis_date', 'Unknown')
            
            adaptive_cards_data.append({
                'adaptive_card_json': adaptive_card_weekly,
                'period_type': 'weekly',
                'date_range': date_range
            })
        
        # Add daily Adaptive Cards if available
        if adaptive_cards_daily:
            daily_data = []
            if consolidated_data and isinstance(consolidated_data[0], dict):
                daily_data = consolidated_data[0].get('daily_singles', [])
            
            for i, daily_card in enumerate(adaptive_cards_daily):
                if daily_card:
                    # Extract date from daily_single_analyses if available
                    date = daily_data[i].get('date', f'Day {i+1}') if i < len(daily_data) else f'Day {i+1}'
                    adaptive_cards_data.append({
                        'adaptive_card_json': daily_card,
                        'period_type': 'daily',
                        'date_range': date
                    })
        
        if adaptive_cards_data:
            try:
                comprehensive_summary = await asyncio.wait_for(
                    agent.generate_summary_from_adaptive_cards(adaptive_cards_data),
                    timeout=3600.0
                )
                return comprehensive_summary
            except Exception as e:
                print(f"❌ Error in generate_summary_from_adaptive_cards: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to text-based method
                print("⚠️ Falling back to text-based summary generation...")
    
    # Check if it's the weekly format (with 'weekly_comparative' and 'daily_singles' keys)
    if consolidated_data and isinstance(consolidated_data[0], dict) and 'weekly_comparative' in consolidated_data[0]:
        # New format from weekly_deep_research.py
        data = consolidated_data[0]
        weekly_data = data.get('weekly_comparative', '')
        daily_data = data.get('daily_singles', [])
        
        # Format weekly data (it's a list of periods from execute_analysis_flow)
        analysis_period_range = None
        interpreter_adaptive_card = None
        if isinstance(weekly_data, list) and weekly_data:
            weekly_comparative_analysis = ""
            for period in weekly_data:
                date_range = period.get('date_range', 'Unknown')
                interpretation = period.get('ai_interpretation', '')
                # Separate text from adaptive card JSON (same format as summarizer)
                if "---ADAPTIVE_CARD_JSON---" in interpretation:
                    text_part, card_part = interpretation.split("---ADAPTIVE_CARD_JSON---", 1)
                    interpretation = text_part.strip()
                    interpreter_adaptive_card = card_part.strip()
                weekly_comparative_analysis += f"{interpretation}\n\n"
                if not analysis_period_range and date_range and date_range != 'Unknown':
                    analysis_period_range = date_range
        elif isinstance(weekly_data, str):
            weekly_comparative_analysis = weekly_data
        else:
            weekly_comparative_analysis = ""
        
        # Daily data is already formatted
        daily_single_analyses = daily_data
        
        # Call generate_comprehensive_summary_stratified (3-step approach)
        try:
            # Prepare metadata and date ranges for the summary agent
            execution_metadata = data.get('metadata', {})
            weekly_params = data.get('weekly_params', {})
            daily_params = data.get('daily_params', {})
            
            # Build date_ranges with the actual analysis period range
            analysis_date_str = execution_metadata.get('analysis_date')
            date_ranges = {}
            if analysis_date_str:
                date_ranges['analysis_date'] = analysis_date_str
            if analysis_period_range:
                date_ranges['analysis_period_range'] = analysis_period_range
            date_ranges = date_ranges or None

            comprehensive_summary = await asyncio.wait_for(
                agent.generate_comprehensive_summary_stratified(
                    weekly_comparative_analysis=weekly_comparative_analysis,
                    daily_single_analyses=daily_single_analyses,
                    date_flight_local=date_flight_local,
                    execution_metadata=execution_metadata,
                    weekly_analysis_params=weekly_params,
                    daily_analysis_params=daily_params,
                    date_ranges=date_ranges,
                    segment=segment
                ),
                timeout=3600.0  # Increased timeout for 3-step process + payload optimization
            )
            return comprehensive_summary
        except Exception as e:
            print(f"❌ Error in generate_comprehensive_summary_stratified: {e}")
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
    daily_periods: int = 7,
    environment: str = "prod",
    focus_touchpoint: Optional[str] = None,
):
    """
    Comprehensive weekly analysis orchestrator.
    Runs three sequential steps:
      1. Weekly comparative flow (7d) — CLAUDE_OPUS_4_6 interpreter.
      2. Daily single analyses (1d × N days) — CLAUDE_OPUS_4_5 interpreter (cost-optimised).
      3. Executive summarizer — CLAUDE_OPUS_4_6; uploads comprehensive report to S3.
    """
    execution_id = str(uuid.uuid4())
    print("🚀 WEEKLY DEEP RESEARCH - Comprehensive NPS Analysis")
    print("=" * 80)
    print(f"📅 Analysis Date: {analysis_date.strftime('%Y-%m-%d')} ({date_parameter})")
    print(f"🎯 Segment Focus: {segment}")
    print(f"🆔 Execution ID: {execution_id}")

    generated_reports = []

    # --- SEQUENTIAL EXECUTION: Weekly first, then Daily Analysis ---
    print("\n" + "=" * 40)
    print("🚀 SEQUENTIAL EXECUTION: Weekly Comparative first, then Daily Single Analyses")
    print("=" * 40)
    
    # Determine anomaly detection mode based on causal filter
    comp_start_str = comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date and hasattr(comparison_start_date, 'strftime') else str(comparison_start_date) if comparison_start_date else None
    comp_end_str = comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date and hasattr(comparison_end_date, 'strftime') else str(comparison_end_date) if comparison_end_date else None
    anomaly_mode, baseline_desc = determine_anomaly_mode_for_vslast(causal_filter, comp_start_str, comp_end_str)
    print(f"🎯 Weekly Anomaly Detection Mode: {anomaly_mode} (baseline: {baseline_desc})")
    print(f"🎯 Daily Anomaly Detection Mode: {daily_anomaly_detection_mode} (periods: {daily_periods})")
    
    async def run_weekly_analysis():
        """Execute weekly comparative analysis"""
        print("\n📊 [STEP 1] Starting Weekly Comparative Analysis...")
        try:
            result = await execute_analysis_flow(
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
                environment=environment,
                focus_touchpoint=focus_touchpoint,
            )
            
            # Check for success (list or non-error string)
            # Avoid using "Error" in str(result) as it might appear in the content
            is_success = isinstance(result, list) or (isinstance(result, str) and not result.strip().startswith("Error") and not result.strip().startswith("❌"))
            
            if result and is_success:
                print(f"✅ [STEP 1] Weekly analysis completed. Data length: {len(str(result))} chars")
                return {'type': 'weekly', 'data': result, 'success': True}
            else:
                print(f"⚠️ [STEP 1] Weekly analysis did not generate a report.")
                return {'type': 'weekly', 'data': None, 'success': False}
        except Exception as e:
            print(f"❌ [STEP 1] Weekly analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'weekly', 'data': None, 'success': False, 'error': str(e)}
    
    async def run_daily_analysis():
        """Execute daily single analysis for each of the last N days"""
        print(f"\n📊 [STEP 2] Starting Daily Analysis ({daily_periods} days) — model: {DAILY_LLM_TYPE.value}...")
        try:
            result = await execute_analysis_flow(
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
                environment=environment,
                focus_touchpoint=focus_touchpoint,
                llm_type=DAILY_LLM_TYPE,
            )
            
            # Check for success (list or non-error string)
            # Avoid using "Error" in str(result) as it might appear in the content
            is_success = isinstance(result, list) or (isinstance(result, str) and not result.strip().startswith("Error") and not result.strip().startswith("❌"))
            
            if result and is_success:
                print(f"✅ [STEP 2] Daily analysis completed. Data length: {len(str(result))} chars")
                return {'type': 'daily', 'data': result, 'success': True}
            else:
                print(f"❌ [STEP 2] Daily analysis failed. Reason: {result}")
                return {'type': 'daily', 'data': None, 'success': False}
        except Exception as e:
            print(f"❌ [STEP 2] Daily analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'daily', 'data': None, 'success': False, 'error': str(e)}
    
    # Execute analyses SEQUENTIALLY: Weekly first, then Daily
    # This prevents resource contention and API throttling issues
    print("\n⚡ Executing WEEKLY analysis first (sequential mode)...")
    
    # STEP 1: Run weekly analysis and wait for completion
    weekly_result = await run_weekly_analysis()
    
    if isinstance(weekly_result, Exception):
        print(f"❌ Weekly analysis failed with exception: {weekly_result}")
    elif isinstance(weekly_result, dict) and weekly_result.get('success') and weekly_result.get('data'):
        generated_reports.append({
            'type': weekly_result['type'],
            'data': weekly_result['data']
        })
        print(f"✅ Weekly analysis: SUCCESS")
    else:
        print(f"⚠️ Weekly analysis: No data generated")
    
    # STEP 2: Only after weekly is complete, run daily analysis
    # Daily interpreter uses DAILY_LLM_TYPE (CLAUDE_OPUS_4_5) to reduce cost.
    print("\n⚡ Weekly complete. Now executing DAILY analyses...")
    daily_result = await run_daily_analysis()

    if isinstance(daily_result, Exception):
        print(f"❌ Daily analysis failed with exception: {daily_result}")
    elif isinstance(daily_result, dict) and daily_result.get('success') and daily_result.get('data'):
        generated_reports.append({
            'type': daily_result['type'],
            'data': daily_result['data']
        })
        print(f"✅ Daily analysis: SUCCESS")
    else:
        print(f"⚠️ Daily analysis: No data generated")
    
    # Summary of sequential execution results
    print("\n" + "=" * 40)
    print("📋 SEQUENTIAL EXECUTION RESULTS")
    print("=" * 40)
    print(f"✅ Total reports generated: {len(generated_reports)}")

    # --- STEP 3: Executive Summarizer ---
    # Summarizer uses SUMMARY_LLM_TYPE (CLAUDE_OPUS_4_6) for highest quality executive report.
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
                if isinstance(weekly_comparative_analysis, list):
                    print(f"✅ Found weekly report: {len(weekly_comparative_analysis)} items")
                else:
                    print(f"✅ Found weekly report: {len(str(weekly_comparative_analysis))} chars")
            elif report['type'] == 'daily':
                daily_single_analyses = report['data']
                print(f"✅ Found daily analysis data: {len(daily_single_analyses)} periods")

        # Convert daily data to the format expected by summary agent if needed
        interpreter_adaptive_cards_daily = []
        if daily_single_analyses and isinstance(daily_single_analyses, list):
            formatted_daily_analyses = []
            for daily in daily_single_analyses:
                if isinstance(daily, dict) and 'ai_interpretation' in daily:
                    analysis_text = daily.get('ai_interpretation', '')
                    if "---ADAPTIVE_CARD_JSON---" in analysis_text:
                        text_part, card_part = analysis_text.split("---ADAPTIVE_CARD_JSON---", 1)
                        analysis_text = text_part.strip()
                        interpreter_adaptive_cards_daily.append(card_part.strip())
                    formatted_daily_analyses.append({
                        'date': daily.get('date_range', daily.get('period', 'Unknown')),
                        'analysis': analysis_text,
                        'anomalies': ['daily_analysis']
                    })
            daily_single_analyses = formatted_daily_analyses
            print(f"✅ Formatted daily analyses: {len(daily_single_analyses)} periods")

        # Use Summary Agent to consolidate
        try:
            print(f"\n🤖 Initializing Summary Agent — model: {SUMMARY_LLM_TYPE.value}...")
            summary_logger = logging.getLogger("summary_agent")
            summary_logger.setLevel(logging.INFO)
            if not summary_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                summary_logger.addHandler(handler)

            summary_agent = AnomalySummaryAgent(
                llm_type=SUMMARY_LLM_TYPE,
                logger=summary_logger,
                environment=environment,
                study_mode="comparative",
                anomaly_detection_mode="vslast",
                causal_filter=causal_filter,
                comparison_start_date=str(comparison_start_date) if comparison_start_date else None,
                comparison_end_date=str(comparison_end_date) if comparison_end_date else None,
                aggregation_days=7,
                baseline_periods=7,
                segment=segment,
                focus_touchpoint=focus_touchpoint,
                execution_id=execution_id,
            )
            print("✅ Summary Agent initialized. Generating executive summary...")

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
                date_flight_local,
                segment=segment
            )

            print("\n" + "=" * 80)
            print("📋 EXECUTIVE SUMMARY")
            print("=" * 80)
            print(executive_summary)
            print("=" * 80)

            # Upload to S3
            try:
                print("\n📤 Uploading comprehensive report to S3...")
                s3_uploader = S3ReportUploader(environment=environment)

                date_ranges = {
                    'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                    'comparison_start_date': comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date else None,
                    'comparison_end_date': comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date else None
                }

                final_synthesis_to_upload = executive_summary
                if environment == "prod" and "---ADAPTIVE_CARD_JSON---" in executive_summary:
                    card_json_str = executive_summary.split("---ADAPTIVE_CARD_JSON---")[-1].strip()
                    try:
                        final_synthesis_to_upload = json.loads(card_json_str)
                    except Exception as e:
                        print(f"⚠️ Could not parse adaptive card JSON: {e}")
                        final_synthesis_to_upload = card_json_str

                s3_key = await s3_uploader.upload_comprehensive_report(
                    execution_date=datetime.now(),
                    analysis_date=analysis_date.strftime('%Y-%m-%d'),
                    segment=segment,
                    explanation_mode="weekly_comprehensive",
                    causal_filter=causal_filter,
                    weekly_analysis_params=weekly_analysis_params,
                    daily_analysis_params=daily_analysis_params,
                    date_ranges=date_ranges,
                    final_synthesis=final_synthesis_to_upload,
                    comparison_start_date=comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date else None,
                    comparison_end_date=comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date else None
                )

                if s3_key:
                    print(f"✅ Report uploaded to S3: {s3_key}")
                else:
                    print("⚠️ S3 upload skipped (empty synthesis or local environment)")

            except Exception as s3_error:
                print(f"⚠️ S3 upload failed (non-critical): {s3_error}")

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
                       help='Simulate today being this date (YYYY-MM-DD). Auto-probes for latest available data (starting from -4 days, up to -10 days)')
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
    
    # Environment parameter
    parser.add_argument('--environment', type=str, default='prod', choices=['local', 'prod'],
                       help='Environment: local (reads .env) or prod (uses system env vars). Default: local')
    
    # Focus touchpoint parameter
    VALID_TOUCHPOINTS = [
        "Response provided to the issue", "Wi-Fi", "Ease of contact by phone",
        "Ease of contact by IB Plus email", "In flight food and beverage",
        "Connections experience", "IFE", "IB Plus loyalty program",
        "Journey preparation support", "Aircraft interior", "Boarding",
        "Lounge", "Comms", "Punctuality", "Arrivals experience",
        "Cabin Crew", "Check-in", "Pilot's announcements", "Airport security",
    ]
    parser.add_argument('--focus-touchpoint', type=str, default=None,
                       choices=VALID_TOUCHPOINTS,
                       help='Touchpoint a investigar en profundidad (filtered_name del modelo PBI)')
    
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
            simulated_today = datetime.strptime(args.insert_date_ci, '%Y-%m-%d')
            
            # Probe for the latest available data date (handles variable PBI lag)
            try:
                analysis_date, actual_lag = await find_latest_available_date(
                    simulated_today=simulated_today,
                    segment=args.segment,
                    environment=args.environment
                )
                date_parameter = 'insert_ci'
                date_description = f"Simulating today as {simulated_today.strftime('%Y-%m-%d')} (actual lag: {actual_lag} days)"
            except ValueError as probe_error:
                print(f"❌ {probe_error}")
                return
                
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
    print(f"   • Analysis date (end of 7-day window): {analysis_date.strftime('%Y-%m-%d')}")
    if date_parameter == 'insert_ci':
        week_start = analysis_date - timedelta(days=6)
        print(f"   • Week analyzed: {week_start.strftime('%Y-%m-%d')} to {analysis_date.strftime('%Y-%m-%d')}")
        print(f"   • Note: Data availability was auto-probed from {args.insert_date_ci}")
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
            daily_baseline_periods=args.daily_baseline_periods,
            daily_aggregation_days=1,
            daily_periods=7,
            environment=args.environment,
            focus_touchpoint=args.focus_touchpoint or None,
        )
    except KeyboardInterrupt:
        print("\n⏸️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")


def cli_main():
    """Synchronous entry point for CLI (used by pyproject.toml scripts)."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()

