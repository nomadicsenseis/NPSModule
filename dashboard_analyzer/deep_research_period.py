#!/usr/bin/env python3
"""
Deep Research Period - Custom NPS Anomaly Analysis
Flexible analysis with customizable parameters (aggregation days, periods, detection mode, etc.)
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import sys
import os
import json
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.data_collection.s3_report_uploader import S3ReportUploader
from dashboard_analyzer.anomaly_detection.flexible_detector import FlexibleAnomalyDetector
from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter
from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type

# Global debug flag
DEBUG_MODE = True

def debug_print(message):
    """Print debug message only when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG: {message}")

def generate_comparison_context(anomaly_detection_mode: str, aggregation_days: int, baseline_periods: int = 7, baseline_description: str = None) -> str:
    """
    Generate comparison context explanation based on anomaly detection mode, aggregation days and baseline periods
    
    Args:
        anomaly_detection_mode: 'vslast', 'vslast_dynamic', 'mean', or 'target'
        aggregation_days: Number of days per period (1, 7, 14, 30, etc.)
        baseline_periods: Number of periods to use as baseline (default: 7)
        baseline_description: Dynamic baseline description (for vslast_dynamic)
    
    Returns:
        String explaining the comparison context
        
    IMPORTANT TERMINOLOGY:
    - For WEEKLY analysis (aggregation_days >= 7): "vs semana anterior" or "vs últimos X días" (NO "media")
    - For DAILY analysis (aggregation_days == 1): "vs media de los últimos X días" (WITH "media")
    """
    if anomaly_detection_mode == 'vslast':
        if aggregation_days == 7:
            return "• **Comparación**: vs semana anterior"
        elif aggregation_days == 1:
            return "• **Comparación**: vs día anterior"
        else:
            return f"• **Comparación**: vs período previo ({aggregation_days} días)"
    elif anomaly_detection_mode == 'vslast_dynamic':
        # Use dynamic baseline description if provided
        if baseline_description:
            # For weekly analysis, don't use "media" terminology
            if aggregation_days >= 7:
                return f"• **Comparación**: vs {baseline_description}"
            else:
                # For daily analysis, use "media" terminology
                return f"• **Comparación**: vs media de los {baseline_description}"
        else:
            # Fallback based on aggregation days
            if aggregation_days >= 7:
                return f"• **Comparación**: vs semana anterior"
            else:
                return f"• **Comparación**: vs media de los últimos {baseline_periods} días"
    elif anomaly_detection_mode == 'mean':
        if aggregation_days == 1:
            return f"• **Comparación**: vs media de los últimos {baseline_periods} días"
        else:
            # For weekly/multi-day periods, don't use "media" - compare against previous period
            return f"• **Comparación**: vs período anterior ({baseline_periods} períodos de {aggregation_days} días)"
    elif anomaly_detection_mode == 'target':
        return "• **Comparación**: vs target mensual establecido"
    else:
        return f"• **Comparación**: modo '{anomaly_detection_mode}' no reconocido"

def debug_save_hierarchical_data(hierarchical_explanation: str, period: int, date_param: Optional[str] = None, 
                                causal_explanations: Optional[dict] = None, relationships: Optional[dict] = None):
    """Save hierarchical explanation data for interpreter debugging"""
    try:
        # Create debug folder
        debug_folder = Path("dashboard_analyzer/agent_conversations/interpreter_debug")
        debug_folder.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hierarchical_data_period_{period}_{timestamp}.json"
        filepath = debug_folder / filename
        
        # Prepare debug data
        debug_data = {
            "timestamp": timestamp,
            "period": period,
            "date_param": date_param,
            "hierarchical_explanation": hierarchical_explanation,
            "causal_explanations": causal_explanations or {},
            "relationships": relationships or {},
            "metadata": {
                "total_nodes": len(causal_explanations) if causal_explanations else 0,
                "explanation_length": len(hierarchical_explanation),
                "nodes_analyzed": list(causal_explanations.keys()) if causal_explanations else []
            }
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        print(f"🔧 DEBUG: Hierarchical data saved to {filepath}")
        
    except Exception as e:
        print(f"⚠️ DEBUG: Failed to save hierarchical data: {e}")

async def collect_flexible_data(aggregation_days: int, target_folder: str, segment: str = "Global", analysis_date: datetime = None, environment: str = "prod"):
    """
    Collect flexible NPS data for all nodes in the specified segment
    
    Args:
        aggregation_days: Number of days per period
        target_folder: Where to save the data
        segment: Root segment to collect (Global, SH, LH, etc.)
        analysis_date: Optional analysis date to use instead of TODAY() in queries
        environment: Environment type ("local" or "prod")
    """
    print(f"📥 Collecting flexible NPS data")
    print(f"   🔧 Aggregation: {aggregation_days} days per period")
    print(f"   📁 Target folder: {target_folder}")
    print(f"   🎯 Segment: {segment}")
    if analysis_date:
        print(f"   📅 Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
    
    collector = PBIDataCollector(environment=environment)
    node_paths = get_segment_node_paths(segment)
    
    total_attempted = 0
    total_success = 0
    
    for node_path in node_paths:
        try:
            debug_print(f"Collecting data for node: {node_path}")
            results = await collector.collect_flexible_data_for_node(
                node_path, aggregation_days, target_folder, analysis_date
            )
            total_attempted += len(results)
            total_success += sum(results.values())
            debug_print(f"Node {node_path}: {sum(results.values())}/{len(results)} files successful")
        except Exception as e:
            print(f"❌ Error collecting data for {node_path}: {e}")
            debug_print(f"Error details for {node_path}: {type(e).__name__}: {str(e)}")
    
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

async def generate_explanations(analysis_data: dict, causal_filter: str = "vs L7d", environment: str = "prod"):
    """Generate comprehensive explanations for nodes with anomalies"""
    if not analysis_data:
        return
    
    print(f"\n📝 STEP 3: Comprehensive Anomaly Explanations")
    print("-" * 60)
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    anomaly_periods = analysis_data['anomaly_periods']
    aggregation_days = analysis_data['aggregation_days']
    
    # Initialize PBI collector and interpreter with full capabilities
    print("🔧 Initializing data collectors...")
    pbi_collector = PBIDataCollector(environment=environment)
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector, causal_filter=causal_filter, environment=environment)
    
    explanation_count = 0
    total_nodes_analyzed = 0
    
    for period in anomaly_periods[:4]:  # Analyze up to 4 periods with anomalies
        print(f"\n{'='*50}")
        print(f"🔍 PERIOD {period} EXPLANATIONS")
        print("="*50)
        
        # Get anomalies for this period
        period_anomalies, period_deviations, _, _ = await detector.analyze_period(data_folder, period, analysis_data.get('analysis_date'))
        
        # Find nodes with anomalies
        nodes_needing_explanation = [
            node for node, state in period_anomalies.items() 
            if state in ['+', '-']
        ]
        
        if not nodes_needing_explanation:
            print(f"   ✅ No anomalies found in period {period}")
            continue
        
        # Get date range for this period
        date_range = interpreter._get_period_date_range(period, aggregation_days)
        if date_range:
            start_date, end_date = date_range
            print(f"📅 Period {period} Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            print(f"⚠️ Could not determine date range for period {period}")
        
        print(f"🎯 Found {len(nodes_needing_explanation)} anomalies to explain:")
        
        # Generate explanations for each anomalous node
        for i, node_path in enumerate(nodes_needing_explanation, 1):  # Process ALL anomalous nodes
            total_nodes_analyzed += 1
            try:
                print(f"\n   {i}. 📊 Analyzing {node_path}")
                print(f"      {'─' * 40}")
                
                # Get the anomaly details
                deviation = period_deviations.get(node_path, 0)
                state = period_anomalies.get(node_path, "?")
                state_desc = "📈 Higher than expected" if state == "+" else "📉 Lower than expected"
                state_icon = "🔺" if state == "+" else "🔻"
                
                print(f"      {state_icon} Status: {state_desc} ({deviation:+.1f} points)")
                
                # Generate comprehensive explanation with timeout
                print(f"      🔍 Collecting explanatory data...")
                explanation = await asyncio.wait_for(
                    interpreter.explain_anomaly(
                        node_path=node_path,
                        target_period=period,
                        aggregation_days=aggregation_days,
                        anomaly_state=state,
                        causal_filter=causal_filter
                    ),
                    timeout=600.0  # 10 minute timeout for comprehensive analysis
                )
                
                # Display the explanation in a structured way
                print(f"      💡 EXPLANATION:")
                explanation_lines = explanation.split(" | ")
                for line in explanation_lines:
                    if line.strip():
                        print(f"         • {line.strip()}")
                
                explanation_count += 1
                
            except asyncio.TimeoutError:
                print(f"      ⏰ Timeout generating explanation for {node_path} (>60s)")
                print(f"         This node requires manual investigation")
            except Exception as e:
                print(f"      ❌ Error generating explanation for {node_path}: {str(e)}")
                print(f"         Check data availability and node path validity")
    
    # Final explanation summary
    print(f"\n{'='*60}")
    print(f"📋 EXPLANATION SUMMARY")
    print("="*60)
    print(f"   📊 Total anomalous nodes analyzed: {total_nodes_analyzed}")
    print(f"   ✅ Successful explanations generated: {explanation_count}")
    if total_nodes_analyzed > 0:
        success_rate = (explanation_count / total_nodes_analyzed) * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")
    
    if explanation_count == 0:
        print(f"   ⚠️ No explanations could be generated")
        print(f"   💡 Possible issues:")
        print(f"      • Missing operational data files")
        print(f"      • PBI connection issues for verbatims")
        print(f"      • Invalid date ranges")
    else:
        print(f"   🎉 Explanations successfully generated!")
        print(f"   💡 Each explanation includes:")
        print(f"      • 🔧 Operational metrics analysis")
        print(f"      • 💬 Customer verbatims sentiment")
        print(f"      • 📅 Date-filtered data for specific periods")

async def process_single_period(
    period: int,
    analysis_data: dict,
    segment: str,
    causal_filter: str,
    comparison_start_date: datetime,
    comparison_end_date: datetime,
    environment: str,
    study_mode: str,
    pbi_collector,
    ai_agent,
    ai_available: bool,
    period_semaphore: asyncio.Semaphore
) -> dict:
    """
    Process a single period's analysis - designed to run in parallel with other periods.
    
    Args:
        period: Period number to analyze
        analysis_data: Dict containing detector, data_folder, aggregation_days, etc.
        segment: Segment to analyze
        causal_filter: Causal comparison filter
        comparison_start_date: Start date for comparison period
        comparison_end_date: End date for comparison period
        environment: Environment (local/prod)
        study_mode: "single" or "comparative"
        pbi_collector: Shared PBI data collector
        ai_agent: AI interpreter agent (can be None)
        ai_available: Whether AI agent is available
        period_semaphore: Semaphore to limit concurrent period processing
    
    Returns:
        Dict with period analysis results
    """
    async with period_semaphore:
        detector = analysis_data['detector']
        data_folder = analysis_data['data_folder']
        aggregation_days = analysis_data['aggregation_days']
        
        print(f"\n{'='*60}")
        print(f"⚡ PERIOD {period} ANALYSIS (PARALLEL)")
        print("="*60)
        
        # Create isolated interpreter for this period
        interpreter = FlexibleAnomalyInterpreter(
            data_folder, 
            pbi_collector=pbi_collector, 
            causal_filter=causal_filter, 
            detection_mode=detector.detection_mode, 
            comparison_start_date=comparison_start_date, 
            comparison_end_date=comparison_end_date, 
            environment=environment
        )
        
        # Get anomalies for this period
        analyze_result = await detector.analyze_period(data_folder, period, analysis_data.get('analysis_date'))
        period_anomalies, period_deviations, _, period_nps_values = analyze_result
        
        # Get date range for this period
        date_range = interpreter._get_period_date_range(period, aggregation_days)
        if date_range:
            start_date, end_date = date_range
            date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            print(f"📅 Period {period}: {date_range_str}")
        else:
            date_range_str = "Unknown dates"
        
        # Generate parent interpretations
        parent_interpretations = generate_parent_interpretations(period_anomalies)
        
        # Collect explanations for anomalous nodes
        explanations = {}
        nodes_with_anomalies = [node for node, state in period_anomalies.items() if state in ['+', '-']]

        # ENHANCEMENT: Always include root segment
        root_segment = normalize_segment_to_root(segment)
        root_state = period_anomalies.get(root_segment, "?")
        root_deviation = period_deviations.get(root_segment, 0.0)
        
        if root_deviation > 0:
            root_state = "+"
        elif root_deviation < 0:
            root_state = "-"
        else:
            root_state = "N"
        period_anomalies[root_segment] = root_state
        
        if root_segment not in nodes_with_anomalies and root_state != "?":
            nodes_with_anomalies.append(root_segment)
        
        if root_segment in nodes_with_anomalies:
            nodes_with_anomalies.remove(root_segment)
            nodes_with_anomalies.insert(0, root_segment)

        detailed_tree_data = {}
        
        if nodes_with_anomalies:
            total_nodes = len(nodes_with_anomalies)
            successful_explanations = 0
            print(f"      📊 Period {period}: Processing {total_nodes} nodes...")
            
            # Semaphore for node-level concurrency within this period
            node_sem = asyncio.Semaphore(5)
            
            async def process_node_anomaly(i, node_path):
                async with node_sem:
                    try:
                        node_interpreter = FlexibleAnomalyInterpreter(
                            data_folder, 
                            pbi_collector=pbi_collector, 
                            causal_filter=causal_filter, 
                            detection_mode=detector.detection_mode, 
                            comparison_start_date=comparison_start_date, 
                            comparison_end_date=comparison_end_date, 
                            environment=environment
                        )
                        
                        anomaly_state = period_anomalies.get(node_path, "?")
                        deviation_value = period_deviations.get(node_path, 0.0)
                        
                        if node_path == "Global" and anomaly_state == "?":
                            if deviation_value > 0:
                                anomaly_state = "+"
                            elif deviation_value < 0:
                                anomaly_state = "-"
                            else:
                                anomaly_state = "N"
                        
                        # Calculate date range
                        node_start_date, node_end_date = None, None
                        analysis_date_local = analysis_data.get('analysis_date')
                        if analysis_date_local:
                            node_start_date, node_end_date = calculate_period_date_range(analysis_date_local, period, aggregation_days)
                        
                        # Build NPS context
                        nps_context = ""
                        comparison_context_local = ""
                        if period_nps_values and node_path in period_nps_values:
                            nps_data = period_nps_values[node_path]
                            if isinstance(nps_data, dict):
                                current_nps = nps_data.get('current', 'N/A')
                                baseline_nps = nps_data.get('baseline', 'N/A')
                                nps_context = f"Current NPS: {current_nps}, Baseline NPS: {baseline_nps}"
                                baseline_description = nps_data.get('baseline_description', None)
                                comparison_context_local = generate_comparison_context(
                                    analysis_data.get('anomaly_detection_mode', 'target'), 
                                    analysis_data.get('aggregation_days', 7),
                                    analysis_data.get('baseline_periods', 7),
                                    baseline_description
                                )
                            else:
                                nps_context = f"NPS: {nps_data}"
                        
                        explanation = await asyncio.wait_for(
                            node_interpreter.explain_anomaly(
                                node_path=node_path,
                                target_period=period,
                                aggregation_days=aggregation_days,
                                anomaly_state=anomaly_state,
                                start_date=node_start_date,
                                end_date=node_end_date,
                                anomaly_magnitude=deviation_value,
                                nps_context=nps_context,
                                causal_filter=causal_filter,
                                anomaly_detection_mode=analysis_data.get('anomaly_detection_mode', 'target'),
                                comparison_context=comparison_context_local,
                                baseline_periods=analysis_data.get('baseline_periods', 7)
                            ),
                            timeout=600.0
                        )
                        
                        investigation_log = []
                        if hasattr(node_interpreter, 'get_last_causal_log'):
                            investigation_log = node_interpreter.get_last_causal_log()
                        
                        return node_path, explanation, investigation_log, True
                    except Exception as e:
                        return node_path, f"Analysis failed: {e}", [], False

            # Process nodes in parallel
            tasks = [process_node_anomaly(i, node) for i, node in enumerate(nodes_with_anomalies, 1)]
            
            for task in asyncio.as_completed(tasks):
                node_path, result, log, success = await task
                explanations[node_path] = result
                if success and log:
                    detailed_tree_data[node_path] = {"explanation": result, "tool_trace": log}
                if success:
                    successful_explanations += 1
            
            print(f"      ✅ Period {period}: {successful_explanations}/{total_nodes} explanations collected")
        
        # AI Interpretation
        ai_interpretation = None
        if ai_available and ai_agent and nodes_with_anomalies:
            try:
                interpreter_comparison_context = generate_comparison_context(
                    analysis_data.get('anomaly_detection_mode', 'mean'),
                    analysis_data.get('aggregation_days', 1),
                    analysis_data.get('baseline_periods', 7)
                )

                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                                 parent_interpretations, explanations, date_range, segment, period_nps_values,
                                                 comparison_context=interpreter_comparison_context)
                
                date_param = None
                if date_range and len(date_range) >= 2:
                    range_start_date, range_end_date = date_range
                    if range_start_date and range_end_date:
                        date_param = f"{range_start_date.strftime('%Y-%m-%d')} to {range_end_date.strftime('%Y-%m-%d')}"
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, date_param, segment),
                    timeout=600.0
                )
                print(f"      🤖 Period {period}: AI interpretation completed")
                
            except Exception as e:
                ai_interpretation = f"AI interpretation failed: {str(e)}"
                print(f"      ⚠️ Period {period}: AI interpretation failed: {e}")
        
        # Return period results (JSON saving and tree printing will be done in consolidation phase)
        return {
            'period': period,
            'date_range': date_range,
            'date_range_str': date_range_str,
            'period_anomalies': period_anomalies,
            'period_deviations': period_deviations,
            'period_nps_values': period_nps_values,
            'parent_interpretations': parent_interpretations,
            'explanations': explanations,
            'detailed_tree_data': detailed_tree_data,
            'ai_interpretation': ai_interpretation,
            'nodes_with_anomalies': nodes_with_anomalies,
            'segment': segment,
            'aggregation_days': aggregation_days
        }


async def show_all_anomaly_periods_with_explanations(analysis_data: dict, segment: str = "Global", causal_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None, environment: str = "prod", study_mode: str = None):
    """Show trees for all periods analyzed INCLUDING explanations and parent interpretations.
    
    PARALLEL EXECUTION: All periods are processed concurrently for faster results.
    """
    if not analysis_data:
        return
    
    print(f"\n🌳 ANOMALY PERIOD ANALYSIS (PARALLEL MODE)")
    print("-" * 50)
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    aggregation_days = analysis_data['aggregation_days']
    anomaly_periods = analysis_data['anomaly_periods']
    periods_analyzed = analysis_data.get('periods_analyzed', anomaly_periods)
    
    # Determine study_mode if not provided
    if study_mode is None:
        if causal_filter is None or causal_filter == 'None':
            study_mode = "single"
        else:
            study_mode = "comparative"
    
    print(f"🔬 Study mode: {study_mode.upper()}")
    
    # Initialize interpreter for explanations with agent mode
    pbi_collector = PBIDataCollector(environment=environment)
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector, causal_filter=causal_filter, detection_mode=detector.detection_mode, comparison_start_date=comparison_start_date, comparison_end_date=comparison_end_date, environment=environment)
    print(f"🔧 Explanation mode: AGENT")
    
    # Initialize AI agent for interpretation
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType, get_default_llm_type
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=get_default_llm_type(),
            config_path="dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
            logger=logging.getLogger("ai_interpreter"),
            study_mode=study_mode,
            environment=environment
        )
        ai_available = True
        print("🤖 AI Agent initialized for interpretations")
    except Exception as e:
        print(f"⚠️ AI Agent not available: {str(e)}")
        ai_available = False
    
    # Collect data for all periods for summary
    all_periods_data = []
    
    # Show all periods with anomalies
    periods_with_anomalies = [p for p in periods_analyzed if p in anomaly_periods]
    
    print(f"\n⚡ PARALLEL EXECUTION: Processing {len(periods_with_anomalies)} periods concurrently...")
    
    # Create semaphore to limit concurrent period processing (to avoid overwhelming APIs)
    period_semaphore = asyncio.Semaphore(3)  # Process max 3 periods at a time
    
    # Create tasks for parallel execution
    period_tasks = [
        process_single_period(
            period=period,
            analysis_data=analysis_data,
            segment=segment,
            causal_filter=causal_filter,
            comparison_start_date=comparison_start_date,
            comparison_end_date=comparison_end_date,
            environment=environment,
            study_mode=study_mode,
            pbi_collector=pbi_collector,
            ai_agent=ai_agent if ai_available else None,
            ai_available=ai_available,
            period_semaphore=period_semaphore
        )
        for period in periods_with_anomalies
    ]
    
    # Execute all periods in parallel
    period_results = await asyncio.gather(*period_tasks, return_exceptions=True)
    
    print(f"\n{'='*60}")
    print("📋 CONSOLIDATING PARALLEL RESULTS")
    print("="*60)
    
    # Process results and save JSON files / print trees
    for result in period_results:
        if isinstance(result, Exception):
            print(f"❌ Period task failed: {result}")
            continue
        
        period = result['period']
        date_range_str = result['date_range_str']
        period_anomalies = result['period_anomalies']
        period_deviations = result['period_deviations']
        period_nps_values = result['period_nps_values']
        parent_interpretations = result['parent_interpretations']
        explanations = result['explanations']
        detailed_tree_data = result['detailed_tree_data']
        ai_interpretation = result['ai_interpretation']
        date_range = result['date_range']
        
        # Print tree for this period
        analysis_date = analysis_data.get('analysis_date')
        date_parameter = analysis_data.get('date_parameter')
        await print_enhanced_tree_with_explanations_and_interpretations(
            period_anomalies, period_deviations, explanations, parent_interpretations,
            aggregation_days, period, date_range, segment, analysis_date, date_parameter, period_nps_values
        )
        
        # Print AI interpretation if available
        if ai_interpretation and "AI interpretation failed" not in ai_interpretation:
            print(f"\n🤖 Period {period} AI INTERPRETATION:")
            print("-" * 40)
            if "📋 SÍNTESIS EJECUTIVA FINAL" in ai_interpretation:
                synthesis_start = ai_interpretation.find("📋 SÍNTESIS EJECUTIVA FINAL")
                if synthesis_start != -1:
                    synthesis_section = ai_interpretation[synthesis_start:]
                    end_markers = ["---", "✅ **ANÁLISIS COMPLETADO**", "*Este análisis utiliza"]
                    synthesis_end = len(synthesis_section)
                    for marker in end_markers:
                        marker_pos = synthesis_section.find(marker)
                        if marker_pos != -1:
                            synthesis_end = min(synthesis_end, marker_pos)
                    final_synthesis = synthesis_section[:synthesis_end].strip()
                    print("=" * 80)
                    print(final_synthesis)
                    print("=" * 80)
        
        # Save detailed JSON
        try:
            root_name = segment
            root_node = {"name": root_name, "children": []}
            
            def find_or_create_node(current_node, path_parts):
                if not path_parts:
                    return current_node
                target_name = path_parts[0]
                found_child = None
                if "children" not in current_node:
                    current_node["children"] = []
                for child in current_node["children"]:
                    if child["name"] == target_name:
                        found_child = child
                        break
                if not found_child:
                    found_child = {"name": target_name, "children": []}
                    current_node["children"].append(found_child)
                return find_or_create_node(found_child, path_parts[1:])

            all_node_paths = set(list(period_anomalies.keys()) + list(detailed_tree_data.keys()))
            
            for node_path in sorted(list(all_node_paths)):
                if node_path == root_name:
                    target_node = root_node
                elif node_path.startswith(root_name + "/"):
                    rel_path = node_path[len(root_name)+1:]
                    path_parts = rel_path.split("/")
                    target_node = find_or_create_node(root_node, path_parts)
                else:
                    continue
                
                target_node["anomaly_state"] = period_anomalies.get(node_path, "?")
                target_node["deviation"] = period_deviations.get(node_path, 0.0)
                
                if node_path in detailed_tree_data:
                    target_node["causal_analysis"] = detailed_tree_data[node_path]
                elif node_path in explanations:
                    target_node["causal_analysis"] = {"explanation": explanations[node_path]}

            json_output = {
                "metadata": {
                    "date_range": date_range_str,
                    "segment": segment,
                    "execution_timestamp": datetime.now().isoformat(),
                    "period": period,
                    "aggregation_days": aggregation_days
                },
                "tree": root_node,
                "ai_interpretation": ai_interpretation
            }
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_date = date_range_str.replace(" ", "_").replace(":", "-").replace("to", "_")
            safe_segment = segment.replace("/", "_")
            
            saved_successfully = False
            
            if environment == "prod":
                try:
                    s3_filename = f"detailed_tree_{safe_date}_{safe_segment}_{timestamp_str}.json"
                    s3_uploader = S3ReportUploader(environment=environment)
                    s3_key = await s3_uploader.upload_mapped_info(json_output, s3_filename)
                    if s3_key:
                        print(f"      ☁️ Period {period}: Uploaded to S3: {s3_key}")
                        saved_successfully = True
                except Exception as s3_error:
                    print(f"      ⚠️ Period {period}: S3 upload error: {s3_error}")
            
            if not saved_successfully:
                filename = f"logs/detailed_tree_{safe_date}_{safe_segment}_{timestamp_str}.json"
                os.makedirs("logs", exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                print(f"      💾 Period {period}: Saved to: {filename}")
            
        except Exception as e:
            print(f"      ⚠️ Period {period}: Failed to save JSON: {e}")

        # Collect period data for summary
        period_data = {
            'period': period,
            'date_range': date_range_str,
            'ai_interpretation': ai_interpretation or "No AI interpretation available"
        }
        all_periods_data.append(period_data)

    # Summary of all 7 periods
    print(f"\n📋 SUMMARY OF 7 PERIODS ANALYZED:")
    print("-" * 40)
    for period in periods_analyzed:
        period_anomalies, period_deviations, _, _ = await detector.analyze_period(data_folder, period, analysis_data.get('analysis_date'))
        anomaly_count = sum(1 for state in period_anomalies.values() if state in ['+', '-'])
        
        if period in anomaly_periods:
            status = f"🚨 {anomaly_count} anomalies"
        else:
            status = "✅ Normal"
        
        date_range = interpreter._get_period_date_range(period, aggregation_days)
        if date_range:
            start_date, end_date = date_range
            date_str = f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
        else:
            date_str = ""
        
        print(f"  Period {period}: {status} {date_str}")
    
    print(f"\n🎯 Total periods with anomalies: {len(anomaly_periods)}/7")
    
    # Note: Summary generation has been moved to weekly_deep_research.py
    # This function now only returns all_periods_data for consolidation
    return all_periods_data

def generate_parent_interpretations(anomalies: dict) -> dict:
    """Generate parent node interpretations based on children states"""
    interpretations = {}
    
    # Helper function to format child lists
    def format_children(children_list):
        if len(children_list) == 0:
            return "none"
        elif len(children_list) == 1:
            return children_list[0]
        elif len(children_list) == 2:
            return f"{children_list[0]}, {children_list[1]}"
        else:
            return ", ".join(children_list[:-1]) + f", {children_list[-1]}"
    
    # NOTE: IB/YW are leaf nodes - they don't get patterns since they have no children
    
    # Global interpretation (has LH, SH children)
    lh_state = anomalies.get("Global/LH", "?")
    sh_state = anomalies.get("Global/SH", "?")
    global_state = anomalies.get("Global", "?")
    
    if lh_state != "?" and sh_state != "?":
        if global_state == "N":
            # Normal parent
            if lh_state == "N" and sh_state == "N":
                interpretations["Global"] = "All children normal (LH, SH)"
            elif lh_state in ["+", "-"] and sh_state == "N":
                interpretations["Global"] = f"{lh_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (LH) diluted by normal nodes (SH)"
            elif sh_state in ["+", "-"] and lh_state == "N":
                interpretations["Global"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (SH) diluted by normal nodes (LH)"
            elif lh_state in ["+", "-"] and sh_state in ["+", "-"]:
                if lh_state != sh_state:
                    interpretations["Global"] = "LH and SH anomalies cancel each other out"
                else:
                    interpretations["Global"] = f"{lh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (LH, SH)"
        elif global_state in ["+", "-"]:
            # Anomalous parent
            if lh_state == "N" and sh_state == "N":
                interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite all children being normal (LH, SH)"
            elif lh_state in ["+", "-"] and sh_state == "N":
                if lh_state == global_state:
                    interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {lh_state.replace('+', 'positive').replace('-', 'negative')} LH, SH normal"
                else:
                    interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {lh_state.replace('+', 'positive').replace('-', 'negative')} LH offsetting SH normal"
            elif sh_state in ["+", "-"] and lh_state == "N":
                if sh_state == global_state:
                    interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {sh_state.replace('+', 'positive').replace('-', 'negative')} SH, LH normal"
                else:
                    interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {sh_state.replace('+', 'positive').replace('-', 'negative')} SH offsetting LH normal"
            elif lh_state in ["+", "-"] and sh_state in ["+", "-"]:
                if lh_state == sh_state == global_state:
                    interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (LH, SH)"
                else:
                    interpretations["Global"] = f"{global_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly from mixed LH ({lh_state}) and SH ({sh_state}) effects"

    # LH interpretation (has Economy, Business, Premium children)
    lh_children = ["Economy", "Business", "Premium"]
    lh_child_states = [anomalies.get(f"Global/LH/{child}", "?") for child in lh_children]
    valid_lh_children = [(child, state) for child, state in zip(lh_children, lh_child_states) if state != "?"]
    
    if valid_lh_children:
        normal_children = [child for child, state in valid_lh_children if state == "N"]
        positive_children = [child for child, state in valid_lh_children if state == "+"]
        negative_children = [child for child, state in valid_lh_children if state == "-"]
        
        if lh_state == "N":
            # Normal parent
            if len(positive_children) == 0 and len(negative_children) == 0:
                interpretations["Global/LH"] = f"All children normal ({format_children([c for c, _ in valid_lh_children])})"
            elif len(positive_children) > 0 and len(negative_children) > 0:
                interpretations["Global/LH"] = f"Mixed anomalies: positive ({format_children(positive_children)}), negative ({format_children(negative_children)}) balance out with normal ({format_children(normal_children)})"
            elif len(positive_children) > 0:
                interpretations["Global/LH"] = f"Positive nodes ({format_children(positive_children)}) diluted by normal nodes ({format_children(normal_children)})"
            elif len(negative_children) > 0:
                interpretations["Global/LH"] = f"Negative nodes ({format_children(negative_children)}) diluted by normal nodes ({format_children(normal_children)})"
        elif lh_state in ["+", "-"]:
            # Anomalous parent
            contributing_children = positive_children if lh_state == "+" else negative_children
            if contributing_children:
                interpretations["Global/LH"] = f"{lh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {lh_state.replace('+', 'positive').replace('-', 'negative')} {format_children(contributing_children)}"
            else:
                interpretations["Global/LH"] = f"{lh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite all children having different states"

    # SH interpretation (has Economy, Business children)
    economy_state = anomalies.get("Global/SH/Economy", "?")
    business_state = anomalies.get("Global/SH/Business", "?")
    sh_state = anomalies.get("Global/SH", "?")
    
    if economy_state != "?" and business_state != "?":
        if sh_state == "N":
            # Normal parent
            if economy_state == "N" and business_state == "N":
                interpretations["Global/SH"] = "All children normal (Economy, Business)"
            elif economy_state in ["+", "-"] and business_state == "N":
                interpretations["Global/SH"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (Economy) diluted by normal nodes (Business)"
            elif business_state in ["+", "-"] and economy_state == "N":
                interpretations["Global/SH"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (Business) diluted by normal nodes (Economy)"
            elif economy_state in ["+", "-"] and business_state in ["+", "-"]:
                if economy_state != business_state:
                    interpretations["Global/SH"] = "Economy and Business anomalies cancel each other out"
                else:
                    interpretations["Global/SH"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (Economy, Business)"
        elif sh_state in ["+", "-"]:
            # Anomalous parent
            if economy_state == "N" and business_state == "N":
                interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite all children being normal (Economy, Business)"
            elif economy_state in ["+", "-"] and business_state == "N":
                if economy_state == sh_state:
                    interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {economy_state.replace('+', 'positive').replace('-', 'negative')} Economy, Business normal"
                else:
                    interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {economy_state.replace('+', 'positive').replace('-', 'negative')} Economy offsetting Business normal"
            elif business_state in ["+", "-"] and economy_state == "N":
                if business_state == sh_state:
                    interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {business_state.replace('+', 'positive').replace('-', 'negative')} Business, Economy normal"
                else:
                    interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {business_state.replace('+', 'positive').replace('-', 'negative')} Business offsetting Economy normal"
            elif economy_state in ["+", "-"] and business_state in ["+", "-"]:
                if economy_state == business_state == sh_state:
                    interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (Economy, Business)"
                else:
                    interpretations["Global/SH"] = f"{sh_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly from mixed Economy ({economy_state}) and Business ({business_state}) effects"

    # SH/Economy interpretation (has IB, YW children)
    ib_eco_state = anomalies.get("Global/SH/Economy/IB", "?")
    yw_eco_state = anomalies.get("Global/SH/Economy/YW", "?")
    
    if ib_eco_state != "?" and yw_eco_state != "?":
        if economy_state == "N":
            # Normal parent
            if ib_eco_state == "N" and yw_eco_state == "N":
                interpretations["Global/SH/Economy"] = "All children normal (IB, YW)"
            elif ib_eco_state in ["+", "-"] and yw_eco_state == "N":
                interpretations["Global/SH/Economy"] = f"{ib_eco_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (IB) diluted by normal nodes (YW)"
            elif yw_eco_state in ["+", "-"] and ib_eco_state == "N":
                interpretations["Global/SH/Economy"] = f"{yw_eco_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (YW) diluted by normal nodes (IB)"
            elif ib_eco_state in ["+", "-"] and yw_eco_state in ["+", "-"]:
                if ib_eco_state != yw_eco_state:
                    interpretations["Global/SH/Economy"] = "IB and YW anomalies cancel each other out"
                else:
                    interpretations["Global/SH/Economy"] = f"{ib_eco_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (IB, YW)"
        elif economy_state in ["+", "-"]:
            # Anomalous parent
            if ib_eco_state == "N" and yw_eco_state == "N":
                interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite all children being normal (IB, YW)"
            elif ib_eco_state in ["+", "-"] and yw_eco_state == "N":
                if ib_eco_state == economy_state:
                    interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {ib_eco_state.replace('+', 'positive').replace('-', 'negative')} IB, YW normal"
                else:
                    interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {ib_eco_state.replace('+', 'positive').replace('-', 'negative')} IB offsetting YW normal"
            elif yw_eco_state in ["+", "-"] and ib_eco_state == "N":
                if yw_eco_state == economy_state:
                    interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {yw_eco_state.replace('+', 'positive').replace('-', 'negative')} YW, IB normal"
                else:
                    interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {yw_eco_state.replace('+', 'positive').replace('-', 'negative')} YW offsetting IB normal"
            elif ib_eco_state in ["+", "-"] and yw_eco_state in ["+", "-"]:
                if ib_eco_state == yw_eco_state == economy_state:
                    interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (IB, YW)"
                else:
                    interpretations["Global/SH/Economy"] = f"{economy_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly: mixed IB ({ib_eco_state}) and YW ({yw_eco_state}) effects"

    # SH/Business interpretation (has IB, YW children)
    ib_bus_state = anomalies.get("Global/SH/Business/IB", "?")
    yw_bus_state = anomalies.get("Global/SH/Business/YW", "?")
    
    if ib_bus_state != "?" and yw_bus_state != "?":
        if business_state == "N":
            # Normal parent
            if ib_bus_state == "N" and yw_bus_state == "N":
                interpretations["Global/SH/Business"] = "All children normal (IB, YW)"
            elif ib_bus_state in ["+", "-"] and yw_bus_state == "N":
                interpretations["Global/SH/Business"] = f"{ib_bus_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (IB) diluted by normal nodes (YW)"
            elif yw_bus_state in ["+", "-"] and ib_bus_state == "N":
                interpretations["Global/SH/Business"] = f"{yw_bus_state.replace('+', 'Positive').replace('-', 'Negative')} nodes (YW) diluted by normal nodes (IB)"
            elif ib_bus_state in ["+", "-"] and yw_bus_state in ["+", "-"]:
                if ib_bus_state != yw_bus_state:
                    interpretations["Global/SH/Business"] = "IB and YW anomalies cancel each other out"
                else:
                    interpretations["Global/SH/Business"] = f"{ib_bus_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (IB, YW)"
        elif business_state in ["+", "-"]:
            # Anomalous parent
            if ib_bus_state == "N" and yw_bus_state == "N":
                interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite all children being normal (IB, YW)"
            elif ib_bus_state in ["+", "-"] and yw_bus_state == "N":
                if ib_bus_state == business_state:
                    interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {ib_bus_state.replace('+', 'positive').replace('-', 'negative')} IB, YW normal"
                else:
                    interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {ib_bus_state.replace('+', 'positive').replace('-', 'negative')} IB offsetting YW normal"
            elif yw_bus_state in ["+", "-"] and ib_bus_state == "N":
                if yw_bus_state == business_state:
                    interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly driven by {yw_bus_state.replace('+', 'positive').replace('-', 'negative')} YW, IB normal"
                else:
                    interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly despite {yw_bus_state.replace('+', 'positive').replace('-', 'negative')} YW offsetting IB normal"
            elif ib_bus_state in ["+", "-"] and yw_bus_state in ["+", "-"]:
                if ib_bus_state == yw_bus_state == business_state:
                    interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly consistent across all children (IB, YW)"
                else:
                    # Complex mixed case - check which child contributes more to the parent anomaly
                    interpretations["Global/SH/Business"] = f"{business_state.replace('+', 'Positive').replace('-', 'Negative')} anomaly: mixed IB ({ib_bus_state}) and YW ({yw_bus_state}) effects"

    return interpretations

def build_ai_input_string(period: int, anomalies: dict, deviations: dict, 
                         interpretations: dict, explanations: dict, date_range: tuple, segment_filter: str = "Global", nps_values: dict = None,
                         comparison_context: str = None) -> str:
    """Build comprehensive input string for AI interpretation, filtered by segment"""
    
    # Normalize segment_filter to match the tree structure
    if segment_filter != "Global":
        normalized_segment = normalize_segment_to_root(segment_filter)
        print(f"🔍 DEBUG: segment_filter '{segment_filter}' normalized to '{normalized_segment}'", file=sys.stderr)
    else:
        normalized_segment = segment_filter
    
    # Use the original anomalies tree directly (period_anomalies that gets enriched)
    # No filtering needed - let the interpreter see the full tree with all explanations
    filtered_anomalies = anomalies  # Use original tree directly
    filtered_deviations = deviations  # Use original deviations directly
    filtered_nps_values = nps_values if nps_values else {}
    
    # DEBUG: Show what we're working with
    print(f"🔍 DEBUG build_ai_input_string:", file=sys.stderr)
    print(f"   segment_filter: {segment_filter}", file=sys.stderr)
    print(f"   using full tree (no filtering)", file=sys.stderr)
    print(f"   explanations keys: {list(explanations.keys())}", file=sys.stderr)
    print(f"   explanations content preview:", file=sys.stderr)
    for key, value in explanations.items():
        preview = value[:200] if value else "None"
        print(f"     {key}: {len(value) if value else 0} chars - {repr(preview)}", file=sys.stderr)
    print(f"   original anomalies: {list(anomalies.keys())}", file=sys.stderr)
    print(f"   filtered_anomalies: {list(filtered_anomalies.keys())}", file=sys.stderr)
    print(f"   original explanations: {list(explanations.keys())}", file=sys.stderr)
    print(f"   nps_values available: {list(filtered_nps_values.keys()) if filtered_nps_values else 'None'}", file=sys.stderr)
    if filtered_nps_values:
        print(f"   Sample NPS data: {list(filtered_nps_values.items())[:3]}", file=sys.stderr)
        # Add more detailed debug for NPS values
        for node_path, nps_data in list(filtered_nps_values.items())[:3]:
            print(f"   NPS data for {node_path}: {nps_data}", file=sys.stderr)
    
    if date_range:
        start_date, end_date = date_range
        ai_input = f"NPS ANOMALY ANALYSIS - PERIOD {period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})\n\n"
    else:
        ai_input = f"NPS ANOMALY ANALYSIS - PERIOD {period}\n\n"
    
    # Add comparison context (baseline reference) if available
    if comparison_context:
        ai_input += f"BASELINE REFERENCE:\n{comparison_context}\n\n"
    
    # Count actual anomalies vs normal variations
    actual_anomalies = [node for node, state in filtered_anomalies.items() if state in ['+', '-']]
    normal_segments = [node for node, state in filtered_anomalies.items() if state == 'N']
    
    ai_input += f"ANOMALY SUMMARY:\n"
    ai_input += f"• Total segments analyzed: {len(filtered_anomalies)}\n"
    ai_input += f"• Actual anomalies detected: {len(actual_anomalies)}\n"
    ai_input += f"• Normal variations: {len(normal_segments)}\n\n"
    
    if actual_anomalies:
        ai_input += f"SEGMENTS WITH ACTUAL ANOMALIES:\n"
        for node_path in actual_anomalies:
            state = filtered_anomalies[node_path]
            deviation = filtered_deviations.get(node_path, 0)
            anomaly_type = "POSITIVE" if state == "+" else "NEGATIVE"
            ai_input += f"• {node_path}: {anomaly_type} ANOMALY ({deviation:+.1f} points)\n"
        ai_input += "\n"
    
    ai_input += "DETAILED HIERARCHY:\n\n"
    
    # Helper function to get state description
    def get_state_desc(state, deviation):
        if state == "+":
            return f"POSITIVE ANOMALY ({deviation:+.1f} pts)"
        elif state == "-":
            return f"NEGATIVE ANOMALY ({deviation:+.1f} pts)"
        elif state == "N":
            if abs(deviation) < 10:
                return f"Normal ({deviation:+.1f} pts - within normal range)"
            else:
                return f"Normal ({deviation:+.1f} pts)"
        else:
            return f"State: {state}"
    
    # Build hierarchical structure with interpretations - filtered by segment
    def add_node_info(node_path: str, name: str, indent: str = ""):
        # Include all nodes (no filtering - show full tree)
        
        # Get state and deviation - default to "N" (Normal) for nodes not in anomalies
        state = filtered_anomalies.get(node_path, "N")
        deviation = filtered_deviations.get(node_path, 0)
        
        # Add NPS values if available
        nps_info = ""
        if filtered_nps_values and node_path in filtered_nps_values:
            nps_data = filtered_nps_values[node_path]
            if isinstance(nps_data, dict):
                current_nps = nps_data.get('current', 'N/A')
                baseline_nps = nps_data.get('baseline', 'N/A')
                nps_info = f" (NPS: {current_nps} vs baseline: {baseline_nps})"
            else:
                nps_info = f" (NPS: {nps_data})"
        
        ai_input_part = f"{indent}{name}: {get_state_desc(state, deviation)}{nps_info}\n"
        
        # Add interpretation if available
        if node_path in interpretations:
            ai_input_part += f"{indent}  └─ Pattern: {interpretations[node_path]}\n"
        
        # Add note for normal segments with NPS values
        elif state == "N" and filtered_nps_values and node_path in filtered_nps_values:
            nps_data = filtered_nps_values[node_path]
            if isinstance(nps_data, dict):
                current_nps = nps_data.get('current', 'N/A')
                baseline_nps = nps_data.get('baseline', 'N/A')
                ai_input_part += f"{indent}  └─ Note: No significant changes detected. Current period maintained stable performance.\n"
                ai_input_part += f"{indent}      • NPS Period: {current_nps:.1f} | NPS Baseline: {baseline_nps:.1f}\n"
        
        # Add note for normal segments WITHOUT NPS data
        elif state == "N" and (not filtered_nps_values or node_path not in filtered_nps_values):
            ai_input_part += f"{indent}  └─ Note: No significant changes detected. NPS data not available for this segment.\n"
        
        # Add explanation if available and is actual anomaly (or has explanation)
        if state in ['+', '-'] or (node_path in explanations and explanations[node_path] and explanations[node_path] != "Analysis timeout"):
            ai_input_part += f"{indent}  └─ ANALYSIS:\n"
            
            if node_path in explanations:
                explanation = explanations[node_path]
                
                if explanation and explanation != "Analysis timeout":
                    # Add the full explanation as-is with clear delimiters
                    ai_input_part += f"{indent}  └─ CAUSAL EXPLANATION:\n"
                    ai_input_part += f"{indent}     [{explanation}]\n"
                else:
                    # No explanation available
                    ai_input_part += f"{indent}     • No causal analysis available\n"
            else:
                ai_input_part += f"{indent}     • No analysis available\n"
        
        return ai_input_part
    
    # Build hierarchical structure based on normalized segment filter
    if normalized_segment == "Global":
        # Full tree
        ai_input += add_node_info("Global", "Global")
        ai_input += add_node_info("Global/LH", "Long Haul (LH)", "  ")
        ai_input += add_node_info("Global/LH/Economy", "├─ Economy", "    ")
        ai_input += add_node_info("Global/LH/Business", "├─ Business", "    ")
        ai_input += add_node_info("Global/LH/Premium", "└─ Premium", "    ")
        ai_input += add_node_info("Global/SH", "Short Haul (SH)", "  ")
        ai_input += add_node_info("Global/SH/Economy", "├─ Economy", "    ")
        ai_input += add_node_info("Global/SH/Economy/IB", "    └─ IB", "      ")
        ai_input += add_node_info("Global/SH/Economy/YW", "    └─ YW", "      ")
        ai_input += add_node_info("Global/SH/Business", "└─ Business", "    ")
        ai_input += add_node_info("Global/SH/Business/IB", "    └─ IB", "      ")
        ai_input += add_node_info("Global/SH/Business/YW", "    └─ YW", "      ")
    elif normalized_segment == "Global/LH":
        # LH tree only
        ai_input += add_node_info("Global/LH", "Long Haul (LH)")
        ai_input += add_node_info("Global/LH/Economy", "├─ Economy", "  ")
        ai_input += add_node_info("Global/LH/Business", "├─ Business", "  ")
        ai_input += add_node_info("Global/LH/Premium", "└─ Premium", "  ")
    elif normalized_segment == "Global/SH":
        # SH tree only
        ai_input += add_node_info("Global/SH", "Short Haul (SH)")
        ai_input += add_node_info("Global/SH/Economy", "├─ Economy", "  ")
        ai_input += add_node_info("Global/SH/Economy/IB", "  └─ IB", "    ")
        ai_input += add_node_info("Global/SH/Economy/YW", "  └─ YW", "    ")
        ai_input += add_node_info("Global/SH/Business", "└─ Business", "  ")
        ai_input += add_node_info("Global/SH/Business/IB", "  └─ IB", "    ")
        ai_input += add_node_info("Global/SH/Business/YW", "  └─ YW", "    ")
    elif normalized_segment == "Global/SH/Economy":
        # SH Economy tree only
        print("🔍 DEBUG: About to process Global/SH/Economy", file=sys.stderr)
        ai_input += add_node_info("Global/SH/Economy", "SH Economy")
        print("🔍 DEBUG: Processed Global/SH/Economy", file=sys.stderr)
        ai_input += add_node_info("Global/SH/Economy/IB", "├─ IB", "  ")
        print("🔍 DEBUG: Processed Global/SH/Economy/IB", file=sys.stderr)
        ai_input += add_node_info("Global/SH/Economy/YW", "└─ YW", "  ")
        print("🔍 DEBUG: Processed Global/SH/Economy/YW", file=sys.stderr)
    elif normalized_segment == "Global/SH/Business":
        # SH Business tree only
        ai_input += add_node_info("Global/SH/Business", "SH Business")
        ai_input += add_node_info("Global/SH/Business/IB", "├─ IB", "  ")
        ai_input += add_node_info("Global/SH/Business/YW", "└─ YW", "  ")
    elif normalized_segment == "Global/LH/Business":
        # LH Business node only
        ai_input += add_node_info("Global/LH/Business", "LH Business")
    elif normalized_segment == "Global/LH/Economy":
        # LH Economy node only
        ai_input += add_node_info("Global/LH/Economy", "LH Economy")
    elif normalized_segment == "Global/LH/Premium":
        # LH Premium node only
        ai_input += add_node_info("Global/LH/Premium", "LH Premium")
    elif normalized_segment == "Global/SH/Economy/IB":
        # SH Economy IB node only
        ai_input += add_node_info("Global/SH/Economy/IB", "SH Economy IB")
    elif normalized_segment == "Global/SH/Economy/YW":
        # SH Economy YW node only
        ai_input += add_node_info("Global/SH/Economy/YW", "SH Economy YW")
    elif normalized_segment == "Global/SH/Business/IB":
        # SH Business IB node only
        ai_input += add_node_info("Global/SH/Business/IB", "SH Business IB")
    elif normalized_segment == "Global/SH/Business/YW":
        # SH Business YW node only
        ai_input += add_node_info("Global/SH/Business/YW", "SH Business YW")
    else:
        # Single node - use the normalized segment
        node_name = normalized_segment.split('/')[-1] if '/' in normalized_segment else normalized_segment
        ai_input += add_node_info(normalized_segment, node_name)
    
    print("🔍 DEBUG: About to add instructions", file=sys.stderr)
    ai_input += "\nINTERPRETATION INSTRUCTIONS:\n"
    ai_input += "• Focus ONLY on segments marked as 'POSITIVE ANOMALY' or 'NEGATIVE ANOMALY'\n"
    ai_input += "• 'Normal' segments (even with deviations) are NOT anomalies - they are expected variations\n"
    ai_input += "• Explain the root causes using the analysis data provided for anomalous segments\n"
    if normalized_segment != "Global":
        ai_input += f"• Analysis scope limited to {segment_filter} segment and its children\n"
    else:
        ai_input += "• If Global shows 'Normal' despite segment anomalies, this means the anomalies are localized and balanced out\n"
    
    print("🔍 DEBUG: About to return ai_input", file=sys.stderr)
    print(f"🔍 DEBUG: ai_input length: {len(ai_input)} characters", file=sys.stderr)
    print(f"🔍 DEBUG: ai_input type: {type(ai_input)}", file=sys.stderr)
    # Check for problematic characters
    try:
        ai_input.encode('utf-8')
        print("🔍 DEBUG: ai_input encoding check passed", file=sys.stderr)
    except Exception as enc_e:
        print(f"🔍 DEBUG: ai_input encoding error: {enc_e}", file=sys.stderr)
    
    return ai_input

async def print_enhanced_tree_with_explanations_and_interpretations(
    anomalies: dict, deviations: dict, explanations: dict, interpretations: dict,
    aggregation_days: int, target_period: int, date_range=None, segment_filter: str = "Global",
    analysis_date: datetime = None, date_parameter: str = None, nps_values: dict = None):
    """Print enhanced tree with explanations and parent interpretations"""
    

    
    # Use the original trees directly (period_anomalies that gets enriched)
    # No filtering needed - show the full tree with all explanations
    filtered_anomalies = anomalies  # Use original tree directly
    filtered_deviations = deviations  # Use original deviations directly
    filtered_explanations = explanations  # Use all explanations
    filtered_interpretations = interpretations  # Use all interpretations
    filtered_nps_values = nps_values if nps_values else {}
    
    # Create a more descriptive title with date range
    if analysis_date and date_parameter:
        # Calculate correct date range based on analysis_date and date_parameter
        start_date, end_date = calculate_period_date_range(analysis_date, target_period, aggregation_days)
        period_title = f"Period {target_period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    elif date_range:
        start_date, end_date = date_range
        period_title = f"Period {target_period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    else:
        period_title = f"Period {target_period} ({aggregation_days}d aggregation)"
    
    print(f"\n📊 NPS Anomaly Analysis - {period_title}")
    print("-" * 70)
    
    # Normalize segment_filter for tree printing comparison
    normalized_segment = segment_filter
    if segment_filter == 'SH':
        normalized_segment = 'Global/SH'
    elif segment_filter == 'LH':
        normalized_segment = 'Global/LH'
    elif segment_filter == 'Economy' and '/' not in segment_filter:
        normalized_segment = 'Global/SH/Economy'
    elif segment_filter == 'Business' and '/' not in segment_filter:
        normalized_segment = 'Global/SH/Business'
    elif segment_filter == 'Business/LH':
        normalized_segment = 'Global/LH/Business'
    elif segment_filter == 'Economy/LH':
        normalized_segment = 'Global/LH/Economy'
    elif segment_filter == 'Premium/LH':
        normalized_segment = 'Global/LH/Premium'
    elif segment_filter == 'Business/SH':
        normalized_segment = 'Global/SH/Business'
    elif segment_filter == 'Economy/SH':
        normalized_segment = 'Global/SH/Economy'
    
    # DEBUG: Show what's in the filtered data
    debug_print(f"Filtered anomalies for segment {segment_filter} (normalized: {normalized_segment}):")
    debug_print(f"  Anomalies: {filtered_anomalies}")
    debug_print(f"  Deviations: {filtered_deviations}")
    debug_print(f"  Explanations available: {list(filtered_explanations.keys())}")
    debug_print(f"  Interpretations available: {list(filtered_interpretations.keys())}")
    
    def get_state_description(state):
        """Get clear state description"""
        if state == "+":
            return "POSITIVE ANOMALY"
        elif state == "-":
            return "NEGATIVE ANOMALY"
        elif state == "N":
            return "Normal"
        elif state == "S":
            return "Low Sample"
        else:
            return "No Data"
    
    def get_deviation_text(node_path):
        deviation = deviations.get(node_path, 0)
        deviation_text = f" ({deviation:+.1f} pts)" if deviation != 0 else ""
        
        # Add NPS values if available - show them more prominently
        nps_info = ""
        if nps_values and node_path in nps_values:
            nps_data = nps_values[node_path]
            if isinstance(nps_data, dict):
                current_nps = nps_data.get('current', 'N/A')
                baseline_nps = nps_data.get('baseline', 'N/A')
                if isinstance(current_nps, (int, float)) and isinstance(baseline_nps, (int, float)):
                    difference = current_nps - baseline_nps
                    nps_info = f" [NPS: {current_nps:.1f} vs {baseline_nps:.1f}, diff: {difference:+.1f}]"
                else:
                    nps_info = f" [NPS: {current_nps} vs {baseline_nps}]"
            else:
                nps_info = f" [NPS: {nps_data}]"
        
        return deviation_text + nps_info
    
    def print_interpretation(node_path, indent=""):
        if node_path in filtered_interpretations:
            print(f"{indent}  └─ Pattern: {filtered_interpretations[node_path]}")
    
    def print_explanation(node_path, indent=""):
        print(f"{indent}  └─ ANALYSIS:")
        
        if node_path in filtered_explanations:
            explanation = filtered_explanations[node_path]
            
            if explanation == "Analysis timeout":
                print(f"{indent}     • Analysis timeout occurred")
                return
            elif explanation and explanation.strip() != "":
                
                # Check if this is an agent explanation (contains AI Causal Investigation or Agent header)
                if "AI Causal Investigation:" in explanation:
                    # Agent mode: Show the narrative explanation directly
                    agent_content = explanation.replace("• AI Causal Investigation:", "").strip()
                    print(f"{indent}     • AI Causal Investigation: {agent_content}")
                    return
                elif "🤖 **AGENT CAUSAL ANALYSIS**" in explanation:
                    # Agent mode: Show the narrative explanation directly
                    agent_content = explanation.replace("🤖 **AGENT CAUSAL ANALYSIS**\n", "").strip()
                    print(f"{indent}     • AI Causal Investigation: {agent_content}")
                    return
                
                # Raw mode: Parse structured explanation with separators
                # Initialize with default values
                operational_content = "No operational data available"
                routes_content = "Not enough answers for statistical analysis"
                verbatims_content = "Not enough answers for statistical analysis"
                drivers_content = "Not enough answers for statistical analysis"
                
                # Split explanation into components and clean them up
                parts = explanation.split(" | ")
                
                for part in parts:
                    part = part.strip()
                    if not part or part.startswith("Period"):
                        continue
                    
                    # Clean up and format different explanation types (look for text patterns, not emojis)
                    if "Customer feedback:" in part or "verbatims collected" in part or "predominantly" in part:
                        if "predominantly negative" in part:
                            sentiment = "negative feedback"
                        elif "predominantly positive" in part:
                            sentiment = "positive feedback"
                        else:
                            sentiment = "mixed feedback"
                        
                        topics = ""
                        if "main topics:" in part:
                            topics_part = part.split("main topics:")[1].strip()
                            if topics_part and not topics_part.endswith("("):
                                topics = f", topics: {topics_part}"
                        
                        count = ""
                        if "verbatims collected" in part:
                            try:
                                count_part = part.split(" verbatims collected")[0]
                                count_num = count_part.split()[-1]
                                count = f"{count_num} verbatims, "
                            except:
                                pass
                        
                        # Remove emoji and clean the part
                        clean_part = part.replace("💬", "").replace("Customer feedback:", "").strip()
                        if clean_part:
                            verbatims_content = f"{count}{sentiment}{topics} - {clean_part}"
                        else:
                            verbatims_content = f"{count}{sentiment}{topics}"
                    
                    elif "Routes:" in part or "🛣️" in part:
                        clean_part = part.replace("🛣️ Routes:", "").replace("🛣️", "").replace("Routes:", "").strip()
                        if clean_part:
                            routes_content = clean_part
                    
                    elif "Operational:" in part:
                        clean_part = part.replace("🔧 Operational:", "").replace("🔧", "").replace("Operational:", "").strip()
                        if clean_part:
                            operational_content = clean_part
                    
                    elif "Drivers:" in part or "🚚" in part:
                        clean_part = part.replace("🚚 Drivers:", "").replace("🚚", "").replace("Drivers:", "").strip()
                        if clean_part:
                            drivers_content = clean_part
                
                # Show all four categories in consistent order for raw mode
                print(f"{indent}     • Operational: {operational_content}")
                print(f"{indent}     • Routes: {routes_content}")
                print(f"{indent}     • Verbatims: {verbatims_content}")
                print(f"{indent}     • Explanatory Drivers: {drivers_content}")
        else:
            # No explanation available
            print(f"{indent}     • No analysis available for this node")
    
    # Print tree based on normalized segment filter
    if normalized_segment == "Global":
        # Show full tree
        print_full_tree(filtered_anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation)
    elif normalized_segment == "Global/LH":
        # Show only LH tree
        print_lh_tree(filtered_anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation)
    elif normalized_segment == "Global/SH":
        # Show only SH tree
        print_sh_tree(filtered_anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation)
    elif normalized_segment == "Global/SH/Economy":
        # Show only SH Economy tree
        print_sh_economy_tree(filtered_anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation)
    elif normalized_segment == "Global/SH/Business":
        # Show only SH Business tree
        print_sh_business_tree(filtered_anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation)
    else:
        # Show single node for leaf segments
        print_single_node(normalized_segment, filtered_anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation)
    
    # Summary
    actual_anomalies = [node for node, state in filtered_anomalies.items() if state in ['+', '-']]
    print(f"\n📋 SUMMARY:")
    if actual_anomalies:
        print(f"  • Anomalies detected in: {', '.join(actual_anomalies)}")
        print(f"  • Total anomalous segments: {len(actual_anomalies)}")
    else:
        print(f"  • No anomalies detected - all segments within normal variation")

async def run_flexible_data_download_with_date(aggregation_days: int, periods: int, start_date, date_parameter: str, segment: str = "Global"):
    """Run flexible data download with custom date and parameter naming"""
    # Generate folder name with new naming convention: {parameter}_{date}_flexible_{aggregation_days}d
    date_str = start_date.strftime('%Y_%m_%d')
    timestamp = datetime.now().strftime('%H%M')  # Add time for uniqueness if needed
    
    # Include segment in folder name if not Global
    if segment != "Global":
        segment_suffix = f"_{segment.replace('/', '_')}"
    else:
        segment_suffix = ""
    
    target_folder = f"tables/{date_parameter}_{date_str}_flexible_{aggregation_days}d{segment_suffix}_{timestamp}"
    
    print(f"📁 Target folder: {target_folder}")
    print(f"📅 Analysis start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"🏷️ Date parameter: {date_parameter}")
    print(f"🎯 Segment focus: {segment}")
    
    success = await collect_flexible_data(aggregation_days, target_folder, segment, start_date)
    
    if success:
        print(f"✅ Data collection completed successfully")
        return target_folder
    else:
        print(f"❌ Data collection failed")
        return None

async def run_flexible_data_download_silent_with_date(aggregation_days: int, periods: int, start_date, date_parameter: str, segment: str = "Global", environment: str = "prod"):
    """Run flexible data download completely silently with custom date and parameter naming"""
    # Generate folder name with new naming convention
    date_str = start_date.strftime('%Y_%m_%d')
    timestamp = datetime.now().strftime('%H%M')  
    
    # Include segment in folder name if not Global
    if segment != "Global":
        segment_suffix = f"_{segment.replace('/', '_')}"
    else:
        segment_suffix = ""
    
    target_folder = f"tables/{date_parameter}_{date_str}_flexible_{aggregation_days}d{segment_suffix}_{timestamp}"
    
    collector = PBIDataCollector(environment=environment)
    
    # Get node paths for the specified segment
    node_paths = get_segment_node_paths(segment)
    
    # Collect data for selected nodes completely silently
    total_success = 0
    total_attempted = 0
    
    # Suppress all output during data collection
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for node_path in node_paths:
                try:
                    results = await collector.collect_flexible_data_for_node(
                        node_path, aggregation_days, target_folder, start_date
                    )
                    total_attempted += len(results)
                    total_success += sum(results.values())
                except Exception:
                    pass
    
    if total_success > 0:
        return target_folder
    else:
        return None

def calculate_baseline_period_for_causal_filter(current_period: int, causal_filter: str, aggregation_days: int = 7) -> tuple[int, str]:
    """
    Calculate the baseline period number and description based on the current period and causal filter.
    
    Args:
        current_period: The period being analyzed (e.g., 1, 2, 3...)
        causal_filter: The causal filter comparison (e.g., "vs L7d", "vs LM", "vs LY", "vs Target")
        aggregation_days: Number of days per period (default: 7 for weekly)
        
    Returns:
        tuple: (baseline_period_number, baseline_description)
    """
    if causal_filter == "vs L7d":
        # Compare with last 7 days (1 period back for weekly aggregation)
        baseline_period = current_period + 1
        return baseline_period, "últimos 7 días"
    
    elif causal_filter == "vs L14d":
        # Compare with last 14 days (2 periods back for weekly aggregation)
        baseline_period = current_period + 2
        return baseline_period, "últimos 14 días"
    
    elif causal_filter == "vs LM":
        # Compare with last month (approximately 4 periods back for weekly aggregation)
        baseline_period = current_period + 4
        return baseline_period, "mes natural anterior"
    
    elif causal_filter == "vs LY":
        # Compare with last year (approximately 52 periods back for weekly aggregation)
        baseline_period = current_period + 52
        return baseline_period, "mismo período año anterior"
    
    elif causal_filter == "vs Target":
        # Target comparison - this will be handled differently in the detector
        return None, "objetivo/target"
    
    elif causal_filter == "vs Sel. Period":
        # Selected period comparison - requires specific dates from user
        return None, "período seleccionado"
    
    else:
        # Default fallback
        baseline_period = current_period + 1
        return baseline_period, "período inmediatamente anterior"


def determine_anomaly_mode_for_vslast(causal_filter: str, comparison_start_date: str = None, comparison_end_date: str = None) -> tuple[str, str]:
    """
    Determine the anomaly detection mode and baseline description for vslast mode
    based on the causal filter comparison.
    
    Args:
        causal_filter: The causal filter comparison (e.g., "vs L7d", "vs LM", "vs LY", "vs Target")
        comparison_start_date: Start date for comparison period (YYYY-MM-DD) when using "vs Sel. Period"
        comparison_end_date: End date for comparison period (YYYY-MM-DD) when using "vs Sel. Period"
        
    Returns:
        tuple: (anomaly_mode, baseline_description)
    """
    if causal_filter == "vs L7d":
        return "vslast_dynamic", "últimos 7 días"
    elif causal_filter == "vs L14d":
        return "vslast_dynamic", "últimos 14 días"
    elif causal_filter == "vs LM":
        return "vslast_dynamic", "mes natural anterior"
    elif causal_filter == "vs LY":
        return "vslast_dynamic", "mismo período año anterior"
    elif causal_filter == "vs Target":
        return "vslast_dynamic", "objetivo/target"
    elif causal_filter == "vs Sel. Period":
        if comparison_start_date and comparison_end_date:
            return "vslast_dynamic", f"período seleccionado ({comparison_start_date} a {comparison_end_date})"
        else:
            return "vslast_dynamic", "período seleccionado"
    else:
        return "vslast_dynamic", "período inmediatamente anterior"

def normalize_segment_to_root(segment: str) -> str:
    """
    Normalize segment parameter to its root node path.
    
    Args:
        segment: Segment parameter (e.g., 'Economy/SH', 'Business/LH', 'Global')
        
    Returns:
        Root node path (e.g., 'Global/SH/Economy', 'Global/LH/Business', 'Global')
    """
    # Handle shortcuts and normalize to full paths
    if segment == 'SH':
        return 'Global/SH'
    elif segment == 'LH':
        return 'Global/LH'
    elif segment == 'Economy/LH':
        return 'Global/LH/Economy'
    elif segment == 'Business/LH':
        return 'Global/LH/Business'
    elif segment == 'Premium/LH':
        return 'Global/LH/Premium'
    elif segment == 'Economy/SH':
        return 'Global/SH/Economy'
    elif segment == 'Business/SH':
        return 'Global/SH/Business'
    elif segment == 'Economy/SH/IB':
        return 'Global/SH/Economy/IB'
    elif segment == 'Economy/SH/YW':
        return 'Global/SH/Economy/YW'
    elif segment == 'Business/SH/IB':
        return 'Global/SH/Business/IB'
    elif segment == 'Business/SH/YW':
        return 'Global/SH/Business/YW'
    elif segment == 'Economy' and '/' not in segment:
        # Ambiguous - default to SH/Economy
        return 'Global/SH/Economy'
    elif segment == 'Business' and '/' not in segment:
        # Ambiguous - default to SH/Business  
        return 'Global/SH/Business'
    elif segment == 'Global' or segment.startswith('Global/'):
        # Already normalized or is Global
        return segment
    else:
        # Default to Global for unknown segments
        return 'Global'

def get_segment_node_paths(segment: str) -> list:
    """
    Generate the list of node paths based on the selected segment.
    
    Args:
        segment: Root segment to analyze (e.g., 'Global', 'SH', 'Global/SH/Economy')
        
    Returns:
        List of node paths to collect and analyze
    """
    # Normalize segment input - handle shortcuts
    if segment == 'SH':
        segment = 'Global/SH'
    elif segment == 'LH':
        segment = 'Global/LH'
    elif segment == 'Economy/LH':
        segment = 'Global/LH/Economy'
    elif segment == 'Business/LH':
        segment = 'Global/LH/Business'
    elif segment == 'Premium/LH':
        segment = 'Global/LH/Premium'
    elif segment == 'Economy/SH':
        segment = 'Global/SH/Economy'
    elif segment == 'Business/SH':
        segment = 'Global/SH/Business'
    elif segment == 'Economy' and '/' not in segment:
        # Ambiguous - default to SH/Economy
        segment = 'Global/SH/Economy'
    elif segment == 'Business' and '/' not in segment:
        # Ambiguous - default to SH/Business  
        segment = 'Global/SH/Business'
    
    # Define the complete hierarchy
    all_nodes = {
        "Global": [
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
        ],
        "Global/LH": [
            "Global/LH",
            "Global/LH/Economy", 
            "Global/LH/Business", 
            "Global/LH/Premium"
        ],
        "Global/SH": [
            "Global/SH",
            "Global/SH/Economy", 
            "Global/SH/Business",
            "Global/SH/Economy/IB", 
            "Global/SH/Economy/YW",
            "Global/SH/Business/IB", 
            "Global/SH/Business/YW"
        ],
        "Global/LH/Economy": [
            "Global/LH/Economy"
        ],
        "Global/LH/Business": [
            "Global/LH/Business"
        ],
        "Global/LH/Premium": [
            "Global/LH/Premium"
        ],
        "Global/SH/Economy": [
            "Global/SH/Economy",
            "Global/SH/Economy/IB", 
            "Global/SH/Economy/YW"
        ],
        "Global/SH/Business": [
            "Global/SH/Business",
            "Global/SH/Business/IB", 
            "Global/SH/Business/YW"
        ],
        "Global/SH/Economy/IB": [
            "Global/SH/Economy/IB"
        ],
        "Global/SH/Economy/YW": [
            "Global/SH/Economy/YW"
        ],
        "Global/SH/Business/IB": [
            "Global/SH/Business/IB"
        ],
        "Global/SH/Business/YW": [
            "Global/SH/Business/YW"
        ]
    }
    
    if segment in all_nodes:
        return all_nodes[segment]
    else:
        # If segment not found, try to find partial matches
        matching_segments = [key for key in all_nodes.keys() if segment in key]
        if matching_segments:
            # Use the first match
            return all_nodes[matching_segments[0]]
        else:
            # Fallback to Global if no match found
            print(f"⚠️ Segment '{segment}' not found. Using Global as fallback.")
            return all_nodes["Global"]

async def run_flexible_analysis_silent(data_folder: str, analysis_date: datetime = None, date_parameter: str = None, anomaly_detection_mode: str = "target", baseline_periods: int = 7, causal_filter: str = "vs L7d", periods: int = 7, causal_comparison_dates: tuple = None, segment: str = "Global", environment: str = "prod"):
    """Run flexible analysis completely silently"""
    import os
    from contextlib import redirect_stdout, redirect_stderr
    
    # Extract aggregation days from folder name
    folder_name = Path(data_folder).name
    if 'flexible_' in folder_name and 'd_' in folder_name:
        try:
            aggregation_days = int(folder_name.split('flexible_')[1].split('d_')[0])
        except:
            aggregation_days = 7  # Default
    else:
        aggregation_days = 7
    
    # Configure anomaly detection based on mode
    detector = FlexibleAnomalyDetector(
        aggregation_days=aggregation_days,
        threshold=5.0,
        min_sample_size=5,
        detection_mode=("vslast_dynamic" if anomaly_detection_mode == "vslast" and locals().get('causal_filter') == "vs Sel. Period" else anomaly_detection_mode),
        baseline_periods=baseline_periods,
        causal_filter=causal_filter,
        causal_comparison_dates=causal_comparison_dates,
        environment=environment
    )
    
    # Calculate the correct period numbers based on date parameter type and periods parameter
    reference_period = None  # Will be used for baseline calculation when date_flight_local is specified
    if analysis_date and date_parameter:
        if any(param in date_parameter for param in ['flight_local', 'available']):
            # For date_flight_local or default available: treat analysis_date as the most recent period (period 1)
            # Analyze the specified number of most recent periods relative to that date
            periods_to_analyze = list(range(1, periods + 1))  # Periods 1, 2, 3, ... up to specified count
            if 'flight_local' in date_parameter:
                # When date_flight_local is specified, use period 1 as reference for baseline calculation
                reference_period = 1
        elif date_parameter == 'insert_ci':
            # For insert_date_ci: calculate actual period numbers relative to today
            base_period = calculate_actual_period_number(analysis_date)
            # Analyze specified number of periods starting from the analysis date
            periods_to_analyze = list(range(base_period, base_period + periods))
        else:
            # Unknown parameter type, default to specified periods
            periods_to_analyze = list(range(1, periods + 1))
    else:
        # Default behavior: analyze the specified number of most recent periods
        periods_to_analyze = list(range(1, periods + 1))
    
    anomaly_periods = []
    
    # Suppress all output during analysis
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                for period in periods_to_analyze:
                    period_anomalies, period_deviations, period_explanations, period_nps_values = await detector.analyze_period(data_folder, period, analysis_date, reference_period)
                    
                    # Check if any node has an anomaly
                    has_anomaly = any(state in ['+', '-'] for state in period_anomalies.values())
                    
                    # Also check if root segment has valid data (even if no anomalies)
                    root_segment = normalize_segment_to_root(segment)
                    root_has_data = root_segment in period_anomalies and period_anomalies[root_segment] != "?"
                    
                    # Include period if there are anomalies OR if root segment has valid data
                    if has_anomaly or root_has_data:
                        anomaly_periods.append(period)
            
            except Exception as e:
                # Exception occurred - will handle after exiting silent block
                exception_occurred = e
                exception_traceback = __import__('traceback').format_exc()
    
    # Check if exception occurred (must be after silent block)
    if 'exception_occurred' in locals():
        print(f"❌ Exception in analysis loop: {exception_occurred}")
        print(exception_traceback)
        return None
    
    return {
        'detector': detector,
        'data_folder': data_folder,
        'aggregation_days': aggregation_days,
        'anomaly_periods': anomaly_periods,
        'total_periods': periods,
        'periods_analyzed': periods_to_analyze,
        'analysis_date': analysis_date,
        'date_parameter': date_parameter,
        'anomaly_detection_mode': anomaly_detection_mode,
        'baseline_periods': baseline_periods
    }

async def show_silent_anomaly_analysis(analysis_data: dict, analysis_type: str, show_all_periods=False, segment: str = "Global", causal_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None, environment: str = "prod"):
    """Show only trees and AI summaries for periods with anomalies - silent version"""
    import os
    from contextlib import redirect_stdout, redirect_stderr
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    aggregation_days = analysis_data['aggregation_days']
    anomaly_periods = analysis_data['anomaly_periods']
    periods_analyzed = analysis_data.get('periods_analyzed', anomaly_periods)
    
    # Initialize interpreter for explanations with specified mode
    pbi_collector = PBIDataCollector(environment=environment)
    
    # Determine study_mode based on analysis_type and causal_filter
    if analysis_type == "WEEKLY_COMPARATIVE" or causal_filter == "vs Sel. Period":
        study_mode = "comparative"
    elif analysis_type == "DIARIO_SINGLE" or causal_filter is None:
        study_mode = "single"
    else:
        study_mode = "comparative"  # Default to comparative
    
    print(f"🔍 DEBUG MAIN: analysis_data['detector'].detection_mode = '{analysis_data['detector'].detection_mode}'")
    print(f"🔍 DEBUG MAIN: About to create FlexibleAnomalyInterpreter with detection_mode='{analysis_data['detector'].detection_mode}'")
    interpreter = FlexibleAnomalyInterpreter(
        data_folder, 
        pbi_collector=pbi_collector, 
        silent_mode=True, 
        detection_mode=analysis_data['detector'].detection_mode, 
        causal_filter=causal_filter, 
        comparison_start_date=comparison_start_date, 
        comparison_end_date=comparison_end_date,
        study_mode=study_mode,
        environment=environment
    )    
    # Initialize AI agent for interpretation
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType, get_default_llm_type
        
        # Determine study_mode based on analysis_type and causal_filter (same logic as interpreter)
        if analysis_type == "WEEKLY_COMPARATIVE" or causal_filter == "vs Sel. Period":
            ai_study_mode = "comparative"
        elif analysis_type == "DIARIO_SINGLE" or causal_filter is None:
            ai_study_mode = "single"
        else:
            ai_study_mode = "comparative"  # Default to comparative
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=get_default_llm_type(),
            config_path="dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
            logger=logging.getLogger("ai_interpreter"),
            study_mode=ai_study_mode,
            environment=environment
        )
        ai_available = True
    except Exception:
        ai_available = False
    
    # Collect data for summary
    all_periods_data = []
    
    # If show_all_periods is True, show all periods analyzed, not just those with anomalies
    if show_all_periods:
        periods_to_show = periods_analyzed
    else:
        periods_to_show = [p for p in periods_analyzed if p in anomaly_periods]
    
    if not periods_to_show:
        print(f"✅ No periods to analyze in {analysis_type.lower()} analysis")
        return []
    
    # Show detailed analysis for selected periods
    for period in periods_to_show:
        print(f"\n{'='*60}")
        print(f"{analysis_type} PERIOD {period} ANALYSIS")
        print("="*60)
        
        # Get anomalies for this period
        analysis_date = analysis_data.get('analysis_date')
        date_parameter = analysis_data.get('date_parameter')
        
                # Calculate reference_period for baseline (same logic as in run_flexible_analysis_silent)
        reference_period = None
        if analysis_date and date_parameter:
            if any(param in date_parameter for param in ['flight_local', 'available']):
                if 'flight_local' in date_parameter:
                    reference_period = 1
        
        print(f"🔍 DEBUG DAILY_ANALYSIS: About to call detector.analyze_period for period {period} with reference_period={reference_period}", file=sys.stderr)
        analyze_result = await detector.analyze_period(data_folder, period, analysis_date, reference_period)
        print(f"🔍 DEBUG DAILY_ANALYSIS: Result type: {type(analyze_result)}", file=sys.stderr)
        print(f"🔍 DEBUG DAILY_ANALYSIS: Result length: {len(analyze_result) if hasattr(analyze_result, '__len__') else 'No length'}", file=sys.stderr)
        
        period_anomalies, period_deviations, _, period_nps_values = analyze_result
        print(f"🔍 DEBUG DAILY_ANALYSIS: period_nps_values type: {type(period_nps_values)}", file=sys.stderr)
        print(f"🔍 DEBUG DAILY_ANALYSIS: period_nps_values content: {period_nps_values}", file=sys.stderr)
        
        # Get date range using the correct method based on date_parameter
        date_parameter = analysis_data.get('date_parameter')
        
        if analysis_date and date_parameter in ['flight_local', 'available']:
            # For date_flight_local or default available: calculate relative to analysis_date
            start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
            date_range = (start_date, end_date)
        else:
            # For other parameters or fallback: use interpreter method
            date_range = interpreter._get_period_date_range(period, aggregation_days)
        
        if date_range:
            start_date, end_date = date_range
            print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            date_range_str = "Unknown dates"
        
        # Generate parent interpretations
        parent_interpretations = generate_parent_interpretations(period_anomalies)
        
        # Collect explanations for anomalous nodes silently
        explanations = {}
        nodes_with_anomalies = [node for node, state in period_anomalies.items() if state in ['+', '-']]

        # ENHANCEMENT: Always include root segment (--segment parameter) in causal analysis
        # Get the root segment for the analysis
        root_segment = normalize_segment_to_root(segment)
        print(f"🔍 DEBUG ROOT SEGMENT: segment='{segment}' -> root_segment='{root_segment}'", file=sys.stderr)
        
        # Check if root segment has valid data (not "?" or missing)
        root_state = period_anomalies.get(root_segment, "?")
        print(f"🔍 DEBUG ROOT STATE: root_state='{root_state}', period_anomalies keys: {list(period_anomalies.keys())}", file=sys.stderr)
        
        # Always recalculate root segment's anomaly state based on deviation (override detector's decision)
        root_deviation = period_deviations.get(root_segment, 0.0)
        print(f"🔍 DEBUG ROOT CALCULATION: root_deviation={root_deviation}", file=sys.stderr)
        if root_deviation > 0:
            root_state = "+"  # Positive anomaly
        elif root_deviation < 0:
            root_state = "-"  # Negative anomaly
        else:
            root_state = "N"  # Neutral (only when deviation is exactly 0)
        # Always update root segment in period_anomalies with calculated state
        period_anomalies[root_segment] = root_state
        print(f"      🔍 DEBUG ROOT: Override {root_segment} state='{root_state}' based on deviation={root_deviation}", file=sys.stderr)
        
        # Always add root segment to the analysis, even if no other anomalies exist
        if root_segment not in nodes_with_anomalies and root_state != "?":
            nodes_with_anomalies.append(root_segment)
            if len(nodes_with_anomalies) == 1:
                print(f"      🔍 ENHANCED: Analyzing root segment {root_segment} (state '{root_state}', no child anomalies)", file=sys.stderr)
            else:
                print(f"      🔍 ENHANCED: Analyzing {len(nodes_with_anomalies)} segments (added {root_segment} with state '{root_state}' + {len([n for n in nodes_with_anomalies if n != root_segment])} anomalous nodes)", file=sys.stderr)
        elif root_segment in nodes_with_anomalies:
            print(f"      🔍 📊 Analyzing {len(nodes_with_anomalies)} anomalous segments (including {root_segment})", file=sys.stderr)
        else:
            print(f"      🔍 📊 Analyzing {len(nodes_with_anomalies)} anomalous segments ({root_segment} has no valid data)", file=sys.stderr)
        
        # PRIORITY: Move root segment to the front to ensure it's always processed first
        if root_segment in nodes_with_anomalies:
            nodes_with_anomalies.remove(root_segment)
            nodes_with_anomalies.insert(0, root_segment)
            print(f"      🎯 PRIORITY: {root_segment} moved to front of processing queue", file=sys.stderr)

        if nodes_with_anomalies:
            # DEBUG: Temporarily NOT suppressing output to see what explanations are being collected
            print(f"🔍 DEBUG: Collecting explanations for {len(nodes_with_anomalies)} anomalous nodes: {nodes_with_anomalies}", file=sys.stderr)
            # with open(os.devnull, 'w') as devnull:
            #     with redirect_stdout(devnull), redirect_stderr(devnull):
            if True:  # Changed from suppressed context
                    for node_path in nodes_with_anomalies:
                        try:
                            anomaly_state = period_anomalies.get(node_path, "?")
                            
                            # For Global node, determine anomaly state based on sign of change (using same nomenclature as other nodes)
                            if node_path == "Global" and anomaly_state == "?":
                                # Get deviation value to determine type
                                deviation_value = period_deviations.get(node_path, 0.0)
                                if deviation_value > 0:
                                    anomaly_state = "+"  # Positive anomaly
                                elif deviation_value < 0:
                                    anomaly_state = "-"  # Negative anomaly
                                else:
                                    anomaly_state = "N"  # Neutral
                            
                            # Calculate correct date range if analysis_date is available
                            start_date, end_date = None, None
                            analysis_date = analysis_data.get('analysis_date')
                            if analysis_date:
                                print(f"🔍 DEBUG SINGLE_MODE_DATES: analysis_date={analysis_date}, period={period}, aggregation_days={aggregation_days}")
                                start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
                                print(f"🔍 DEBUG SINGLE_MODE_DATES: calculated start_date={start_date}, end_date={end_date}")
                            
                            # Build NPS context for the causal agent and calculate anomaly magnitude
                            nps_context = ""
                            anomaly_magnitude = 0.0
                            print(f"🔍 DEBUG SILENT_ANOMALY NPS BUILD: period_nps_values keys: {list(period_nps_values.keys()) if period_nps_values else 'None'}", file=sys.stderr)
                            print(f"🔍 DEBUG SILENT_ANOMALY NPS BUILD: Looking for node_path: {node_path}", file=sys.stderr)
                            if period_nps_values and node_path in period_nps_values:
                                nps_data = period_nps_values[node_path]
                                print(f"🔍 DEBUG SILENT_ANOMALY Causal agent NPS for {node_path}: {nps_data}", file=sys.stderr)
                                if isinstance(nps_data, dict):
                                    current_nps = nps_data.get('current', 'N/A')
                                    baseline_nps = nps_data.get('baseline', 'N/A')
                                    nps_context = f"Current NPS: {current_nps}, Baseline NPS: {baseline_nps}"
                                    
                                    # Generate comparison context
                                    baseline_description = nps_data.get('baseline_description', None)
                                    comparison_context = generate_comparison_context(
                                        analysis_data.get('anomaly_detection_mode', 'target'), 
                                        analysis_data.get('aggregation_days', 7),
                                        analysis_data.get('baseline_periods', 7),
                                        baseline_description
                                    )
                                    
                                    # Calculate anomaly magnitude from NPS values
                                    if isinstance(current_nps, (int, float)) and isinstance(baseline_nps, (int, float)):
                                        anomaly_magnitude = current_nps - baseline_nps
                                else:
                                    nps_context = f"NPS: {nps_data}"
                            else:
                                print(f"🔍 DEBUG SILENT_ANOMALY Causal agent NO NPS for {node_path}", file=sys.stderr)
                            
                            explanation = await asyncio.wait_for(
                                interpreter.explain_anomaly(
                                    node_path=node_path,
                                    target_period=period,
                                    aggregation_days=aggregation_days,
                                    anomaly_state=anomaly_state,
                                    anomaly_magnitude=anomaly_magnitude,  # ✅ Now passing anomaly magnitude
                                    start_date=start_date,
                                    end_date=end_date,
                                    nps_context=nps_context,  # ✅ Now passing NPS context
                                    causal_filter=causal_filter,
                                    comparison_start_date=comparison_start_date,
                                    comparison_end_date=comparison_end_date,
                                    # New parameters for enriched context
                                    anomaly_detection_mode=analysis_data.get('anomaly_detection_mode', 'target'),
                                    comparison_context=comparison_context,
                                    baseline_periods=analysis_data.get('baseline_periods', 7)
                                ),
                                timeout=1500.0  # 25 minutes for complex Claude Sonnet 4 analysis
                            )
                            explanations[node_path] = explanation
                            print(f"🔍 EXPLANATION COLLECTED for {node_path}: {len(explanation) if explanation else 0} chars", file=sys.stderr)
                            if explanation:
                                print(f"   Preview: {explanation[:300]}...", file=sys.stderr)
                        except Exception:
                            explanations[node_path] = "Analysis timeout"
        
        # Show the tree
        analysis_date = analysis_data.get('analysis_date')
        date_parameter = analysis_data.get('date_parameter')
        await print_enhanced_tree_with_explanations_and_interpretations(
            period_anomalies, period_deviations, explanations, parent_interpretations,
            aggregation_days, period, date_range, segment, analysis_date, date_parameter
        )
        
        # AI Interpretation
        ai_interpretation = None
        if ai_available:
            print(f"\n🤖 AI INTERPRETATION:")
            print("-" * 40)
            
            try:
                # Generate comparison context for interpreter
                interpreter_comparison_context = generate_comparison_context(
                    analysis_data.get('anomaly_detection_mode', 'mean'),
                    analysis_data.get('aggregation_days', 1),
                    analysis_data.get('baseline_periods', 7)
                )
                
                # Always use the complete tree format with integrated causal explanations
                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                                 parent_interpretations, explanations, date_range, segment, period_nps_values,
                                                 comparison_context=interpreter_comparison_context)
                
                debug_print(f"AI input string length: {len(ai_input)} characters")
                debug_print(f"AI input preview: {ai_input[:500]}...")
                print(f"🔍 Using complete tree format with integrated explanations: {len(ai_input)} characters")
                
                # Fix: Extract start_date from date_range if available
                date_param = None
                if date_range and len(date_range) >= 2:
                    range_start_date, _ = date_range
                    if range_start_date:
                        date_param = range_start_date.strftime('%Y-%m-%d')
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, date_param, segment),
                    timeout=600.0
                )
                
                print("🎯 IMPRIMIENDO INTERPRETACIÓN FINAL:")
                print(ai_interpretation)
                
            except Exception as e:
                ai_interpretation = f"AI interpretation failed: {str(e)}"
                print("🎯 IMPRIMIENDO INTERPRETACIÓN FINAL:")
                print(ai_interpretation)
        
        # Collect period data for summary
        period_data = {
            'period': period,
            'date_range': date_range_str,
            'ai_interpretation': ai_interpretation or "No AI interpretation available"
        }
        all_periods_data.append(period_data)
    
    # Return the list of period interpretations (no summary agent here)
    # The caller (e.g., weekly_deep_research.py) will consolidate if needed
    return all_periods_data

async def show_clean_anomaly_analysis(analysis_data: dict, segment: str = "Global", causal_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None, environment: str = "prod", study_mode: str = None):
    """Show clean, focused analysis: tree + agent workflow + summary"""
    from contextlib import redirect_stdout, redirect_stderr
    import os
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    aggregation_days = analysis_data['aggregation_days']
    anomaly_periods = analysis_data['anomaly_periods']
    periods_analyzed = analysis_data.get('periods_analyzed', anomaly_periods)
    
    # Determine study_mode if not provided
    if study_mode is None:
        if causal_filter is None or causal_filter == 'None':
            study_mode = "single"
        else:
            study_mode = "comparative"
    
    # Initialize interpreter for explanations
    pbi_collector = PBIDataCollector(environment=environment)
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector, silent_mode=True, causal_filter=causal_filter, environment=environment)
    
    # Initialize AI agent for interpretation
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType, get_default_llm_type
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=get_default_llm_type(),
            config_path="dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
            logger=logging.getLogger("ai_interpreter"),
            study_mode=study_mode,
            environment=environment
        )
        ai_available = True
    except Exception:
        ai_available = False
    
    # Show only periods with anomalies
    periods_with_anomalies = [p for p in periods_analyzed if p in anomaly_periods]
    
    all_periods_data = []
    
    for period in periods_with_anomalies:
        print(f"\n{'='*60}")
        print(f"📊 PERIOD {period} ANALYSIS")
        print("="*60)
        
        # Get anomalies for this period
        analysis_date = analysis_data.get('analysis_date')
        
        # Ensure detector has the comparison parameters for vslast_dynamic mode
        if hasattr(detector, 'causal_comparison_dates') and detector.causal_comparison_dates:
            # Detector already has comparison dates configured
            print(f"🔍 DEBUG: Using existing comparison dates: {detector.causal_comparison_dates}")
            period_anomalies, period_deviations, _, _ = await detector.analyze_period(data_folder, period, analysis_date)
        elif comparison_start_date and comparison_end_date:
            # Configure detector with comparison dates before calling analyze_period
            # Convert datetime objects to strings if needed
            if hasattr(comparison_start_date, 'strftime'):
                start_str = comparison_start_date.strftime('%Y-%m-%d')
                end_str = comparison_end_date.strftime('%Y-%m-%d')
            else:
                start_str = str(comparison_start_date)
                end_str = str(comparison_end_date)
            
            detector.causal_comparison_dates = (start_str, end_str)
            print(f"🔍 DEBUG: Configured comparison dates: {detector.causal_comparison_dates}")
            period_anomalies, period_deviations, _, _ = await detector.analyze_period(data_folder, period, analysis_date)
        else:
            # Fallback to original call
            print(f"🔍 DEBUG: Using fallback call (no comparison dates)")
            period_anomalies, period_deviations, _, _ = await detector.analyze_period(data_folder, period, analysis_date)
        
        # DEBUG: Log what the detector returned
        print(f"🔍 DEBUG DETECTOR RESULT: period_anomalies type={type(period_anomalies)}, keys={list(period_anomalies.keys()) if isinstance(period_anomalies, dict) else 'Not a dict'}")
        print(f"🔍 DEBUG DETECTOR RESULT: period_deviations type={type(period_deviations)}, keys={list(period_deviations.keys()) if isinstance(period_deviations, dict) else 'Not a dict'}")
        if isinstance(period_anomalies, dict) and period_anomalies:
            print(f"🔍 DEBUG DETECTOR RESULT: period_anomalies content={period_anomalies}")
        if isinstance(period_deviations, dict) and period_deviations:
            print(f"🔍 DEBUG DETECTOR RESULT: period_deviations content={period_deviations}")
        
        # Calculate date range
        date_parameter = analysis_data.get('date_parameter')
        if analysis_date and date_parameter in ['flight_local', 'available']:
            start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
            date_range = (start_date, end_date)
        else:
            date_range = interpreter._get_period_date_range(period, aggregation_days)
        
        if date_range:
            start_date, end_date = date_range
            print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            date_range_str = "Unknown dates"
        
        # Generate parent interpretations
        parent_interpretations = generate_parent_interpretations(period_anomalies)
        
        # Show the clean tree first
        print(f"\n🌳 ANOMALY TREE:")
        print("-" * 40)
        await print_clean_tree_only(period_anomalies, period_deviations, parent_interpretations, segment, None)
        
        # Collect explanations and show agent workflow SILENTLY for anomalous nodes
        explanations = {}
        workflow_decisions = {}
        nodes_with_anomalies = [node for node, state in period_anomalies.items() if state in ['+', '-']]

        # ENHANCEMENT: Always include root segment in causal analysis if it has valid data
        if nodes_with_anomalies:
            # Get the root segment for the analysis
            root_segment = normalize_segment_to_root(segment)
            print(f"🔍 DEBUG ROOT SEGMENT: segment='{segment}' -> root_segment='{root_segment}'")
            
            # Check if root segment has valid data (not "?" or missing)
            root_state = period_anomalies.get(root_segment, "?")
            print(f"🔍 DEBUG ROOT STATE: root_state='{root_state}', period_anomalies keys: {list(period_anomalies.keys())}")
            
            # Always recalculate root segment's anomaly state based on deviation (override detector's decision)
            root_deviation = period_deviations.get(root_segment, 0.0)
            print(f"🔍 DEBUG ROOT CALCULATION: root_deviation={root_deviation}")
            if root_deviation > 0:
                root_state = "+"  # Positive anomaly
            elif root_deviation < 0:
                root_state = "-"  # Negative anomaly
            else:
                root_state = "N"  # Neutral (only when deviation is exactly 0)
            # Always update root segment in period_anomalies with calculated state
            period_anomalies[root_segment] = root_state
            print(f"      🔍 DEBUG ROOT: Override {root_segment} state='{root_state}' based on deviation={root_deviation}")
            
            if root_segment not in nodes_with_anomalies and root_state != "?":
                # Add root segment to the analysis even if it doesn't have anomalies but has valid data
                nodes_with_anomalies.append(root_segment)
                print(f"🔍 ENHANCED: Analyzing {len(nodes_with_anomalies)} segments (added {root_segment} with state '{root_state}' + {len([n for n in nodes_with_anomalies if n != root_segment])} anomalous nodes)")
            elif root_segment in nodes_with_anomalies:
                print(f"📊 Analyzing {len(nodes_with_anomalies)} anomalous segments (including {root_segment})")
            else:
                print(f"📊 Analyzing {len(nodes_with_anomalies)} anomalous segments ({root_segment} has no valid data)")
            
            # PRIORITY: Move root segment to the front to ensure it's always processed first
            if root_segment in nodes_with_anomalies:
                nodes_with_anomalies.remove(root_segment)
                nodes_with_anomalies.insert(0, root_segment)
                print(f"🎯 PRIORITY: {root_segment} moved to front of processing queue")

        if nodes_with_anomalies:
            print(f"\n🤖 AGENT WORKFLOW ANALYSIS:")
            print("-" * 40)
            
            for node_path in nodes_with_anomalies:
                print(f"\n📋 Analyzing: {node_path}")
                
                try:
                    anomaly_state = period_anomalies.get(node_path, "?")
                    deviation = period_deviations.get(node_path, 0)
                    
                    # For Global node, determine anomaly state based on sign of change (using same nomenclature as other nodes)
                    if node_path == "Global" and anomaly_state == "?":
                        if deviation > 0:
                            anomaly_state = "+"  # Positive anomaly
                        elif deviation < 0:
                            anomaly_state = "-"  # Negative anomaly
                        else:
                            anomaly_state = "N"  # Neutral
                    
                    # Calculate anomaly magnitude from deviation since NPS values not available
                    anomaly_magnitude = deviation
                    
                    # Collect explanation with agent workflow info
                    start_date, end_date = None, None
                    if analysis_date:
                        start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
                    
                    # Capture workflow info during explanation
                    explanation = await asyncio.wait_for(
                        interpreter.explain_anomaly(
                            node_path=node_path,
                            target_period=period,
                            aggregation_days=aggregation_days,
                            anomaly_state=anomaly_state,
                            anomaly_magnitude=anomaly_magnitude,  # ✅ Now passing anomaly magnitude
                            start_date=start_date,
                            end_date=end_date
                        ),
                        timeout=600.0
                    )
                    
                    explanations[node_path] = explanation
                    
                    # Show clean summary of the explanation
                    if explanation and "🤖 **AGENT CAUSAL ANALYSIS**" in explanation:
                        clean_explanation = explanation.split("🤖 **AGENT CAUSAL ANALYSIS**\n")[1]
                        print(f"   📊 Result: {clean_explanation[:200]}...")
                    
                except Exception as e:
                    explanations[node_path] = f"Analysis failed: {str(e)}"
                    print(f"   ❌ Analysis failed: {str(e)}")
        
        # AI Tree Interpretation
        print(f"🔍 DEBUG AI INTERPRETER: ai_available={ai_available}, explanations_count={len(explanations) if explanations else 0}")
        if explanations:
            print(f"🔍 DEBUG AI INTERPRETER: explanations keys={list(explanations.keys())}")
            for node, exp in explanations.items():
                print(f"🔍 DEBUG AI INTERPRETER: {node} has explanation length={len(exp) if exp else 0}")
        
        if ai_available and explanations:
            print(f"\n🧠 TREE INTERPRETER SUMMARY:")
            print("-" * 40)
            
            try:
                # Generate comparison context for interpreter
                interpreter_comparison_context = generate_comparison_context(
                    analysis_data.get('anomaly_detection_mode', 'mean'),
                    analysis_data.get('aggregation_days', 1),
                    analysis_data.get('baseline_periods', 7)
                )
                
                # Always use the complete tree format with integrated causal explanations
                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                               parent_interpretations, explanations, date_range, segment, None,
                                               comparison_context=interpreter_comparison_context)
                
                print(f"🔍 Using complete tree format with integrated explanations: {len(ai_input)} characters")
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, 
                                                   start_date.strftime('%Y-%m-%d') if date_range else None, segment),
                    timeout=600.0
                )
                
                print("🎯 IMPRIMIENDO INTERPRETACIÓN FINAL:")
                print(ai_interpretation)
                
            except Exception as e:
                print(f"❌ Tree interpretation failed: {str(e)}")
        
        # Collect period data for final summary
        period_data = {
            'period': period,
            'date_range': date_range_str,
            'ai_interpretation': ai_interpretation if 'ai_interpretation' in locals() else "No interpretation available"
        }
        all_periods_data.append(period_data)
    
    return all_periods_data

async def print_clean_tree_only(anomalies: dict, deviations: dict, interpretations: dict, segment_filter: str = "Global", nps_values: dict = None):
    """Print only the tree structure without explanations"""
    
    def get_state_description(state):
        if state == '+':
            return "POSITIVE ANOMALY"
        elif state == '-':
            return "NEGATIVE ANOMALY"
        else:
            return "Normal"
    
    def get_deviation_text(node_path):
        deviation = deviations.get(node_path, 0)
        deviation_text = f" ({deviation:+.1f} pts)" if deviation != 0 else ""
        
        # Add NPS values if available - show them more prominently
        nps_info = ""
        if nps_values and node_path in nps_values:
            nps_data = nps_values[node_path]
            if isinstance(nps_data, dict):
                current_nps = nps_data.get('current', 'N/A')
                baseline_nps = nps_data.get('baseline', 'N/A')
                if isinstance(current_nps, (int, float)) and isinstance(baseline_nps, (int, float)):
                    difference = current_nps - baseline_nps
                    nps_info = f" [NPS: {current_nps:.1f} vs {baseline_nps:.1f}, diff: {difference:+.1f}]"
                else:
                    nps_info = f" [NPS: {current_nps} vs {baseline_nps}]"
            else:
                nps_info = f" [NPS: {nps_data}]"
        
        return deviation_text + nps_info
    
    def print_interpretation(node_path, indent=""):
        interp = interpretations.get(node_path, "")
        if interp:
            print(f"{indent}  └─ Pattern: {interp}")
    
    # Normalize segment_filter for correct tree printing
    normalized_segment = segment_filter
    if segment_filter == 'SH':
        normalized_segment = 'Global/SH'
    elif segment_filter == 'LH':
        normalized_segment = 'Global/LH'
    elif segment_filter == 'Economy' and '/' not in segment_filter:
        normalized_segment = 'Global/SH/Economy'
    elif segment_filter == 'Business' and '/' not in segment_filter:
        normalized_segment = 'Global/SH/Business'
    elif segment_filter == 'Business/LH':
        normalized_segment = 'Global/LH/Business'
    elif segment_filter == 'Economy/LH':
        normalized_segment = 'Global/LH/Economy'
    elif segment_filter == 'Premium/LH':
        normalized_segment = 'Global/LH/Premium'
    elif segment_filter == 'Business/SH':
        normalized_segment = 'Global/SH/Business'
    elif segment_filter == 'Economy/SH':
        normalized_segment = 'Global/SH/Economy'
    
    # Print tree based on normalized segment filter
    if normalized_segment == "Global":
        print_full_tree_clean(anomalies, get_state_description, get_deviation_text, print_interpretation)
    elif normalized_segment == "Global/LH":
        print_lh_tree_clean(anomalies, get_state_description, get_deviation_text, print_interpretation)
    elif normalized_segment == "Global/SH":
        print_sh_tree_clean(anomalies, get_state_description, get_deviation_text, print_interpretation)
    else:
        print_single_node_clean(normalized_segment, anomalies, get_state_description, get_deviation_text, print_interpretation)

def print_full_tree_clean(anomalies, get_state_description, get_deviation_text, print_interpretation):
    """Print clean full tree without explanations"""
    # Global
    global_state = get_state_description(anomalies.get("Global", "?"))
    global_dev = get_deviation_text("Global")
    print(f"Global: {global_state}{global_dev}")
    print_interpretation("Global", "")
    
    # LH branch
    lh_state = get_state_description(anomalies.get("Global/LH", "?"))
    lh_dev = get_deviation_text("Global/LH")
    print(f"├─ Long Haul (LH): {lh_state}{lh_dev}")
    print_interpretation("Global/LH", "│")
    
    lh_cabins = ["Economy", "Business", "Premium"]
    for i, cabin in enumerate(lh_cabins):
        node = f"Global/LH/{cabin}"
        if node in anomalies:
            state = get_state_description(anomalies[node])
            dev = get_deviation_text(node)
            print(f"│  ├── {cabin}: {state}{dev}")
            print_interpretation(node, "│  ")
    
    # SH branch
    sh_state = get_state_description(anomalies.get("Global/SH", "?"))
    sh_dev = get_deviation_text("Global/SH")
    print(f"└─ Short Haul (SH): {sh_state}{sh_dev}")
    print_interpretation("Global/SH", " ")
    
    # SH children
    for cabin in ["Economy", "Business"]:
        node = f"Global/SH/{cabin}"
        if node in anomalies:
            state = get_state_description(anomalies[node])
            dev = get_deviation_text(node)
            print(f"   ├── {cabin}: {state}{dev}")
            print_interpretation(node, "   ")
            
            # Show company breakdown for SH
            for company in ["IB", "YW"]:
                company_node = f"{node}/{company}"
                if company_node in anomalies:
                    comp_state = get_state_description(anomalies[company_node])
                    comp_dev = get_deviation_text(company_node)
                    print(f"   │  ├──── {company}: {comp_state}{comp_dev}")
                    print_interpretation(company_node, "   │  ")

def print_lh_tree_clean(anomalies, get_state_description, get_deviation_text, print_interpretation):
    """Print clean LH tree"""
    lh_state = get_state_description(anomalies.get("Global/LH", "?"))
    lh_dev = get_deviation_text("Global/LH")
    print(f"Long Haul (LH): {lh_state}{lh_dev}")
    print_interpretation("Global/LH", "")
    
    for cabin in ["Economy", "Business", "Premium"]:
        node = f"Global/LH/{cabin}"
        if node in anomalies:
            state = get_state_description(anomalies[node])
            dev = get_deviation_text(node)
            print(f"├── {cabin}: {state}{dev}")
            print_interpretation(node, "")

def print_sh_tree_clean(anomalies, get_state_description, get_deviation_text, print_interpretation):
    """Print clean SH tree"""
    sh_state = get_state_description(anomalies.get("Global/SH", "?"))
    sh_dev = get_deviation_text("Global/SH")
    print(f"Short Haul (SH): {sh_state}{sh_dev}")
    print_interpretation("Global/SH", "")
    
    for cabin in ["Economy", "Business"]:
        node = f"Global/SH/{cabin}"
        if node in anomalies:
            state = get_state_description(anomalies[node])
            dev = get_deviation_text(node)
            print(f"├── {cabin}: {state}{dev}")
            print_interpretation(node, "")
            
            for company in ["IB", "YW"]:
                company_node = f"{node}/{company}"
                if company_node in anomalies:
                    comp_state = get_state_description(anomalies[company_node])
                    comp_dev = get_deviation_text(company_node)
                    print(f"│  ├──── {company}: {comp_state}{comp_dev}")
                    print_interpretation(company_node, "│  ")

def print_single_node_clean(node_path, anomalies, get_state_description, get_deviation_text, print_interpretation):
    """Print single node clean"""
    if node_path in anomalies:
        state = get_state_description(anomalies[node_path])
        dev = get_deviation_text(node_path)
        node_name = node_path.split('/')[-1] if '/' in node_path else node_path
        print(f"{node_name}: {state}{dev}")
        print_interpretation(node_path, "")

async def main():
    """Enhanced main entry point for comprehensive anomaly analysis"""
    print("🚀 Enhanced Flexible NPS Anomaly Detection System")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deep Research Period - Custom NPS Anomaly Analysis')
    parser.add_argument('--study-mode', choices=['single', 'comparative'], default='comparative',
                       help='Analysis mode: single (no comparison) or comparative (with comparison). Default: comparative')
    parser.add_argument('--aggregation-days', type=int, default=1,
                       help='Number of days per aggregation period (default: 1)')
    parser.add_argument('--periods', type=int, default=74, help="Number of periods to download/analyze")
    
    # New date-related parameters - clearer logic
    parser.add_argument('--insert-date-ci', type=str,
                       help='Simulate today being this date (YYYY-MM-DD). Data available until this date - 4 days')
    parser.add_argument('--date-flight-local', type=str,
                       help='Use this date directly as available in dashboard (YYYY-MM-DD)')
    
    # New segment parameter for focused analysis
    parser.add_argument('--segment', type=str, default='Global',
                       help='Root segment to analyze (e.g., Global, SH, Global/SH/Economy, Global/LH). Default: Global (full tree)')
    
    # Anomaly detection mode parameter
    parser.add_argument('--anomaly-detection-mode', choices=['target', 'mean', 'vslast'], default='target',
                       help='Anomaly detection mode: target (uses monthly targets), mean (uses mean of last n periods), or vslast (compares against previous period). Default: target')
    parser.add_argument('--baseline-periods', type=int, default=7,
                       help='Number of previous periods to use as a baseline for "mean" anomaly detection')
    
    # Causal comparison filter parameters
    parser.add_argument('--causal-filter-comparison',
                        type=str,
                        default="vs L7d",
                        help='Causal filter for comparative analysis (e.g., "vs L7d", "vs LM", "vs LY", "vs Target", "vs Sel. Period")')
    # Comparison dates for selected period analysis
    parser.add_argument('--comparison-start-date', type=str,
                        help='Start date for comparison period (YYYY-MM-DD) when using --causal-filter-comparison "vs Sel. Period"')
    parser.add_argument('--comparison-end-date', type=str,
                        help='End date for comparison period (YYYY-MM-DD) when using --causal-filter-comparison "vs Sel. Period"')
    
    # Environment parameter
    parser.add_argument('--environment', type=str, default='prod', choices=['local', 'prod'],
                       help='Environment: local (reads .env) or prod (uses system env vars). Default: local')
    
    args = parser.parse_args()

    # Add placeholders for arguments that might not be defined by the parser in all cases
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
            analysis_date = datetime.combine(simulated_today - timedelta(days=pbi_lag_days), datetime.min.time())  # Simulate 4-day lag
            date_parameter = 'insert_ci'
            date_description = f"Simulating today as {simulated_today.strftime('%Y-%m-%d')}"
        except ValueError:
            print("❌ Error: --insert-date-ci must be in YYYY-MM-DD format")
            return
    elif args.date_flight_local:
        try:
            analysis_date = datetime.strptime(args.date_flight_local, '%Y-%m-%d')
            date_parameter = 'flight_local'
            date_description = f"Using direct dashboard date"
        except ValueError:
            print("❌ Error: --date-flight-local must be in YYYY-MM-DD format")
            return
    else:
        # Default behavior: use available date (today - 4 days)
        analysis_date = datetime.combine(today - timedelta(days=pbi_lag_days), datetime.min.time())
        date_parameter = 'available'
        date_description = f"Using default available date (today - {pbi_lag_days} days)"
    
    # Display date information
    print(f"\n📅 DATE CONFIGURATION:")
    print(f"   • {date_description}")
    print(f"   • Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
    if args.insert_date_ci:
        print(f"   • Note: Simulating {pbi_lag_days}-day lag from {args.insert_date_ci}")
    elif args.date_flight_local:
        print(f"   • Note: Using date directly from dashboard without lag simulation")
    
    # Process comparison dates if provided
    comparison_start_date = None
    comparison_end_date = None
    if args.comparison_start_date and args.comparison_end_date:
        try:
            comparison_start_date = datetime.strptime(args.comparison_start_date, '%Y-%m-%d')
            comparison_end_date = datetime.strptime(args.comparison_end_date, '%Y-%m-%d')
            print(f"   • Comparison period: {comparison_start_date.strftime('%Y-%m-%d')} to {comparison_end_date.strftime('%Y-%m-%d')}")
        except ValueError:
            print("❌ Error: Comparison dates must be in YYYY-MM-DD format")
            return
    
    # --- Date & Lag Configuration ---
    # ... (existing date logic) ...

    # --- Study Mode Configuration ---
    print("\n🔬 STUDY MODE CONFIGURATION:")
    # Determine study_mode automatically if not specified
    if not args.study_mode:
        if args.aggregation_days == 1:
            study_mode = "single"
        else:
            study_mode = "comparative"
    else:
        study_mode = args.study_mode
    
    print(f"   • Study mode: {study_mode.upper()}")
    
    # --- EXECUTION LOGIC ---
    try:
        print("\n🔬 RUNNING CUSTOM PERIOD ANALYSIS")
        print("=" * 60)
        await execute_analysis_flow(
            analysis_date=analysis_date,
            date_parameter=date_parameter,
            segment=args.segment,
            anomaly_detection_mode=args.anomaly_detection_mode,
            baseline_periods=args.baseline_periods,
            aggregation_days=args.aggregation_days,
            periods=args.periods,
            causal_filter=args.causal_filter_comparison,
            comparison_start_date=args.comparison_start_date,
            comparison_end_date=args.comparison_end_date,
            date_flight_local=args.date_flight_local,
            study_mode=args.study_mode,
            environment=args.environment
        )

    except KeyboardInterrupt:
        print("\n⏸️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")

def print_full_tree(anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation):
    """Print the complete Global tree"""
    # Global summary
    global_state = anomalies.get("Global", "?")
    global_dev = get_deviation_text("Global")
    global_desc = get_state_description(global_state)
    
    print(f"Global: {global_desc} {global_dev}")
    actual_anomalies = [node for node, state in anomalies.items() if state in ['+', '-']]
    if global_state == "N" and actual_anomalies:
        print(f"  └─ Note: Global shows normal variation despite {len(actual_anomalies)} segment anomalies below")
    elif global_state in ["+", "-"]:
        print(f"  └─ Global anomaly detected: investigate segments below")
    print_interpretation("Global", "")
    if global_state in ["+", "-"]:
        print_explanation("Global", "")
    
    # Long Haul
    lh_state = anomalies.get("Global/LH", "?")
    lh_dev = get_deviation_text("Global/LH")
    lh_desc = get_state_description(lh_state)
    
    print(f"\n├─ Long Haul (LH): {lh_desc} {lh_dev}")
    print_interpretation("Global/LH", "│")
    if lh_state in ["+", "-"]:
        print_explanation("Global/LH", "│")
    
    lh_cabins = ["Economy", "Business", "Premium"]
    for i, cabin in enumerate(lh_cabins):
        cabin_path = f"Global/LH/{cabin}"
        cabin_state = anomalies.get(cabin_path, "?")
        cabin_dev = get_deviation_text(cabin_path)
        cabin_desc = get_state_description(cabin_state)
        
        connector = "├──" if i < len(lh_cabins) - 1 else "└──"
        print(f"│  {connector} {cabin}: {cabin_desc} {cabin_dev}")
        print_interpretation(cabin_path, "│  ")
        if cabin_state in ["+", "-"]:
            print_explanation(cabin_path, "│  ")
    
    # Short Haul
    sh_state = anomalies.get("Global/SH", "?")
    sh_dev = get_deviation_text("Global/SH")
    sh_desc = get_state_description(sh_state)
    
    print(f"\n└─ Short Haul (SH): {sh_desc} {sh_dev}")
    print_interpretation("Global/SH", " ")
    if sh_state in ["+", "-"]:
        print_explanation("Global/SH", "")
    
    sh_cabins = ["Economy", "Business"]
    for i, cabin in enumerate(sh_cabins):
        cabin_path = f"Global/SH/{cabin}"
        cabin_state = anomalies.get(cabin_path, "?")
        cabin_dev = get_deviation_text(cabin_path)
        cabin_desc = get_state_description(cabin_state)
        
        connector = "├──" if i < len(sh_cabins) - 1 else "└──"
        print(f"   {connector} {cabin}: {cabin_desc} {cabin_dev}")
        print_interpretation(cabin_path, "   ")
        if cabin_state in ["+", "-"]:
            print_explanation(cabin_path, "   ")
        
        # Company subdivisions for SH
        companies = ["IB", "YW"]
        for j, company in enumerate(companies):
            company_path = f"Global/SH/{cabin}/{company}"
            company_state = anomalies.get(company_path, "?")
            company_dev = get_deviation_text(company_path)
            company_desc = get_state_description(company_state)
            
            # Adjust indentation based on SH cabin position
            if i < len(sh_cabins) - 1:  # Not the last cabin
                company_connector = "├────" if j < len(companies) - 1 else "└────"
                print(f"   │  {company_connector} {company}: {company_desc} {company_dev}")
                company_indent = "   │  "
            else:  # Last cabin
                company_connector = "├────" if j < len(companies) - 1 else "└────"
                print(f"      {company_connector} {company}: {company_desc} {company_dev}")
                company_indent = "      "
            
            print_interpretation(company_path, company_indent)
            if company_state in ["+", "-"]:
                print_explanation(company_path, company_indent)

def print_lh_tree(anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation):
    """Print only the Long Haul tree"""
    # Long Haul root
    lh_state = anomalies.get("Global/LH", "?")
    lh_dev = get_deviation_text("Global/LH")
    lh_desc = get_state_description(lh_state)
    
    print(f"Long Haul (LH): {lh_desc} {lh_dev}")
    print_interpretation("Global/LH", "")
    if lh_state in ["+", "-"]:
        print_explanation("Global/LH", "")
    
    # LH cabins
    lh_cabins = ["Economy", "Business", "Premium"]
    for i, cabin in enumerate(lh_cabins):
        cabin_path = f"Global/LH/{cabin}"
        cabin_state = anomalies.get(cabin_path, "?")
        cabin_dev = get_deviation_text(cabin_path)
        cabin_desc = get_state_description(cabin_state)
        
        connector = "├──" if i < len(lh_cabins) - 1 else "└──"
        print(f"{connector} {cabin}: {cabin_desc} {cabin_dev}")
        print_interpretation(cabin_path, "")
        if cabin_state in ["+", "-"]:
            print_explanation(cabin_path, "")

def print_sh_tree(anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation):
    """Print only the Short Haul tree"""
    # Short Haul root
    sh_state = anomalies.get("Global/SH", "?")
    sh_dev = get_deviation_text("Global/SH")
    sh_desc = get_state_description(sh_state)
    
    print(f"Short Haul (SH): {sh_desc} {sh_dev}")
    print_interpretation("Global/SH", "")
    if sh_state in ["+", "-"]:
        print_explanation("Global/SH", "")
    
    # SH cabins
    sh_cabins = ["Economy", "Business"]
    for i, cabin in enumerate(sh_cabins):
        cabin_path = f"Global/SH/{cabin}"
        cabin_state = anomalies.get(cabin_path, "?")
        cabin_dev = get_deviation_text(cabin_path)
        cabin_desc = get_state_description(cabin_state)
        
        connector = "├──" if i < len(sh_cabins) - 1 else "└──"
        print(f"{connector} {cabin}: {cabin_desc} {cabin_dev}")
        print_interpretation(cabin_path, "")
        if cabin_state in ["+", "-"]:
            print_explanation(cabin_path, "")
        
        # Company subdivisions
        companies = ["IB", "YW"]
        for j, company in enumerate(companies):
            company_path = f"Global/SH/{cabin}/{company}"
            company_state = anomalies.get(company_path, "?")
            company_dev = get_deviation_text(company_path)
            company_desc = get_state_description(company_state)
            
            # Adjust indentation based on cabin position
            if i < len(sh_cabins) - 1:  # Not the last cabin
                company_connector = "├────" if j < len(companies) - 1 else "└────"
                print(f"│  {company_connector} {company}: {company_desc} {company_dev}")
                company_indent = "│  "
            else:  # Last cabin
                company_connector = "├────" if j < len(companies) - 1 else "└────"
                print(f"   {company_connector} {company}: {company_desc} {company_dev}")
                company_indent = "   "
            
            print_interpretation(company_path, company_indent)
            if company_state in ["+", "-"]:
                print_explanation(company_path, company_indent)

def print_sh_economy_tree(anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation):
    """Print only the SH Economy tree"""
    # SH Economy root
    economy_state = anomalies.get("Global/SH/Economy", "?")
    economy_dev = get_deviation_text("Global/SH/Economy")
    economy_desc = get_state_description(economy_state)
    
    print(f"SH Economy: {economy_desc} {economy_dev}")
    print_interpretation("Global/SH/Economy", "")
    if economy_state in ["+", "-"]:
        print_explanation("Global/SH/Economy", "")
    
    # IB/YW subdivisions
    companies = ["IB", "YW"]
    for j, company in enumerate(companies):
        company_path = f"Global/SH/Economy/{company}"
        company_state = anomalies.get(company_path, "?")
        company_dev = get_deviation_text(company_path)
        company_desc = get_state_description(company_state)
        
        connector = "├──" if j < len(companies) - 1 else "└──"
        print(f"{connector} {company}: {company_desc} {company_dev}")
        print_interpretation(company_path, "")
        if company_state in ["+", "-"]:
            print_explanation(company_path, "")

def print_sh_business_tree(anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation):
    """Print only the SH Business tree"""
    # SH Business root
    business_state = anomalies.get("Global/SH/Business", "?")
    business_dev = get_deviation_text("Global/SH/Business")
    business_desc = get_state_description(business_state)
    
    print(f"SH Business: {business_desc} {business_dev}")
    print_interpretation("Global/SH/Business", "")
    if business_state in ["+", "-"]:
        print_explanation("Global/SH/Business", "")
    
    # IB/YW subdivisions
    companies = ["IB", "YW"]
    for j, company in enumerate(companies):
        company_path = f"Global/SH/Business/{company}"
        company_state = anomalies.get(company_path, "?")
        company_dev = get_deviation_text(company_path)
        company_desc = get_state_description(company_state)
        
        connector = "├──" if j < len(companies) - 1 else "└──"
        print(f"{connector} {company}: {company_desc} {company_dev}")
        print_interpretation(company_path, "")
        if company_state in ["+", "-"]:
            print_explanation(company_path, "")

def print_single_node(node_path, anomalies, get_state_description, get_deviation_text, print_interpretation, print_explanation):
    """Print a single node (for leaf segments)"""
    # Debug: print available keys in anomalies dict
    print(f"DEBUG: Looking for node_path: {node_path}")
    print(f"DEBUG: Available keys in anomalies: {list(anomalies.keys())}")
    node_state = anomalies.get(node_path, "?")
    print(f"DEBUG: node_state for {node_path}: {node_state}")
    node_dev = get_deviation_text(node_path)
    node_desc = get_state_description(node_state)
    
    # Extract readable name from path
    if node_path == "Global/LH/Economy":
        node_name = "LH Economy"
    elif node_path == "Global/LH/Business":
        node_name = "LH Business"
    elif node_path == "Global/LH/Premium":
        node_name = "LH Premium"
    elif node_path == "Global/SH/Economy/IB":
        node_name = "SH Economy IB"
    elif node_path == "Global/SH/Economy/YW":
        node_name = "SH Economy YW"
    elif node_path == "Global/SH/Business/IB":
        node_name = "SH Business IB"
    elif node_path == "Global/SH/Business/YW":
        node_name = "SH Business YW"
    else:
        node_name = node_path.split('/')[-1] if '/' in node_path else node_path
    
    print(f"{node_name}: {node_desc} {node_dev}")
    print_interpretation(node_path, "")
    if node_state in ["+", "-"]:
        print_explanation(node_path, "")

def calculate_period_date_range(analysis_date: datetime, target_period: int, aggregation_days: int) -> tuple:
    """
    Calculate the correct date range for a period relative to the analysis date
    
    Args:
        analysis_date: The reference date for the analysis (e.g., 2025-01-20)
        target_period: Period number (1 = most recent relative to analysis_date)
        aggregation_days: Days per period (1, 7, 14, 30, etc.)
        
    Returns:
        Tuple of (start_date, end_date) for the target period
        
    Examples:
        For daily analysis (aggregation_days=1) with analysis_date=2025-01-20:
        - Period 1: (2025-01-20, 2025-01-20) - the analysis date itself
        - Period 2: (2025-01-19, 2025-01-19) - 1 day before
        - Period 3: (2025-01-18, 2025-01-18) - 2 days before
        
        For weekly analysis (aggregation_days=7) with analysis_date=2025-01-20:
        - Period 1: (2025-01-14, 2025-01-20) - week ending on analysis date
        - Period 2: (2025-01-07, 2025-01-13) - previous week
    """
    print(f"🔍 DEBUG calculate_period_date_range: analysis_date={analysis_date}, target_period={target_period}, aggregation_days={aggregation_days}")
    
    # Calculate how many days back from analysis date
    days_back = (target_period - 1) * aggregation_days
    print(f"🔍 DEBUG calculate_period_date_range: days_back = (target_period - 1) * aggregation_days = ({target_period} - 1) * {aggregation_days} = {days_back}")
    
    # For daily analysis, each period is exactly one day
    # For weekly analysis, each period is 7 days, etc.
    period_end = analysis_date - timedelta(days=days_back)
    period_start = period_end - timedelta(days=aggregation_days - 1)
    
    print(f"🔍 DEBUG calculate_period_date_range: period_end = analysis_date - {days_back} days = {period_end}")
    print(f"🔍 DEBUG calculate_period_date_range: period_start = period_end - {aggregation_days - 1} days = {period_start}")
    print(f"🔍 DEBUG calculate_period_date_range: returning ({period_start}, {period_end})")
    
    return period_start, period_end

def calculate_actual_period_number(analysis_date: datetime, today_date: datetime = None) -> int:
    """
    Calculate the actual period number in the PBI data for a given analysis date
    
    Args:
        analysis_date: The date we want to analyze (e.g., 2025-01-20)
        today_date: Today's date (defaults to actual today)
        
    Returns:
        The period number in the PBI data that corresponds to the analysis date
        
    Example:
        If today is 2025-06-05 and analysis_date is 2025-01-20:
        Days difference = 137, so analysis_date is Period 137
    """
    if today_date is None:
        today_date = datetime.now().date()
    elif isinstance(today_date, datetime):
        today_date = today_date.date()
    
    if isinstance(analysis_date, datetime):
        analysis_date = analysis_date.date()
    
    # Calculate days between today and analysis date
    days_diff = (today_date - analysis_date).days
    
    # Period number = days difference + 1 (Period 1 = today)
    period_number = days_diff + 1
    
    return period_number

def detect_parent_child_relationships(node_paths: list) -> dict:
    """
    Detect parent-child relationships among anomalous nodes.
    Returns a dictionary with relationship information for each node.
    """
    relationships = {}
    
    for node_path in node_paths:
        relationships[node_path] = {
            'type': 'standalone',
            'parent': None,
            'children': [],
            'all_related': [node_path]
        }
    
    # Detect parent-child relationships
    for node_path in node_paths:
        path_parts = node_path.split('/')
        
        # Check if this node is a parent of any other nodes
        children = []
        for other_path in node_paths:
            if other_path != node_path and other_path.startswith(node_path + '/'):
                # Check if it's a direct child (not grandchild)
                other_parts = other_path.split('/')
                if len(other_parts) == len(path_parts) + 1:
                    children.append(other_path)
        
        if children:
            relationships[node_path]['type'] = 'parent'
            relationships[node_path]['children'] = children
            relationships[node_path]['all_related'] = [node_path] + children
            
            # Mark children as having this parent
            for child in children:
                relationships[child]['type'] = 'child'
                relationships[child]['parent'] = node_path
                relationships[child]['all_related'] = [node_path] + children
    
    return relationships

def should_consolidate_explanations(causal_explanations: dict, relationships: dict) -> bool:
    """
    Determine if parent-child explanations should be consolidated based on cause similarity.
    Returns True if causes appear to be common/related, False if distinct.
    """
    if len(causal_explanations) <= 1:
        return False
    
    # Look for parent-child groups
    for node_path, rel_info in relationships.items():
        if rel_info['type'] == 'parent' and len(rel_info['children']) > 0:
            parent_explanation = causal_explanations.get(node_path, "")
            
            # Check for operational/service-related keywords that suggest common causes
            operational_keywords = [
                'punctuality', 'delay', 'puntualidad', 'retraso', 'retrasos',
                'boarding', 'embarque', 'crew', 'tripulación', 'tripulacion',
                'arrivals', 'llegada', 'llegadas', 'operational', 'operacional',
                'operativo', 'service', 'servicio', 'incident', 'incidente',
                'technical', 'técnico', 'tecnico', 'system', 'sistema'
            ]
            
            # Count operational keywords in parent explanation
            parent_ops_count = sum(1 for keyword in operational_keywords 
                                  if keyword.lower() in parent_explanation.lower())
            
            # Check children explanations
            common_cause_indicators = 0
            total_children = len(rel_info['children'])
            
            for child_node in rel_info['children']:
                child_explanation = causal_explanations.get(child_node, "")
                child_ops_count = sum(1 for keyword in operational_keywords 
                                     if keyword.lower() in child_explanation.lower())
                
                # If both parent and child mention operational issues
                if parent_ops_count > 0 and child_ops_count > 0:
                    common_cause_indicators += 1
            
            # Consolidate if majority of children show similar operational causes
            if common_cause_indicators >= (total_children * 0.6):  # 60% threshold
                return True
    
    return False

async def execute_analysis_flow(
    analysis_date: datetime,
    date_parameter: str,
    segment: str,
    anomaly_detection_mode: str,
    baseline_periods: int,
    aggregation_days: int,
    periods: int,
    causal_filter: Optional[str],
    comparison_start_date: Optional[datetime] = None,
    comparison_end_date: Optional[datetime] = None,
    date_flight_local: Optional[str] = None,
    study_mode: str = "comparative",
    environment: str = "prod",
) -> str:
    """
    Executes a complete analysis flow for a given configuration.
    This includes data download, anomaly detection, and interpretation.
    """
    
    print(f"\n🚀 DEBUG: execute_analysis_flow CALLED!")
    print(f"🔍 Parameters: segment={segment}, study_mode={study_mode}, environment={environment}")
    print(f"🔍 Parameters: causal_filter={causal_filter}, comparison_dates={comparison_start_date} to {comparison_end_date}")

    # Adjust causal_filter based on study_mode
    if study_mode == "single":
        causal_filter = None
    
    # Get baseline description for display
    if study_mode == "single":
        # For single mode, baseline is mean of last N periods
        if aggregation_days == 1:
            baseline_desc = f"media de los últimos {baseline_periods} días"
        else:
            baseline_desc = f"media de los últimos {baseline_periods} períodos de {aggregation_days} días"
    else:
        # For comparative mode, use causal filter
        comp_start_str = comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date and hasattr(comparison_start_date, 'strftime') else str(comparison_start_date) if comparison_start_date else None
        comp_end_str = comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date and hasattr(comparison_end_date, 'strftime') else str(comparison_end_date) if comparison_end_date else None
        _, baseline_desc = determine_anomaly_mode_for_vslast(causal_filter or "vs L7d", comp_start_str, comp_end_str)
    
    print(f"\n{'='*60}")
    print(f"🚀 EXECUTING ANALYSIS FLOW")
    print(f"   - Study Mode: {study_mode.upper() if study_mode else 'AUTO-DETECTED'}")
    print(f"   - Aggregation: {aggregation_days} days")
    print(f"   - Periods: {periods}")
    print(f"   - Causal Filter: {causal_filter}")
    print(f"   - Baseline: {baseline_desc}")
    print(f"{'='*60}")

    # 1. Data Download
    data_folder = await run_flexible_data_download_silent_with_date(
        aggregation_days=aggregation_days,
        periods=periods,
        start_date=analysis_date,
        date_parameter=f"{date_parameter}_{study_mode}_{aggregation_days}d",
        segment=segment,
        environment=environment
    )

    if not data_folder:
        print(f"❌ Data collection failed for {study_mode} {aggregation_days}d analysis.")
        return None

    # 2. Analysis
    analysis_data = await run_flexible_analysis_silent(
        data_folder,
        analysis_date,
        date_parameter,
        anomaly_detection_mode,
        baseline_periods,
        causal_filter,
        periods=periods,
        causal_comparison_dates=(str(comparison_start_date), str(comparison_end_date)) if comparison_start_date and comparison_end_date else None,
        segment=segment,
        environment=environment
    )

    if not analysis_data or not analysis_data.get('anomaly_periods'):
        print(f"✅ No anomalies found for {study_mode} {aggregation_days}d analysis.")
        return None

    # 3. Get Summary with AI Interpretation (using full analysis instead of silent)
    print(f"\n🔍 DEBUG EXECUTE_ANALYSIS_FLOW: About to call show_all_anomaly_periods_with_explanations")
    print(f"🔍 DEBUG: analysis_data type={type(analysis_data)}, has_anomaly_periods={bool(analysis_data and analysis_data.get('anomaly_periods'))}")
    print(f"🔍 DEBUG: segment={segment}, causal_filter={causal_filter}")
    
    summary_data = await show_all_anomaly_periods_with_explanations(
        analysis_data,
        segment=segment,
        causal_filter=causal_filter,
        comparison_start_date=comparison_start_date,
        comparison_end_date=comparison_end_date,
        environment=environment,
        study_mode=study_mode
    )
    
    print(f"🔍 DEBUG EXECUTE_ANALYSIS_FLOW: show_all_anomaly_periods_with_explanations completed")
    print(f"🔍 DEBUG: summary_data type={type(summary_data)}")
    
    return summary_data

if __name__ == "__main__":
    # To run this from the command line:
    # python -m dashboard_analyzer.main --mode both --segment "Global" --date-flight-local "2025-01-15"
    asyncio.run(main())
