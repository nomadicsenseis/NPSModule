#!/usr/bin/env python3
"""
Enhanced Flexible NPS Anomaly Detection System
Shows all periods with anomalies and includes explanations
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_detection.flexible_detector import FlexibleAnomalyDetector
from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter

# Global debug flag
DEBUG_MODE = False

def debug_print(message):
    """Print debug message only when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG: {message}")

async def collect_flexible_data(aggregation_days: int, target_folder: str, segment: str = "Global", analysis_date: datetime = None):
    """
    Collect flexible NPS data for all nodes in the specified segment
    
    Args:
        aggregation_days: Number of days per period
        target_folder: Where to save the data
        segment: Root segment to collect (Global, SH, LH, etc.)
        analysis_date: Optional analysis date to use instead of TODAY() in queries
    """
    print(f"📥 Collecting flexible NPS data")
    print(f"   🔧 Aggregation: {aggregation_days} days per period")
    print(f"   📁 Target folder: {target_folder}")
    print(f"   🎯 Segment: {segment}")
    if analysis_date:
        print(f"   📅 Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
    
    collector = PBIDataCollector()
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

async def generate_explanations(analysis_data: dict):
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
    pbi_collector = PBIDataCollector()
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector)
    
    explanation_count = 0
    total_nodes_analyzed = 0
    
    for period in anomaly_periods[:4]:  # Analyze up to 4 periods with anomalies
        print(f"\n{'='*50}")
        print(f"🔍 PERIOD {period} EXPLANATIONS")
        print("="*50)
        
        # Get anomalies for this period
        period_anomalies, period_deviations, _ = detector.analyze_period(data_folder, period)
        
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
        for i, node_path in enumerate(nodes_needing_explanation[:3], 1):  # Limit to 3 per period
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
                        anomaly_state=state
                    ),
                    timeout=60.0  # 60 second timeout for comprehensive analysis
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

async def show_all_anomaly_periods_with_explanations(analysis_data: dict, segment: str = "Global"):
    """Show trees for all periods analyzed INCLUDING explanations and parent interpretations"""
    if not analysis_data:
        return
    
    print(f"\n🌳 ANOMALY PERIOD ANALYSIS")
    print("-" * 50)
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    aggregation_days = analysis_data['aggregation_days']
    anomaly_periods = analysis_data['anomaly_periods']
    periods_analyzed = analysis_data.get('periods_analyzed', anomaly_periods)
    
    # Initialize interpreter for explanations
    pbi_collector = PBIDataCollector()
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector)
    
    # Initialize AI agent for interpretation
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=LLMType.O3,
            logger=logging.getLogger("ai_interpreter")
        )
        ai_available = True
        print("🤖 AI Agent initialized for interpretations")
    except Exception as e:
        print(f"⚠️ AI Agent not available: {str(e)}")
        ai_available = False
    
    # Initialize Summary Agent for final report
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
        
        summary_agent = AnomalySummaryAgent(
            llm_type=LLMType.O3,
            logger=logging.getLogger("summary_agent")
        )
        summary_available = True
        print("📋 Summary Agent initialized for executive report")
    except Exception as e:
        print(f"⚠️ Summary Agent not available: {str(e)}")
        summary_available = False
    
    # Collect data for all periods for summary
    all_periods_data = []
    
    # Show all periods with anomalies (remove the [:3] limit)
    periods_with_anomalies = [p for p in periods_analyzed if p in anomaly_periods]
    
    # Show detailed analysis for all periods with anomalies
    for period in periods_with_anomalies:
        print(f"\n{'='*60}")
        print(f"PERIOD {period} ANALYSIS")
        print("="*60)
        
        # Get anomalies for this period (silently)
        period_anomalies, period_deviations, _ = detector.analyze_period(data_folder, period)
        
        # Get date range for this period
        date_range = interpreter._get_period_date_range(period, aggregation_days)
        if date_range:
            start_date, end_date = date_range
            print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            print(f"⚠️ Could not determine date range for period {period}")
            date_range_str = "Unknown dates"
        
        # Generate parent interpretations
        parent_interpretations = generate_parent_interpretations(period_anomalies)
        
        # Collect explanations for anomalous nodes (quietly, no verbose output)
        explanations = {}
        nodes_with_anomalies = [node for node, state in period_anomalies.items() if state in ['+', '-']]
        
        if nodes_with_anomalies:
            # Collect explanations for anomalous nodes
            for node_path in nodes_with_anomalies:
                try:
                    anomaly_state = period_anomalies.get(node_path, "?")
                    print(f"      🔍 Collecting explanation for {node_path} (state: {anomaly_state})")
                    
                    # Calculate correct date range if analysis_date is available
                    start_date, end_date = None, None
                    analysis_date = analysis_data.get('analysis_date')
                    if analysis_date:
                        start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
                        print(f"         📅 Using calculated date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    
                    explanation = await asyncio.wait_for(
                        interpreter.explain_anomaly(
                            node_path=node_path,
                            target_period=period,
                            aggregation_days=aggregation_days,
                            anomaly_state=anomaly_state,
                            start_date=start_date,
                            end_date=end_date
                        ),
                        timeout=30.0
                    )
                    explanations[node_path] = explanation
                    print(f"         ✅ Explanation collected: {len(explanation)} chars")
                except Exception as e:
                    print(f"         ❌ Explanation failed: {str(e)}")
                    explanations[node_path] = "Analysis timeout"
        
        # Debug: Print collected explanations
        debug_print(f"📊 Collected {len(explanations)} explanations:")
        for node_path, explanation in explanations.items():
            debug_print(f"  {node_path}: {explanation[:200] if explanation else 'None'}...")
        
        # Show the enhanced tree with explanations and parent interpretations
        await print_enhanced_tree_with_explanations_and_interpretations(
            period_anomalies, period_deviations, explanations, parent_interpretations,
            aggregation_days, period, date_range, segment
        )
        
        # AI Interpretation (ensure it completes)
        ai_interpretation = None
        if ai_available and nodes_with_anomalies:
            print(f"\n🤖 AI INTERPRETATION:")
            print("-" * 40)
            
            try:
                # Build comprehensive input for AI
                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                                 parent_interpretations, explanations, date_range, segment)
                
                debug_print(f"AI input string length: {len(ai_input)} characters")
                debug_print(f"AI input preview: {ai_input[:500]}...")
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, 
                                                   start_date.strftime('%Y-%m-%d') if date_range else None),
                    timeout=45.0
                )
                debug_print(f"AI interpretation received: {len(ai_interpretation) if ai_interpretation else 0} characters")
                
                print(ai_interpretation)
                print("="*60)  # Clear separator after each period
                
            except Exception as e:
                print(f"❌ AI interpretation failed: {str(e)}")
                ai_interpretation = f"AI interpretation failed: {str(e)}"
                print("="*60)
        
        # Collect period data for summary
        if summary_available:
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
        period_anomalies, period_deviations, _ = detector.analyze_period(data_folder, period)
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
    
    # Generate Executive Summary Report
    if summary_available and all_periods_data:
        print(f"\n" + "="*80)
        print(f"📋 EXECUTIVE SUMMARY REPORT")
        print("="*80)
        
        try:
            print("🤖 Generating comprehensive summary across all periods...")
            summary_report = await asyncio.wait_for(
                summary_agent.generate_summary_report(all_periods_data),
                timeout=60.0
            )
            
            print(f"\n{summary_report}")
            
            # Performance metrics
            metrics = summary_agent.get_performance_metrics()
            print(f"\n📊 Summary Generation Metrics:")
            print(f"   • Input tokens: {metrics.get('input_tokens', 0)}")
            print(f"   • Output tokens: {metrics.get('output_tokens', 0)}")
            print(f"   • LLM: {metrics.get('llm_type', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Executive summary generation failed: {str(e)}")
            print(f"   Manual review recommended for the {len(all_periods_data)} periods with anomalies")
        
        print("="*80)
    
    elif summary_available and not all_periods_data:
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print("No anomalies detected in any of the 7 periods analyzed.")
        print("All segments are operating within normal NPS variation ranges.")
    
    elif not summary_available:
        print(f"\n⚠️ Executive summary not available (Summary Agent initialization failed)")
        print(f"   Individual period analyses completed for {len(periods_with_anomalies)} periods with anomalies")

def generate_parent_interpretations(anomalies: dict) -> dict:
    """Generate parent node interpretations based on children states"""
    interpretations = {}
    
    # Helper function to format child lists
    def format_children(children_list):
        if len(children_list) == 1:
            return children_list[0]
        elif len(children_list) == 2:
            return f"{children_list[0]}, {children_list[1]}"
        else:
            return ", ".join(children_list[:-1]) + f", {children_list[-1]}"
    
    # NOTE: IB/YW are leaf nodes - they don't get patterns since they have no children
    
    # Global interpretation
    lh_state = anomalies.get("Global/LH", "?")
    sh_state = anomalies.get("Global/SH", "?")
    global_state = anomalies.get("Global", "?")
    
    if lh_state != "?" and sh_state != "?":
        if global_state == "N":
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
            # Global itself is anomalous
            contributing_children = []
            if lh_state in ["+", "-"]:
                contributing_children.append("LH")
            if sh_state in ["+", "-"]:
                contributing_children.append("SH")
            if contributing_children:
                interpretations["Global"] = f"Global anomaly driven by {format_children(contributing_children)}"
    
    # LH interpretation
    lh_children = ["Economy", "Business", "Premium"]
    lh_child_states = [anomalies.get(f"Global/LH/{child}", "?") for child in lh_children]
    valid_lh_children = [(child, state) for child, state in zip(lh_children, lh_child_states) if state != "?"]
    
    if valid_lh_children and lh_state == "N":
        normal_children = [child for child, state in valid_lh_children if state == "N"]
        positive_children = [child for child, state in valid_lh_children if state == "+"]
        negative_children = [child for child, state in valid_lh_children if state == "-"]
        
        if len(positive_children) == 0 and len(negative_children) == 0:
            interpretations["Global/LH"] = f"All children normal ({format_children([c for c, _ in valid_lh_children])})"
        elif len(positive_children) > 0 and len(negative_children) > 0:
            interpretations["Global/LH"] = f"Mixed anomalies: positive ({format_children(positive_children)}), negative ({format_children(negative_children)}), normal ({format_children(normal_children)}) balance out"
        elif len(positive_children) > 0:
            interpretations["Global/LH"] = f"Positive nodes ({format_children(positive_children)}) diluted by normal nodes ({format_children(normal_children)})"
        elif len(negative_children) > 0:
            interpretations["Global/LH"] = f"Negative nodes ({format_children(negative_children)}) diluted by normal nodes ({format_children(normal_children)})"
    
    # SH interpretation
    economy_state = anomalies.get("Global/SH/Economy", "?")
    business_state = anomalies.get("Global/SH/Business", "?")
    
    if economy_state != "?" and business_state != "?" and anomalies.get("Global/SH", "?") == "N":
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
    
    # SH/Economy interpretation (IB vs YW) - only if Economy itself is normal
    ib_eco_state = anomalies.get("Global/SH/Economy/IB", "?")
    yw_eco_state = anomalies.get("Global/SH/Economy/YW", "?")
    
    if ib_eco_state != "?" and yw_eco_state != "?" and economy_state == "N":
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
    
    # SH/Business interpretation (IB vs YW) - only if Business itself is normal
    ib_bus_state = anomalies.get("Global/SH/Business/IB", "?")
    yw_bus_state = anomalies.get("Global/SH/Business/YW", "?")
    
    if ib_bus_state != "?" and yw_bus_state != "?" and business_state == "N":
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
    
    return interpretations

def build_ai_input_string(period: int, anomalies: dict, deviations: dict, 
                         interpretations: dict, explanations: dict, date_range: tuple, segment_filter: str = "Global") -> str:
    """Build comprehensive input string for AI interpretation, filtered by segment"""
    
    # Filter data to only include relevant nodes for the selected segment
    relevant_nodes = get_segment_node_paths(segment_filter)
    filtered_anomalies = {node: state for node, state in anomalies.items() if node in relevant_nodes}
    filtered_deviations = {node: dev for node, dev in deviations.items() if node in relevant_nodes}
    
    if date_range:
        start_date, end_date = date_range
        ai_input = f"NPS ANOMALY ANALYSIS - PERIOD {period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})\n\n"
    else:
        ai_input = f"NPS ANOMALY ANALYSIS - PERIOD {period}\n\n"
    
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
        if node_path not in filtered_anomalies:
            return ""  # Skip nodes not in the filtered segment
        
        state = filtered_anomalies.get(node_path, "?")
        deviation = filtered_deviations.get(node_path, 0)
        
        ai_input_part = f"{indent}{name}: {get_state_desc(state, deviation)}\n"
        
        # Add interpretation if available
        if node_path in interpretations:
            ai_input_part += f"{indent}  └─ Pattern: {interpretations[node_path]}\n"
        
        # Add explanation if available and is actual anomaly
        if state in ['+', '-']:
            ai_input_part += f"{indent}  └─ Analysis:\n"
            
            # Initialize with default values
            routes_content = "Not enough answers for statistical analysis"
            verbatims_content = "Not enough answers for statistical analysis"
            drivers_content = "Not enough answers for statistical analysis"
            
            if node_path in explanations:
                explanation = explanations[node_path]
                
                if explanation and explanation != "Analysis timeout":
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
                        
                        elif "routes analyzed" in part or "Routes:" in part:
                            clean_part = part.replace("🛣️ Routes:", "").replace("🛣️", "").replace("Routes:", "").strip()
                            if clean_part:
                                routes_content = clean_part
                        
                        elif "Operational:" in part:
                            clean_part = part.replace("🔧 Operational:", "").replace("🔧", "").strip()
                            if clean_part:
                                # We can add operational data here if needed, or skip it
                                pass
                        
                        elif "NPS change:" in part or "touchpoints analyzed" in part or "Drivers:" in part:
                            clean_part = part.replace("🚚 Drivers:", "").replace("🚚", "").replace("Drivers:", "").strip()
                            if clean_part:
                                drivers_content = clean_part
            
            # Always show all three categories in consistent order
            ai_input_part += f"{indent}     • Routes: {routes_content}\n"
            ai_input_part += f"{indent}     • Verbatims: {verbatims_content}\n"
            ai_input_part += f"{indent}     • Explanatory Drivers: {drivers_content}\n"
        
        return ai_input_part
    
    # Build hierarchical structure based on segment filter
    if segment_filter == "Global":
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
    elif segment_filter == "Global/LH":
        # LH tree only
        ai_input += add_node_info("Global/LH", "Long Haul (LH)")
        ai_input += add_node_info("Global/LH/Economy", "├─ Economy", "  ")
        ai_input += add_node_info("Global/LH/Business", "├─ Business", "  ")
        ai_input += add_node_info("Global/LH/Premium", "└─ Premium", "  ")
    elif segment_filter == "Global/SH":
        # SH tree only
        ai_input += add_node_info("Global/SH", "Short Haul (SH)")
        ai_input += add_node_info("Global/SH/Economy", "├─ Economy", "  ")
        ai_input += add_node_info("Global/SH/Economy/IB", "  └─ IB", "    ")
        ai_input += add_node_info("Global/SH/Economy/YW", "  └─ YW", "    ")
        ai_input += add_node_info("Global/SH/Business", "└─ Business", "  ")
        ai_input += add_node_info("Global/SH/Business/IB", "  └─ IB", "    ")
        ai_input += add_node_info("Global/SH/Business/YW", "  └─ YW", "    ")
    elif segment_filter == "Global/SH/Economy":
        # SH Economy tree only
        ai_input += add_node_info("Global/SH/Economy", "SH Economy")
        ai_input += add_node_info("Global/SH/Economy/IB", "├─ IB", "  ")
        ai_input += add_node_info("Global/SH/Economy/YW", "└─ YW", "  ")
    elif segment_filter == "Global/SH/Business":
        # SH Business tree only
        ai_input += add_node_info("Global/SH/Business", "SH Business")
        ai_input += add_node_info("Global/SH/Business/IB", "├─ IB", "  ")
        ai_input += add_node_info("Global/SH/Business/YW", "└─ YW", "  ")
    else:
        # Single node
        node_name = segment_filter.split('/')[-1] if '/' in segment_filter else segment_filter
        ai_input += add_node_info(segment_filter, node_name)
    
    ai_input += "\nINTERPRETATION INSTRUCTIONS:\n"
    ai_input += "• Focus ONLY on segments marked as 'POSITIVE ANOMALY' or 'NEGATIVE ANOMALY'\n"
    ai_input += "• 'Normal' segments (even with deviations) are NOT anomalies - they are expected variations\n"
    ai_input += "• Explain the root causes using the analysis data provided for anomalous segments\n"
    if segment_filter != "Global":
        ai_input += f"• Analysis scope limited to {segment_filter} segment and its children\n"
    else:
        ai_input += "• If Global shows 'Normal' despite segment anomalies, this means the anomalies are localized and balanced out\n"
    
    return ai_input

async def print_enhanced_tree_with_explanations_and_interpretations(
    anomalies: dict, deviations: dict, explanations: dict, interpretations: dict,
    aggregation_days: int, target_period: int, date_range=None, segment_filter: str = "Global"):
    """Print enhanced anomaly tree with explanations and parent interpretations, filtered by segment"""
    
    # Create a more descriptive title with date range
    if date_range:
        start_date, end_date = date_range
        period_title = f"Period {target_period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    else:
        period_title = f"Period {target_period} ({aggregation_days}d aggregation)"
    
    print(f"\n📊 NPS Anomaly Analysis - {period_title}")
    print("-" * 70)
    
    # Filter anomalies to only include nodes that are part of the selected segment
    relevant_nodes = get_segment_node_paths(segment_filter)
    filtered_anomalies = {node: state for node, state in anomalies.items() if node in relevant_nodes}
    filtered_deviations = {node: dev for node, dev in deviations.items() if node in relevant_nodes}
    
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
    
    # DEBUG: Show what's in the filtered data
    debug_print(f"Filtered anomalies for segment {segment_filter} (normalized: {normalized_segment}):")
    debug_print(f"  Anomalies: {filtered_anomalies}")
    debug_print(f"  Deviations: {filtered_deviations}")
    debug_print(f"  Explanations available: {list(explanations.keys())}")
    debug_print(f"  Interpretations available: {list(interpretations.keys())}")
    
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
        """Get deviation text with context"""
        if node_path in filtered_deviations:
            deviation = filtered_deviations[node_path]
            if abs(deviation) < 10:
                return f"({deviation:+.1f} pts - within normal range)"
            else:
                return f"({deviation:+.1f} pts)"
        return ""
    
    def print_interpretation(node_path, indent=""):
        if node_path in interpretations:
            print(f"{indent}  └─ Pattern: {interpretations[node_path]}")
    
    def print_explanation(node_path, indent=""):
        print(f"{indent}  └─ ANALYSIS:")
        
        # Initialize with default values
        routes_content = "Not enough answers for statistical analysis"
        verbatims_content = "Not enough answers for statistical analysis"
        drivers_content = "Not enough answers for statistical analysis"
        
        if node_path in explanations:
            explanation = explanations[node_path]
            
            if explanation == "Analysis timeout":
                print(f"{indent}     • Analysis timeout occurred")
                return
            elif explanation and explanation.strip() != "":
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
                        clean_part = part.replace("🔧 Operational:", "").replace("🔧", "").strip()
                        if clean_part:
                            # We can add operational data here if needed, or skip it
                            pass
                    
                    elif "Drivers:" in part or "🚚" in part:
                        clean_part = part.replace("🚚 Drivers:", "").replace("🚚", "").replace("Drivers:", "").strip()
                        if clean_part:
                            drivers_content = clean_part
        
        # Always show all three categories in consistent order
        print(f"{indent}     • Routes: {routes_content}")
        print(f"{indent}     • Verbatims: {verbatims_content}")
        print(f"{indent}     • Explanatory Drivers: {drivers_content}")
    
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

async def run_flexible_data_download_silent_with_date(aggregation_days: int, periods: int, start_date, date_parameter: str, segment: str = "Global"):
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
    
    collector = PBIDataCollector()
    
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

async def run_comprehensive_analysis(analysis_date, date_parameter, segment: str = "Global"):
    """
    Run comprehensive analysis: Daily (last 7 days) + Weekly (current vs 3-week average) + Consolidated Summary
    """
    print("🚀 COMPREHENSIVE NPS ANALYSIS")
    print("=" * 80)
    print(f"📅 Analysis Date: {analysis_date.strftime('%Y-%m-%d')} ({date_parameter})")
    print(f"🎯 Segment Focus: {segment}")
    
    # Initialize Summary Agent for consolidated report
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType
        
        summary_agent = AnomalySummaryAgent(
            llm_type=LLMType.O3,
            logger=logging.getLogger("summary_agent")
        )
        summary_available = True
        print("📋 Summary Agent initialized for consolidated report")
    except Exception as e:
        print(f"⚠️ Summary Agent not available: {str(e)}")
        summary_available = False
    
    # Store all analysis results for consolidated summary
    consolidated_data = []
    
    try:
        # === DAILY ANALYSIS (Last 7 Days) ===
        print("\n" + "="*60)
        print("📅 DAILY ANALYSIS (Last 7 Days)")
        print("="*60)
        print("📥 Collecting daily data...")
        debug_print(f"Daily analysis parameters: 1 day aggregation, 7 periods, start_date={analysis_date}")
        daily_folder = await run_flexible_data_download_silent_with_date(
            aggregation_days=1,
            periods=7,
            start_date=analysis_date,
            date_parameter=f"{date_parameter}_daily",
            segment=segment
        )
        debug_print(f"Daily data collected in folder: {daily_folder}")
    
        daily_summary_data = []
        if daily_folder:
            print("🔍 Analyzing daily patterns...")
            debug_print(f"Running daily analysis on folder: {daily_folder}")
            daily_analysis = await run_flexible_analysis_silent(daily_folder, analysis_date)
            debug_print(f"Daily analysis results: {daily_analysis is not None}")
            # Analysis date is already included in the analysis
            
            if daily_analysis:
                anomaly_count = len(daily_analysis.get('anomaly_periods', []))
                debug_print(f"Daily anomaly periods: {daily_analysis['anomaly_periods']}")
                print(f"📊 Found anomalies in {anomaly_count} of 7 days")
                
                # Show daily trees and AI interpretations
                daily_summary_data = await show_silent_anomaly_analysis(daily_analysis, "DAILY", segment=segment)
                
                # Add to consolidated data
                if daily_summary_data:
                    consolidated_data.append({
                        'analysis_type': 'DIARIO',
                        'periods': 7,
                        'summary': daily_summary_data
                    })
        
        # === WEEKLY ANALYSIS (Current Week vs 3-Week Average) ===
        print("\n" + "="*60)
        print("📅 WEEKLY ANALYSIS (Current Week vs 3-Week Average)")
        print("="*60)
        print("📥 Collecting weekly data...")
        
        try:
            # Add timeout for weekly data collection
            debug_print("Weekly collection parameters:")
            debug_print(f"   • aggregation_days: 7")
            debug_print(f"   • periods: 4")
            debug_print(f"   • start_date: {analysis_date}")
            debug_print(f"   • date_parameter: {date_parameter}_weekly")
            debug_print(f"   • segment: {segment}")
            
            weekly_folder = await asyncio.wait_for(
                run_flexible_data_download_silent_with_date(
                    aggregation_days=7,
                    periods=4,  # Current week + 3 weeks for average
                    start_date=analysis_date,
                    date_parameter=f"{date_parameter}_weekly",
                    segment=segment
                ),
                timeout=120.0  # 2 minute timeout for weekly data collection
            )
        except asyncio.TimeoutError:
            print("⏰ Weekly data collection timed out (>2 minutes)")
            weekly_folder = None
        except Exception as e:
            print(f"❌ Weekly data collection failed: {str(e)}")
            weekly_folder = None
        
        weekly_summary_data = []
        if weekly_folder:
            print("🔍 Analyzing weekly patterns...")
            debug_print(f"Running weekly analysis on folder: {weekly_folder}")
            try:
                # Add timeout for weekly analysis
                weekly_analysis = await asyncio.wait_for(
                    run_weekly_current_vs_average_analysis_silent(weekly_folder),
                    timeout=60.0  # 1 minute timeout for weekly analysis
                )
                debug_print(f"Weekly analysis results: {weekly_analysis is not None}")
                # Add analysis date to weekly analysis
                if weekly_analysis:
                    weekly_analysis['analysis_date'] = analysis_date
            except asyncio.TimeoutError:
                print("⏰ Weekly analysis timed out (>1 minute)")
                weekly_analysis = None
            except Exception as e:
                print(f"❌ Weekly analysis failed: {str(e)}")
                weekly_analysis = None
        
            if weekly_analysis:
                anomaly_count = len(weekly_analysis.get('anomaly_periods', []))
                if anomaly_count > 0:
                    print(f"📊 Found anomalies in current week")
                else:
                    print(f"📊 Current week shows normal patterns")
                
                # Show the date range for weekly analysis
                detector = weekly_analysis['detector']
                data_folder = weekly_analysis['data_folder']
                aggregation_days = weekly_analysis['aggregation_days']
                
                # Initialize interpreter to get date range
                from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter
                from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
                
                pbi_collector = PBIDataCollector()
                interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector)
                
                # Get date range for period 1 (current week)
                date_range = interpreter._get_period_date_range(1, aggregation_days)
                if date_range:
                    start_date, end_date = date_range
                    print(f"📅 Weekly period date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (7 days)")
                    print(f"📅 Analysis date setting: {analysis_date.strftime('%Y-%m-%d')}")
                    print(f"📅 Date parameter: {date_parameter}")
                else:
                    print(f"⚠️ Could not determine weekly date range")
                
                try:
                    # ALWAYS show weekly trees and AI interpretations (whether anomalies or not)
                    weekly_summary_data = await asyncio.wait_for(
                        show_silent_anomaly_analysis(weekly_analysis, "WEEKLY", show_all_periods=True, segment=segment),
                        timeout=180.0  # 3 minute timeout for weekly AI analysis
                    )
                    
                    # Add to consolidated data
                    if weekly_summary_data:
                        consolidated_data.append({
                            'analysis_type': 'SEMANAL',
                            'periods': 1,
                            'summary': weekly_summary_data
                        })
                except asyncio.TimeoutError:
                    print("⏰ Weekly AI analysis timed out (>3 minutes)")
                    print("   Continuing without weekly summary...")
                except Exception as e:
                    print(f"❌ Weekly AI analysis failed: {str(e)}")
                    print("   Continuing without weekly summary...")
            else:
                print("⚠️ Weekly analysis failed")
        else:
            print("❌ Weekly data collection failed")
        
        # === CONSOLIDATED SUMMARY ===
        if summary_available and consolidated_data:
            print("\n" + "="*80)
            print("📋 CONSOLIDATED EXECUTIVE SUMMARY")
            print("="*80)
            
            try:
                print("🤖 Generating comprehensive summary across daily and weekly analyses...")
                consolidated_summary = await generate_consolidated_summary(summary_agent, consolidated_data)
                
                print(f"\n{consolidated_summary}")
                
                # Performance metrics
                metrics = summary_agent.get_performance_metrics()
                print(f"\n📊 Summary Generation Metrics:")
                print(f"   • Input tokens: {metrics.get('input_tokens', 0)}")
                print(f"   • Output tokens: {metrics.get('output_tokens', 0)}")
                print(f"   • LLM: {metrics.get('llm_type', 'Unknown')}")
            
            except Exception as e:
                print(f"❌ Consolidated summary generation failed: {str(e)}")
                print(f"   Manual review recommended")
        elif not consolidated_data:
            print("\n📋 CONSOLIDATED SUMMARY:")
            print("No anomalies detected in either daily or weekly analysis.")
            print("All segments are operating within normal NPS variation ranges.")
        elif not summary_available:
            print("\n⚠️ Consolidated summary not available (Summary Agent initialization failed)")
            if consolidated_data:
                total_periods = sum(data['periods'] for data in consolidated_data)
                print(f"   Individual analyses completed for {len(consolidated_data)} timeframes with {total_periods} total periods")
        
        print(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETED")
        print(f"📅 Based on data from: {analysis_date.strftime('%Y-%m-%d')} ({date_parameter})")
        print(f"🎯 Segment analyzed: {segment}")
        
    except Exception as e:
        print(f"❌ Error during comprehensive analysis: {str(e)}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")

async def run_clean_daily_analysis(analysis_date, date_parameter, segment: str = "Global"):
    """
    Run clean daily analysis: Last 7 days with only AI summaries and trees
    """
    print("🚀 CLEAN DAILY NPS ANALYSIS")
    print("=" * 60)
    print(f"📅 Analysis Date: {analysis_date.strftime('%Y-%m-%d')} ({date_parameter})")
    print(f"🎯 Segment Focus: {segment}")
    
    # Download daily data silently
    daily_folder = await run_flexible_data_download_silent_with_date(
        aggregation_days=1,
        periods=7,
        start_date=analysis_date,
        date_parameter=f"{date_parameter}_daily_clean",
        segment=segment
    )
    
    if not daily_folder:
        print("❌ Data collection failed")
        return
    
    # Run analysis silently with the correct analysis date
    daily_analysis = await run_flexible_analysis_silent(daily_folder, analysis_date)
    
    if not daily_analysis:
        print("❌ Analysis failed")
        return
    
    # Show only trees and AI summaries
    await show_clean_anomaly_analysis(daily_analysis, segment)
    
    print(f"\n🎯 CLEAN DAILY ANALYSIS COMPLETED")
    print(f"📅 Based on data from: {analysis_date.strftime('%Y-%m-%d')} ({date_parameter})")
    print(f"🎯 Segment analyzed: {segment}")

async def run_flexible_analysis_silent(data_folder: str, analysis_date: datetime = None):
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
    
    detector = FlexibleAnomalyDetector(
        aggregation_days=aggregation_days,
        threshold=10.0,
        min_sample_size=5
    )
    
    # Calculate the correct period numbers if analysis_date is provided
    if analysis_date:
        # Calculate the period number for the analysis date (most recent period)
        base_period = calculate_actual_period_number(analysis_date)
        # Analyze 7 periods starting from the analysis date
        periods_to_analyze = list(range(base_period, base_period + 7))
    else:
        # Default behavior: analyze the 7 most recent periods (1-7)
        periods_to_analyze = list(range(1, 8))  # Periods 1, 2, 3, 4, 5, 6, 7
    
    anomaly_periods = []
    
    # Suppress all output during analysis
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                for period in periods_to_analyze:
                    period_anomalies, period_deviations, period_explanations = detector.analyze_period(data_folder, period)
                    
                    # Check if any node has an anomaly
                    has_anomaly = any(state in ['+', '-'] for state in period_anomalies.values())
                    if has_anomaly:
                        anomaly_periods.append(period)
            
            except Exception:
                return None
    
    return {
        'detector': detector,
        'data_folder': data_folder,
        'aggregation_days': aggregation_days,
        'anomaly_periods': anomaly_periods,
        'total_periods': 7,
        'periods_analyzed': periods_to_analyze,
        'analysis_date': analysis_date
    }

async def run_flexible_analysis(data_folder: str):
    """Run flexible period analysis and return results"""
    print(f"🔄 Analyzing periods in: {data_folder}")
    
    # Extract aggregation days from folder name
    folder_name = Path(data_folder).name
    if 'flexible_' in folder_name and 'd_' in folder_name:
        try:
            aggregation_days = int(folder_name.split('flexible_')[1].split('d_')[0])
        except:
            aggregation_days = 7  # Default
    else:
        aggregation_days = 7
    
    detector = FlexibleAnomalyDetector(
        aggregation_days=aggregation_days,
        threshold=10.0,
        min_sample_size=5
    )
    
    # Analyze the 7 most recent periods (periods 1-7)
    print("🔍 Analyzing the 7 most recent periods...")
    periods_to_analyze = list(range(1, 8))  # Periods 1, 2, 3, 4, 5, 6, 7
    anomaly_periods = []
    
    try:
        for period in periods_to_analyze:
            period_anomalies, period_deviations, period_explanations = detector.analyze_period(data_folder, period)
            
            # Check if any node has an anomaly
            has_anomaly = any(state in ['+', '-'] for state in period_anomalies.values())
            if has_anomaly:
                anomaly_periods.append(period)
    
    except Exception as e:
        print(f"⚠️ Analysis stopped: {str(e)}")
    
    if anomaly_periods:
        print(f"🚨 Found anomalies in {len(anomaly_periods)} of 7 periods: {anomaly_periods}")
        return {
            'detector': detector,
            'data_folder': data_folder,
            'aggregation_days': aggregation_days,
            'anomaly_periods': anomaly_periods,
            'total_periods': 7,
            'periods_analyzed': periods_to_analyze,
            'analysis_date': None  # Will be set when available
        }
    else:
        print("✅ No anomalies detected in the 7 most recent periods")
        return {
            'detector': detector,
            'data_folder': data_folder,
            'aggregation_days': aggregation_days,
            'anomaly_periods': [],
            'total_periods': 7,
            'periods_analyzed': periods_to_analyze,
            'analysis_date': None  # Will be set when available
        }

async def run_weekly_current_vs_average_analysis_silent(data_folder: str):
    """Run weekly analysis silently focusing only on current week (period 1) vs 3-week average"""
    import os
    from contextlib import redirect_stdout, redirect_stderr
    
    # Extract aggregation days from folder name (should be 7 for weekly)
    folder_name = Path(data_folder).name
    if 'flexible_' in folder_name and 'd_' in folder_name:
        try:
            aggregation_days = int(folder_name.split('flexible_')[1].split('d_')[0])
        except:
            aggregation_days = 7  # Default for weekly
    else:
        aggregation_days = 7
    
    detector = FlexibleAnomalyDetector(
        aggregation_days=aggregation_days,
        threshold=10.0,
        min_sample_size=5
    )
    
    # Analyze only period 1 (current week vs 3-week moving average) silently
    current_week_period = 1
    anomaly_periods = []
    
    # Suppress all output during analysis
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                # Analyze only the current week (period 1)
                period_anomalies, period_deviations, period_explanations = detector.analyze_period(data_folder, current_week_period)
                
                # Check if current week has any anomaly
                has_anomaly = any(state in ['+', '-'] for state in period_anomalies.values())
                if has_anomaly:
                    anomaly_periods.append(current_week_period)
            
            except Exception:
                return None
    
    return {
        'detector': detector,
        'data_folder': data_folder,
        'aggregation_days': aggregation_days,
        'anomaly_periods': anomaly_periods,
        'total_periods': 1,  # Only analyzing current week
        'periods_analyzed': [current_week_period],
        'analysis_date': None  # Will be set when available
    }

async def show_silent_anomaly_analysis(analysis_data: dict, analysis_type: str, show_all_periods=False, segment: str = "Global"):
    """Show only trees and AI summaries for periods with anomalies - silent version"""
    import os
    from contextlib import redirect_stdout, redirect_stderr
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    aggregation_days = analysis_data['aggregation_days']
    anomaly_periods = analysis_data['anomaly_periods']
    periods_analyzed = analysis_data.get('periods_analyzed', anomaly_periods)
    
    # Initialize interpreter for explanations
    pbi_collector = PBIDataCollector()
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector)
    
    # Initialize AI agent for interpretation
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=LLMType.O3,
            logger=logging.getLogger("ai_interpreter")
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
        period_anomalies, period_deviations, _ = detector.analyze_period(data_folder, period)
        
        # Get date range
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
        
        if nodes_with_anomalies:
            # Suppress all output during explanation collection
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    for node_path in nodes_with_anomalies:
                        try:
                            anomaly_state = period_anomalies.get(node_path, "?")
                            
                            # Calculate correct date range if analysis_date is available
                            start_date, end_date = None, None
                            analysis_date = analysis_data.get('analysis_date')
                            if analysis_date:
                                start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
                            
                            explanation = await asyncio.wait_for(
                                interpreter.explain_anomaly(
                                    node_path=node_path,
                                    target_period=period,
                                    aggregation_days=aggregation_days,
                                    anomaly_state=anomaly_state,
                                    start_date=start_date,
                                    end_date=end_date
                                ),
                                timeout=30.0
                            )
                            explanations[node_path] = explanation
                        except Exception:
                            explanations[node_path] = "Analysis timeout"
        
        # Show the tree
        await print_enhanced_tree_with_explanations_and_interpretations(
            period_anomalies, period_deviations, explanations, parent_interpretations,
            aggregation_days, period, date_range, segment
        )
        
        # AI Interpretation
        ai_interpretation = None
        if ai_available:
            print(f"\n🤖 AI INTERPRETATION:")
            print("-" * 40)
            
            try:
                # Build AI input (works for both anomalous and normal periods)
                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                                 parent_interpretations, explanations, date_range, segment)
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, 
                                                   start_date.strftime('%Y-%m-%d') if date_range else None),
                    timeout=45.0
                )
                
                print(ai_interpretation)
                
            except Exception as e:
                ai_interpretation = f"AI interpretation failed: {str(e)}"
                print(ai_interpretation)
        
        # Collect period data for summary
        period_data = {
            'period': period,
            'date_range': date_range_str,
            'ai_interpretation': ai_interpretation or "No AI interpretation available"
        }
        all_periods_data.append(period_data)
    
    return all_periods_data

async def show_clean_anomaly_analysis(analysis_data: dict, segment: str = "Global"):
    """Show only trees and AI summaries for periods with anomalies"""
    import os
    from contextlib import redirect_stdout, redirect_stderr
    
    detector = analysis_data['detector']
    data_folder = analysis_data['data_folder']
    aggregation_days = analysis_data['aggregation_days']
    anomaly_periods = analysis_data['anomaly_periods']
    periods_analyzed = analysis_data.get('periods_analyzed', anomaly_periods)
    
    # Initialize interpreter for explanations
    pbi_collector = PBIDataCollector()
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector)
    
    # Initialize AI agent for interpretation
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=LLMType.O3,
            logger=logging.getLogger("ai_interpreter")
        )
        ai_available = True
    except Exception:
        ai_available = False
    
    # Show only periods with anomalies
    periods_with_anomalies = [p for p in periods_analyzed if p in anomaly_periods]
    
    # Show detailed analysis for periods with anomalies
    for period in periods_with_anomalies:
        print(f"\n{'='*60}")
        print(f"PERIOD {period} ANALYSIS")
        print("="*60)
        
        # Get anomalies for this period
        period_anomalies, period_deviations, _ = detector.analyze_period(data_folder, period)
        
        # Get date range
        date_range = interpreter._get_period_date_range(period, aggregation_days)
        if date_range:
            start_date, end_date = date_range
            print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            date_range_str = "Unknown dates"
        
        # Generate parent interpretations
        parent_interpretations = generate_parent_interpretations(period_anomalies)
        
        # Collect explanations for anomalous nodes
        explanations = {}
        nodes_with_anomalies = [node for node, state in period_anomalies.items() if state in ['+', '-']]
        
        if nodes_with_anomalies:
            # Suppress all output during explanation collection
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    for node_path in nodes_with_anomalies:
                        try:
                            anomaly_state = period_anomalies.get(node_path, "?")
                            
                            # Calculate correct date range if analysis_date is available
                            start_date, end_date = None, None
                            analysis_date = analysis_data.get('analysis_date')
                            if analysis_date:
                                start_date, end_date = calculate_period_date_range(analysis_date, period, aggregation_days)
                            
                            explanation = await asyncio.wait_for(
                                interpreter.explain_anomaly(
                                    node_path=node_path,
                                    target_period=period,
                                    aggregation_days=aggregation_days,
                                    anomaly_state=anomaly_state,
                                    start_date=start_date,
                                    end_date=end_date
                                ),
                                timeout=30.0
                            )
                            explanations[node_path] = explanation
                        except Exception:
                            explanations[node_path] = "Analysis timeout"
        
        # Show the tree
        await print_enhanced_tree_with_explanations_and_interpretations(
            period_anomalies, period_deviations, explanations, parent_interpretations,
            aggregation_days, period, date_range, segment
        )
        
        # AI Interpretation
        ai_interpretation = None
        if ai_available and nodes_with_anomalies:
            print(f"\n🤖 AI INTERPRETATION:")
            print("-" * 40)
            
            try:
                # Build AI input
                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                                 parent_interpretations, explanations, date_range, segment)
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, 
                                                   start_date.strftime('%Y-%m-%d') if date_range else None),
                    timeout=45.0
                )
                
                print(ai_interpretation)
                
            except Exception as e:
                ai_interpretation = f"AI interpretation failed: {str(e)}"
                print(ai_interpretation)

async def generate_consolidated_summary(agent, consolidated_data: List[Dict]) -> str:
    """Generate a consolidated summary from multiple analysis types."""
    # Format the consolidated input
    formatted_sections = []
    
    for data in consolidated_data:
        analysis_type = data['analysis_type']
        periods_count = data['periods']
        summary_data = data['summary']
        
        section = f"\n=== ANÁLISIS {analysis_type} ===\n"
        section += f"Períodos analizados: {periods_count}\n\n"
        
        for period_data in summary_data:
            period = period_data['period']
            date_range = period_data['date_range']
            interpretation = period_data['ai_interpretation']
            
            section += f"Período {period} ({date_range}):\n{interpretation}\n\n"
        
        formatted_sections.append(section)
    
    # Create consolidated input
    consolidated_input = "\n".join(formatted_sections)
    
    # Use a modified prompt for consolidated analysis
    message_history = agent._get_message_history_for_consolidated(consolidated_input)
    
    response, _, _ = await agent.agent.invoke(messages=message_history.get_messages())
    return response.content.strip()

async def main():
    """Enhanced main entry point for comprehensive anomaly analysis"""
    print("🚀 Enhanced Flexible NPS Anomaly Detection System")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Flexible NPS Anomaly Detection')
    parser.add_argument('--mode', choices=['download', 'analyze', 'both', 'comprehensive', 'clean-daily'], default='both',
                       help='Mode: download data, analyze existing data, both, comprehensive (daily + weekly), or clean-daily (only AI summaries)')
    parser.add_argument('--folder', type=str, 
                       help='Specific folder to analyze (e.g., tables/available_2025_06_04)')
    parser.add_argument('--aggregation-days', type=int, default=7,
                       help='Number of days per aggregation period (default: 7)')
    parser.add_argument('--periods', type=int, default=74,
                       help='Number of periods to analyze (default: 74)')
    
    # New date-related parameters - clearer logic
    parser.add_argument('--insert-date-ci', type=str,
                       help='Simulate today being this date (YYYY-MM-DD). Data available until this date - 4 days')
    parser.add_argument('--date-flight-local', type=str,
                       help='Use this date directly as available in dashboard (YYYY-MM-DD)')
    
    # New segment parameter for focused analysis
    parser.add_argument('--segment', type=str, default='Global',
                       help='Root segment to analyze (e.g., Global, SH, Global/SH/Economy, Global/LH). Default: Global (full tree)')
    
    # Debug mode parameter
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose print statements')
    
    args = parser.parse_args()
    
    # Set global debug mode based on flag
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    if DEBUG_MODE:
        print("🔍 DEBUG MODE ENABLED - Verbose output activated")
    
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
            analysis_date = simulated_today - timedelta(days=pbi_lag_days)  # Simulate 4-day lag
            date_parameter = 'insert_ci'
            date_description = f"Simulating today as {simulated_today.strftime('%Y-%m-%d')}"
        except ValueError:
            print("❌ Error: --insert-date-ci must be in YYYY-MM-DD format")
            return
    elif args.date_flight_local:
        try:
            analysis_date = datetime.strptime(args.date_flight_local, '%Y-%m-%d').date()
            date_parameter = 'flight_local'
            date_description = f"Using direct dashboard date"
        except ValueError:
            print("❌ Error: --date-flight-local must be in YYYY-MM-DD format")
            return
    else:
        # Default behavior: use available date (today - 4 days)
        analysis_date = today - timedelta(days=pbi_lag_days)
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
    
    print(f"\n🎯 SEGMENT CONFIGURATION:")
    print(f"   • Selected segment: {args.segment}")
    if args.segment != "Global":
        segment_nodes = get_segment_node_paths(args.segment)
        print(f"   • Nodes to analyze: {len(segment_nodes)} ({', '.join(segment_nodes)})")
    else:
        print(f"   • Analyzing full hierarchy tree (all segments)")
    
    try:
        if args.mode == 'clean-daily':
            # Run clean daily analysis with specified date
            await run_clean_daily_analysis(analysis_date, date_parameter, args.segment)
            return
        
        if args.mode == 'comprehensive':
            # Run comprehensive analysis with specified date
            await run_comprehensive_analysis(analysis_date, date_parameter, args.segment)
            return
        
        if args.mode in ['download', 'both']:
            # Download data
            print("\n📥 STEP 1: Data Collection")
            print("-" * 40)
            
            total_days = args.aggregation_days * args.periods
            
            print(f"⚙️ Configuration:")
            print(f"   • Aggregation: {args.aggregation_days} days per period") 
            print(f"   • Total periods: {args.periods}")
            print(f"   • Analysis span: {total_days} days")
            
            await run_flexible_data_download_with_date(
                aggregation_days=args.aggregation_days,
                periods=args.periods,
                start_date=analysis_date,
                date_parameter=date_parameter,
                segment=args.segment
            )
        
        # Determine analysis folder
        if args.folder:
            analysis_folder = args.folder
        else:
            # Use the most recent folder with the date parameter
            tables_dir = Path("tables")
            if not tables_dir.exists():
                print("❌ No tables directory found. Run download first.")
                return
                
            # Look for folders with the current date parameter and aggregation days
            date_str = analysis_date.strftime('%Y_%m_%d')
            folder_pattern = f'{date_parameter}_{date_str}_flexible_{args.aggregation_days}d'
            
            matching_folders = [f for f in tables_dir.iterdir() 
                                if f.is_dir() and folder_pattern in f.name]
            
            if not matching_folders:
                print(f"❌ No folders found matching pattern '{folder_pattern}'. Run download first.")
                return
                
            # Get most recent folder
            analysis_folder = str(max(matching_folders, key=lambda x: x.stat().st_mtime))
        
        # Analyze data
        if args.mode in ['analyze', 'both']:
            print(f"\n🔍 STEP 2: Anomaly Detection & Analysis")
            print("-" * 40)
            print(f"📁 Analysis folder: {analysis_folder}")
            
            analysis_data = await run_flexible_analysis(analysis_folder)
            
            if analysis_data and analysis_data.get('anomaly_periods'):
                # Show trees with explanations
                await show_all_anomaly_periods_with_explanations(analysis_data, args.segment)
                
                # Final summary
                anomaly_periods = analysis_data['anomaly_periods']
                aggregation_days = analysis_data['aggregation_days']
                total_periods = analysis_data.get('total_periods', 74)
                
                print(f"\n🎯 FINAL SUMMARY:")
                print(f"   📊 Total periods analyzed: 7")
                print(f"   🚨 Periods with anomalies: {len(anomaly_periods)}")
                print(f"   📈 Aggregation: {aggregation_days} days")
                print(f"   📅 Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
                print(f"   💾 Data saved in: {analysis_folder}")
                if anomaly_periods:
                    print(f"   🔍 Anomaly periods: {anomaly_periods}")
                else:
                    print(f"   ✅ All periods normal")
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
    print_interpretation("Global/SH", "")
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
    node_state = anomalies.get(node_path, "?")
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
    # Calculate how many days back from analysis date
    days_back = (target_period - 1) * aggregation_days
    
    # For daily analysis, each period is exactly one day
    # For weekly analysis, each period is 7 days, etc.
    period_end = analysis_date - timedelta(days=days_back)
    period_start = period_end - timedelta(days=aggregation_days - 1)
    
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

if __name__ == "__main__":
    asyncio.run(main())
