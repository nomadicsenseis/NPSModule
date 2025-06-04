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

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_detection.flexible_detector import FlexibleAnomalyDetector
from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter

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

async def show_all_anomaly_periods_with_explanations(analysis_data: dict):
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
        from dashboard_analyzer.anomaly_explanation.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=LLMType.GPT4o_MINI,
            logger=logging.getLogger("ai_interpreter")
        )
        ai_available = True
        print("🤖 AI Agent initialized for interpretations")
    except Exception as e:
        print(f"⚠️ AI Agent not available: {str(e)}")
        ai_available = False
    
    # Initialize Summary Agent for final report
    try:
        from dashboard_analyzer.anomaly_explanation.anomaly_summary_agent import AnomalySummaryAgent
        
        summary_agent = AnomalySummaryAgent(
            llm_type=LLMType.GPT4o_MINI,
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
            print("🔍 Collecting explanations...")
            for node_path in nodes_with_anomalies:
                try:
                    anomaly_state = period_anomalies.get(node_path, "?")
                    explanation = await asyncio.wait_for(
                        interpreter.explain_anomaly(
                            node_path=node_path,
                            target_period=period,
                            aggregation_days=aggregation_days,
                            anomaly_state=anomaly_state
                        ),
                        timeout=30.0
                    )
                    explanations[node_path] = explanation
                except Exception:
                    explanations[node_path] = "Analysis timeout"
        
        # Show the enhanced tree with explanations and parent interpretations
        await print_enhanced_tree_with_explanations_and_interpretations(
            period_anomalies, period_deviations, explanations, parent_interpretations,
            aggregation_days, period, date_range
        )
        
        # AI Interpretation (ensure it completes)
        ai_interpretation = None
        if ai_available and nodes_with_anomalies:
            print(f"\n🤖 AI INTERPRETATION:")
            print("-" * 40)
            
            try:
                # Build comprehensive input for AI
                ai_input = build_ai_input_string(period, period_anomalies, period_deviations, 
                                               parent_interpretations, explanations, date_range)
                
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, 
                                                   start_date.strftime('%Y-%m-%d') if date_range else None),
                    timeout=45.0
                )
                
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
    
    # Global interpretation
    lh_state = anomalies.get("Global/LH", "?")
    sh_state = anomalies.get("Global/SH", "?")
    global_state = anomalies.get("Global", "?")
    
    if lh_state != "?" and sh_state != "?":
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
    
    if economy_state != "?" and business_state != "?" and sh_state == "N":
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
    
    # SH/Economy interpretation
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
    
    # SH/Business interpretation
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
                         interpretations: dict, explanations: dict, date_range: tuple) -> str:
    """Build comprehensive input string for AI interpretation"""
    
    if date_range:
        start_date, end_date = date_range
        ai_input = f"NPS ANOMALY ANALYSIS - PERIOD {period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})\n\n"
    else:
        ai_input = f"NPS ANOMALY ANALYSIS - PERIOD {period}\n\n"
    
    # Count actual anomalies vs normal variations
    actual_anomalies = [node for node, state in anomalies.items() if state in ['+', '-']]
    normal_segments = [node for node, state in anomalies.items() if state == 'N']
    
    ai_input += f"ANOMALY SUMMARY:\n"
    ai_input += f"• Total segments analyzed: {len(anomalies)}\n"
    ai_input += f"• Actual anomalies detected: {len(actual_anomalies)}\n"
    ai_input += f"• Normal variations: {len(normal_segments)}\n\n"
    
    if actual_anomalies:
        ai_input += f"SEGMENTS WITH ACTUAL ANOMALIES:\n"
        for node_path in actual_anomalies:
            state = anomalies[node_path]
            deviation = deviations.get(node_path, 0)
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
    
    # Build hierarchical structure with interpretations
    def add_node_info(node_path: str, name: str, indent: str = ""):
        state = anomalies.get(node_path, "?")
        deviation = deviations.get(node_path, 0)
        
        ai_input_part = f"{indent}{name}: {get_state_desc(state, deviation)}\n"
        
        # Add interpretation if available
        if node_path in interpretations:
            ai_input_part += f"{indent}  └─ Pattern: {interpretations[node_path]}\n"
        
        # Add explanation if available and is actual anomaly
        if node_path in explanations and state in ['+', '-']:
            explanation = explanations[node_path]
            ai_input_part += f"{indent}  └─ Analysis:\n"
            # Clean up explanation for AI
            if " | " in explanation:
                parts = explanation.split(" | ")[1:]  # Skip period description
                for part in parts:
                    if part.strip():
                        clean_part = part.strip()
                        # Remove emoji prefixes for cleaner AI input
                        clean_part = clean_part.replace("💬", "").replace("🛣️", "").replace("🔧", "").replace("🚚", "")
                        clean_part = clean_part.replace("Customer feedback:", "Verbatims:").replace("Routes:", "Routes:").replace("Operational:", "Operations:").replace("Drivers:", "Key Drivers:")
                        ai_input_part += f"{indent}     • {clean_part.strip()}\n"
        
        return ai_input_part
    
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
    
    ai_input += "\nINTERPRETATION INSTRUCTIONS:\n"
    ai_input += "• Focus ONLY on segments marked as 'POSITIVE ANOMALY' or 'NEGATIVE ANOMALY'\n"
    ai_input += "• 'Normal' segments (even with deviations) are NOT anomalies - they are expected variations\n"
    ai_input += "• Explain the root causes using the analysis data provided for anomalous segments\n"
    ai_input += "• If Global shows 'Normal' despite segment anomalies, this means the anomalies are localized and balanced out\n"
    
    return ai_input

async def print_enhanced_tree_with_explanations_and_interpretations(
    anomalies: dict, deviations: dict, explanations: dict, interpretations: dict,
    aggregation_days: int, target_period: int, date_range=None):
    """Print enhanced anomaly tree with explanations and parent interpretations"""
    
    # Create a more descriptive title with date range
    if date_range:
        start_date, end_date = date_range
        period_title = f"Period {target_period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    else:
        period_title = f"Period {target_period} ({aggregation_days}d aggregation)"
    
    print(f"\n📊 NPS Anomaly Analysis - {period_title}")
    print("-" * 70)
    
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
        if node_path in deviations:
            deviation = deviations[node_path]
            if abs(deviation) < 10:
                return f"({deviation:+.1f} pts - within normal range)"
            else:
                return f"({deviation:+.1f} pts)"
        return ""
    
    def print_interpretation(node_path, indent=""):
        if node_path in interpretations:
            print(f"{indent}  └─ Pattern: {interpretations[node_path]}")
    
    def print_explanation(node_path, indent=""):
        if node_path in explanations:
            explanation = explanations[node_path]
            print(f"{indent}  └─ ANALYSIS:")
            
            # Debug: Show raw explanation
            if explanation == "Analysis timeout":
                print(f"{indent}     • Analysis timeout occurred")
                return
            elif not explanation or explanation.strip() == "":
                print(f"{indent}     • No explanation data available")
                return
            
            # Split explanation into components and clean them up
            parts = explanation.split(" | ")
            
            displayed_count = 0
            for part in parts:
                part = part.strip()
                if not part or part.startswith("Period"):
                    continue
                
                # Clean up and format different explanation types
                if part.startswith("💬") and "Customer feedback:" in part:
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
                    
                    print(f"{indent}     • Verbatims: {count}{sentiment}{topics}")
                    displayed_count += 1
                
                elif part.startswith("🛣️") and ("routes analyzed" in part or "Routes:" in part):
                    clean_part = part.replace("🛣️ Routes: 🛣️ Routes:", "Routes:")
                    clean_part = clean_part.replace("🛣️ 🛣️", "").replace("🛣️", "").strip()
                    
                    if clean_part and "routes analyzed" in clean_part:
                        print(f"{indent}     • Routes: {clean_part}")
                        displayed_count += 1
                
                elif part.startswith("🔧"):
                    clean_part = part.replace("🔧 Operational:", "").strip()
                    if clean_part:
                        print(f"{indent}     • Operations: {clean_part}")
                        displayed_count += 1
                
                elif part.startswith("🚚") and ("NPS change:" in part or "touchpoints analyzed" in part):
                    clean_part = part.replace("🚚 Drivers: 🚚 Drivers:", "").replace("🚚 Drivers:", "").replace("🚚 🚚", "").replace("🚚", "").strip()
                    
                    if clean_part:
                        print(f"{indent}     • Key Drivers: {clean_part}")
                        displayed_count += 1
            
            # If no parts were displayed, show debug info
            if displayed_count == 0:
                print(f"{indent}     • Debug: Raw explanation = '{explanation[:100]}...'")
                print(f"{indent}     • Debug: Parts found = {len(parts)}")
                for i, part in enumerate(parts[:3]):  # Show first 3 parts
                    print(f"{indent}       Part {i}: '{part[:50]}...'")
        else:
            print(f"{indent}  └─ ANALYSIS:")
            print(f"{indent}     • No explanation available for {node_path}")
    
    # Count actual anomalies (not normal deviations)
    actual_anomalies = [node for node, state in anomalies.items() if state in ['+', '-']]
    
    # Global summary
    global_state = anomalies.get("Global", "?")
    global_dev = get_deviation_text("Global")
    global_desc = get_state_description(global_state)
    
    print(f"Global: {global_desc} {global_dev}")
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
    
    # Summary
    print(f"\n📋 SUMMARY:")
    if actual_anomalies:
        print(f"  • Anomalies detected in: {', '.join(actual_anomalies)}")
        print(f"  • Total anomalous segments: {len(actual_anomalies)}")
    else:
        print(f"  • No anomalies detected - all segments within normal variation")

async def run_flexible_data_download(aggregation_days: int, periods: int, start_date):
    """Run flexible data download"""
    # Generate folder name
    current_date = datetime.now()
    date_str = current_date.strftime('%d_%m_%Y')
    target_folder = f"tables/flexible_{aggregation_days}d_{date_str}"
    
    print(f"📁 Target folder: {target_folder}")
    
    success = await collect_flexible_data(aggregation_days, target_folder)
    
    if success:
        print(f"✅ Data collection completed successfully")
        return target_folder
    else:
        print(f"❌ Data collection failed")
        return None

async def main():
    """Enhanced main entry point for comprehensive anomaly analysis"""
    print("🚀 Enhanced Flexible NPS Anomaly Detection System")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Flexible NPS Anomaly Detection')
    parser.add_argument('--mode', choices=['download', 'analyze', 'both'], default='both',
                       help='Mode: download data, analyze existing data, or both')
    parser.add_argument('--folder', type=str, 
                       help='Specific folder to analyze (e.g., tables/flexible_7d_04_06_2025)')
    parser.add_argument('--aggregation-days', type=int, default=7,
                       help='Number of days per aggregation period (default: 7)')
    parser.add_argument('--periods', type=int, default=74,
                       help='Number of periods to analyze (default: 74)')
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['download', 'both']:
            # Download data
            print("\n📥 STEP 1: Data Collection")
            print("-" * 40)
            
            start_date = datetime.now().date()
            total_days = args.aggregation_days * args.periods
            
            print(f"⚙️ Configuration:")
            print(f"   • Aggregation: {args.aggregation_days} days per period") 
            print(f"   • Total periods: {args.periods}")
            print(f"   • Analysis span: {total_days} days")
            
            await run_flexible_data_download(
                aggregation_days=args.aggregation_days,
                periods=args.periods,
                start_date=start_date
            )
        
        # Determine analysis folder
        if args.folder:
            analysis_folder = args.folder
        else:
            # Use the most recent folder
            tables_dir = Path("tables")
            if not tables_dir.exists():
                print("❌ No tables directory found. Run download first.")
                return
                
            flexible_folders = [f for f in tables_dir.iterdir() 
                              if f.is_dir() and f.name.startswith(f'flexible_{args.aggregation_days}d_')]
            if not flexible_folders:
                print(f"❌ No flexible {args.aggregation_days}d folders found. Run download first.")
                return
                
            # Get most recent folder
            analysis_folder = str(max(flexible_folders, key=lambda x: x.stat().st_mtime))
        
        # Analyze data
        if args.mode in ['analyze', 'both']:
            print(f"\n🔍 STEP 2: Anomaly Detection & Analysis")
            print("-" * 40)
            print(f"📁 Analysis folder: {analysis_folder}")
            
            analysis_data = await run_flexible_analysis(analysis_folder)
            
            if analysis_data and analysis_data.get('anomaly_periods'):
                # Show trees with explanations
                await show_all_anomaly_periods_with_explanations(analysis_data)
                
                # Final summary
                anomaly_periods = analysis_data['anomaly_periods']
                aggregation_days = analysis_data['aggregation_days']
                total_periods = analysis_data.get('total_periods', 74)
                
                print(f"\n🎯 FINAL SUMMARY:")
                print(f"   📊 Total periods analyzed: 7")
                print(f"   🚨 Periods with anomalies: {len(anomaly_periods)}")
                print(f"   📈 Aggregation: {aggregation_days} days")
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

# Remove the redundant old main and summary functions
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
            'periods_analyzed': periods_to_analyze
        }
    else:
        print("✅ No anomalies detected in the 7 most recent periods")
        return {
            'detector': detector,
            'data_folder': data_folder,
            'aggregation_days': aggregation_days,
            'anomaly_periods': [],
            'total_periods': 7,
            'periods_analyzed': periods_to_analyze
        }

if __name__ == "__main__":
    asyncio.run(main()) 