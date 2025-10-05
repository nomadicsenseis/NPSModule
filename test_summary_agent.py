#!/usr/bin/env python3
"""
Test script para validar el Summary Agent con datos reales del Interpreter
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Import the summary agent and helper function
from dashboard_analyzer.weekly_deep_research import generate_consolidated_summary
from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type


def load_interpreter_results(conversations_dir: str) -> list:
    """Load interpreter results from JSON files"""
    conversations_path = Path(conversations_dir)
    
    # Load the 7 daily analyses
    daily_files = [
        "interpreter_2025-09-24 to 2025-09-24_20251004_221333.json",
        "interpreter_2025-09-25 to 2025-09-25_20251004_215619.json",
        "interpreter_2025-09-26 to 2025-09-26_20251004_213720.json",
        "interpreter_2025-09-27 to 2025-09-27_20251004_211911.json",
        "interpreter_2025-09-28 to 2025-09-28_20251004_205931.json",
        "interpreter_2025-09-29 to 2025-09-29_20251004_204230.json",
        "interpreter_2025-09-30 to 2025-09-30_20251004_202428.json",
    ]
    
    all_periods_data = []
    
    for filename in daily_files:
        file_path = conversations_path / filename
        
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try multiple strategies to extract the interpretation
        final_interpretation = None
        
        # Strategy 1: Look for EXECUTIVE_SYNTHESIS in conversation_log
        for entry in reversed(data.get('conversation_log', [])):
            if entry.get('type') == 'STEP_RESPONSE' and entry.get('metadata', {}).get('step') == 'EXECUTIVE_SYNTHESIS':
                content = entry.get('content', '')
                if content and len(content) > 100:  # Must have substantial content
                    final_interpretation = content
                    break
        
        # Strategy 2: Look for FINAL_SYNTHESIS type
        if not final_interpretation:
            for entry in reversed(data.get('conversation_log', [])):
                if entry.get('type') == 'FINAL_SYNTHESIS':
                    content = entry.get('content', '')
                    if content and len(content) > 100:
                        final_interpretation = content
                        break
        
        # Strategy 3: Get the last substantial STEP_RESPONSE
        if not final_interpretation:
            for entry in reversed(data.get('conversation_log', [])):
                if entry.get('type') == 'STEP_RESPONSE':
                    content = entry.get('content', '')
                    if content and len(content) > 100:
                        final_interpretation = content
                        break
        
        # Strategy 4: Check hierarchy_summary field
        if not final_interpretation:
            hierarchy_summary = data.get('hierarchy_summary', '')
            if isinstance(hierarchy_summary, str) and len(hierarchy_summary) > 100:
                final_interpretation = hierarchy_summary
        
        if final_interpretation:
            # Extract date range from metadata
            date_range = data.get('metadata', {}).get('analysis_date', 'Unknown')
            
            period_data = {
                'period': len(all_periods_data) + 1,
                'date_range': date_range,
                'ai_interpretation': final_interpretation
            }
            all_periods_data.append(period_data)
            print(f"✅ Loaded: {filename} ({len(final_interpretation)} chars)")
        else:
            print(f"❌ No interpretation found in: {filename}")
    
    return all_periods_data


async def test_summary_single_period():
    """Test with a single period (deep_research_period just returns the interpretation)"""
    print("\n" + "="*80)
    print("TEST 1: SINGLE PERIOD (deep_research_period returns raw interpretation)")
    print("="*80)
    
    conversations_dir = "dashboard_analyzer/agent_conversations/anomaly_interpreter"
    all_periods_data = load_interpreter_results(conversations_dir)
    
    if not all_periods_data:
        print("❌ No data loaded!")
        return
    
    # Take only the first period
    single_period = [all_periods_data[0]]
    
    print(f"\n📊 Testing with {len(single_period)} period")
    print(f"   Date: {single_period[0]['date_range']}")
    print(f"   Interpretation length: {len(single_period[0]['ai_interpretation'])} chars")
    
    # Simulate the conditional logic from show_all_anomaly_periods_with_explanations
    if len(single_period) == 1:
        result = single_period[0]['ai_interpretation']
        print(f"\n✅ Result type: {type(result)}")
        print(f"✅ Result length: {len(result)} chars")
        print(f"\n📝 Result preview (first 500 chars):")
        print("-" * 80)
        print(result[:500])
        print("-" * 80)
        return result
    

async def test_summary_multiple_periods():
    """Test with multiple periods (weekly_deep_research consolidates with summary agent)"""
    print("\n" + "="*80)
    print("TEST 2: WEEKLY CONSOLIDATION (weekly calls summary agent)")
    print("="*80)
    
    conversations_dir = "dashboard_analyzer/agent_conversations/anomaly_interpreter"
    
    # Load weekly analysis (comparative)
    weekly_file = Path(conversations_dir) / "interpreter_2025-09-24 to 2025-09-30_20251004_195314.json"
    weekly_interpretation = ""
    
    if weekly_file.exists():
        with open(weekly_file, 'r', encoding='utf-8') as f:
            weekly_data = json.load(f)
        
        # Try to extract weekly interpretation using same strategies
        for entry in reversed(weekly_data.get('conversation_log', [])):
            if entry.get('type') == 'STEP_RESPONSE' and entry.get('metadata', {}).get('step') == 'EXECUTIVE_SYNTHESIS':
                content = entry.get('content', '')
                if content and len(content) > 100:
                    weekly_interpretation = content
                    break
        
        if not weekly_interpretation:
            for entry in reversed(weekly_data.get('conversation_log', [])):
                if entry.get('type') == 'STEP_RESPONSE':
                    content = entry.get('content', '')
                    if content and len(content) > 100:
                        weekly_interpretation = content
                        break
        
        if weekly_interpretation:
            print(f"✅ Loaded weekly analysis: {len(weekly_interpretation)} chars")
        else:
            print(f"⚠️ Weekly file found but no interpretation extracted")
    else:
        print(f"⚠️ Weekly file not found: {weekly_file.name}")
    
    # Load daily analyses
    all_periods_data = load_interpreter_results(conversations_dir)
    
    if len(all_periods_data) < 2:
        print(f"❌ Not enough data loaded! Got {len(all_periods_data)} periods, need at least 2")
        return
    
    print(f"\n📊 Simulating weekly consolidation with:")
    print(f"   Weekly comparative: {len(weekly_interpretation)} chars")
    print(f"   Daily periods: {len(all_periods_data)} days")
    for i, period in enumerate(all_periods_data, 1):
        print(f"   Day {i}: {period['date_range']} ({len(period['ai_interpretation'])} chars)")
    
    # Initialize Summary Agent (as weekly_deep_research.py does)
    try:
        print("\n🤖 Initializing Summary Agent...")
        summary_agent = AnomalySummaryAgent(
            llm_type=get_default_llm_type(),
            logger=logging.getLogger("test_summary_agent")
        )
        print("✅ Summary Agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Summary Agent: {e}")
        return
    
    # Format data as weekly_deep_research.py does
    try:
        # Format daily analyses in the expected format
        daily_single_analyses = []
        for period in all_periods_data:
            daily_single_analyses.append({
                'date': period['date_range'],
                'analysis': period['ai_interpretation'],
                'anomalies': ['daily_analysis']
            })
        
        print(f"✅ Formatted weekly + {len(daily_single_analyses)} daily analyses")
        
        # Prepare consolidated data structure (as in weekly_deep_research.py)
        consolidated_data = [{
            'weekly_comparative': weekly_interpretation,
            'daily_singles': daily_single_analyses,
            'metadata': {
                'analysis_date': '2025-09-30',
                'segment': 'Global',
                'causal_filter': 'vs L7d'
            },
            'weekly_params': {
                'anomaly_detection_mode': 'vslast',
                'baseline_periods': 7,
                'aggregation_days': 7,
                'periods': 1,
                'study_mode': 'comparative'
            },
            'daily_params': {
                'daily_anomaly_detection_mode': 'mean',
                'daily_baseline_periods': 7,
                'daily_aggregation_days': 1,
                'daily_periods': 7
            }
        }]
        
        print("\n🔄 Calling generate_consolidated_summary (weekly format)...")
        consolidated_summary = await asyncio.wait_for(
            generate_consolidated_summary(
                summary_agent,
                consolidated_data,
                date_flight_local="2025-09-30"
            ),
            timeout=120.0
        )
        
        print(f"\n✅ Summary generated successfully!")
        print(f"✅ Result type: {type(consolidated_summary)}")
        print(f"✅ Result length: {len(consolidated_summary)} chars")
        print(f"\n📝 Consolidated Summary:")
        print("=" * 80)
        print(consolidated_summary)
        print("=" * 80)
        
        # Get performance metrics
        metrics = summary_agent.get_performance_metrics()
        print(f"\n📊 Summary Agent Metrics:")
        print(f"   • Input tokens: {metrics.get('input_tokens', 0):,}")
        print(f"   • Output tokens: {metrics.get('output_tokens', 0):,}")
        print(f"   • Total cost: ${metrics.get('total_cost', 0):.4f}")
        
        return consolidated_summary
        
    except asyncio.TimeoutError:
        print("❌ Summary generation timed out after 120 seconds")
        return None
    except Exception as e:
        print(f"❌ Error generating consolidated summary: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all tests"""
    print("\n" + "🧪 " + "="*78)
    print("TESTING SUMMARY AGENT WITH REAL INTERPRETER DATA")
    print("="*80)
    
    # Test 1: Single period
    await test_summary_single_period()
    
    # Test 2: Multiple periods
    await test_summary_multiple_periods()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(main())
