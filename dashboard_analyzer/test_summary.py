"""
Test script for debugging the Summary Agent.

Loads existing interpreter JSON files to test the summary agent
without running the full analysis pipeline.
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_agent_conversations_folder


def load_interpreter_json(filepath: str) -> dict:
    """Load an interpreter JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_final_synthesis(interpreter_data: dict) -> str:
    """Extract the final synthesis from interpreter conversation log."""
    conversation_log = interpreter_data.get('conversation_log', [])
    
    # Find the EXECUTIVE_SYNTHESIS step response
    for entry in conversation_log:
        if (entry.get('type') == 'STEP_RESPONSE' and 
            entry.get('metadata', {}).get('step') == 'EXECUTIVE_SYNTHESIS'):
            return entry.get('content', '')
    
    # Fallback: look for final_interpretation
    if 'final_interpretation' in interpreter_data:
        return interpreter_data['final_interpretation']
    
    return ''


def get_analysis_date(interpreter_data: dict) -> str:
    """Extract analysis date from interpreter metadata."""
    return interpreter_data.get('metadata', {}).get('analysis_date', 'Unknown')


async def run_test():
    """Run the summary agent test."""
    
    # Path to interpreter JSONs
    interpreter_dir = Path(__file__).parent / get_agent_conversations_folder() / 'anomaly_interpreter'
    
    print("=" * 80)
    print("🧪 SUMMARY AGENT TEST")
    print("=" * 80)
    
    # Find available interpreter files
    interpreter_files = sorted(interpreter_dir.glob('interpreter_*.json'))
    
    if not interpreter_files:
        print("❌ No interpreter JSON files found!")
        return
    
    print(f"\n📂 Found {len(interpreter_files)} interpreter files:")
    for f in interpreter_files:
        print(f"   - {f.name}")
    
    # Identify weekly vs daily files
    weekly_file = None
    daily_files = []
    
    for f in interpreter_files:
        # Parse filename: interpreter_2025-11-24 to 2025-11-30_...
        name = f.stem  # Remove .json
        # Extract date range from filename
        date_part = name.replace('interpreter_', '').split('_')[0]
        
        if ' to ' in date_part:
            start_date, end_date = date_part.split(' to ')
            if start_date != end_date:
                # Weekly file (date range spans multiple days)
                weekly_file = f
            else:
                # Single day file
                daily_files.append(f)
        else:
            daily_files.append(f)
    
    print(f"\n📊 Classification:")
    print(f"   Weekly (comparative): {weekly_file.name if weekly_file else 'Not found'}")
    print(f"   Daily (single): {len(daily_files)} files")
    
    if not weekly_file:
        print("❌ No weekly interpreter file found!")
        return
    
    # Load weekly analysis
    print(f"\n📖 Loading weekly analysis: {weekly_file.name}")
    weekly_data = load_interpreter_json(str(weekly_file))
    weekly_synthesis = extract_final_synthesis(weekly_data)
    
    if not weekly_synthesis:
        print("❌ Could not extract weekly synthesis!")
        return
    
    print(f"   ✅ Extracted synthesis: {len(weekly_synthesis)} chars")
    
    # Load daily analyses
    daily_analyses = []
    for daily_file in sorted(daily_files):
        print(f"📖 Loading daily analysis: {daily_file.name}")
        daily_data = load_interpreter_json(str(daily_file))
        
        date = get_analysis_date(daily_data)
        synthesis = extract_final_synthesis(daily_data)
        
        if synthesis:
            daily_analyses.append({
                'date': date.replace(' to ', ''),  # Clean up "2025-11-30 to 2025-11-30"
                'analysis': synthesis,
                'anomalies': []
            })
            print(f"   ✅ Extracted: {len(synthesis)} chars")
        else:
            print(f"   ⚠️ No synthesis found")
    
    print(f"\n📊 Summary input prepared:")
    print(f"   Weekly: {len(weekly_synthesis)} chars")
    print(f"   Daily: {len(daily_analyses)} days")
    
    # Initialize summary agent
    print("\n🤖 Initializing Summary Agent...")
    agent = AnomalySummaryAgent(environment="local")
    
    # Run the stratified summary
    print("\n🚀 Running STRATIFIED summary (3 steps)...")
    print("-" * 80)
    
    result = await agent.generate_comprehensive_summary_stratified(
        weekly_comparative_analysis=weekly_synthesis,
        daily_single_analyses=daily_analyses,
        date_flight_local=datetime.now().strftime('%Y-%m-%d')
    )
    
    print("-" * 80)
    print("\n📋 FINAL RESULT:")
    print("=" * 80)
    print(result)
    print("=" * 80)
    
    # Save result for inspection
    output_file = Path(__file__).parent / 'test_summary_output.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Summary Agent Test Output\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Weekly Input\n\n")
        f.write(f"```\n{weekly_synthesis[:2000]}...\n```\n\n")
        f.write(f"## Daily Inputs ({len(daily_analyses)} days)\n\n")
        for d in daily_analyses:
            f.write(f"### {d['date']}\n")
            f.write(f"```\n{d['analysis'][:500]}...\n```\n\n")
        f.write(f"## Final Output\n\n")
        f.write(result)
    
    print(f"\n💾 Output saved to: {output_file}")


async def run_legacy_test():
    """Run the legacy single-step summary for comparison."""
    
    interpreter_dir = Path(__file__).parent / 'agent_conversations' / 'anomaly_interpreter'
    interpreter_files = sorted(interpreter_dir.glob('interpreter_*.json'))
    
    weekly_file = None
    daily_files = []
    
    for f in interpreter_files:
        name = f.stem
        date_part = name.replace('interpreter_', '').split('_')[0]
        if ' to ' in date_part:
            start_date, end_date = date_part.split(' to ')
            if start_date != end_date:
                weekly_file = f
            else:
                daily_files.append(f)
    
    if not weekly_file:
        print("❌ No weekly interpreter file found!")
        return
    
    weekly_data = load_interpreter_json(str(weekly_file))
    weekly_synthesis = extract_final_synthesis(weekly_data)
    
    daily_analyses = []
    for daily_file in sorted(daily_files):
        daily_data = load_interpreter_json(str(daily_file))
        date = get_analysis_date(daily_data)
        synthesis = extract_final_synthesis(daily_data)
        if synthesis:
            daily_analyses.append({
                'date': date.replace(' to ', ''),
                'analysis': synthesis,
                'anomalies': []
            })
    
    print("\n🤖 Running LEGACY summary (1 step)...")
    print("-" * 80)
    
    agent = AnomalySummaryAgent(environment="local")
    result = await agent.generate_comprehensive_summary(
        weekly_comparative_analysis=weekly_synthesis,
        daily_single_analyses=daily_analyses,
        date_flight_local=datetime.now().strftime('%Y-%m-%d')
    )
    
    print("-" * 80)
    print("\n📋 LEGACY RESULT:")
    print("=" * 80)
    print(result)
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Summary Agent')
    parser.add_argument('--legacy', action='store_true', help='Run legacy single-step mode')
    args = parser.parse_args()
    
    if args.legacy:
        asyncio.run(run_legacy_test())
    else:
        asyncio.run(run_test())

