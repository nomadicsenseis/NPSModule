import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType

async def test_summary_consolidation():
    print("🚀 Starting Summary Agent Test...")
    
    # Path to logs
    logs_dir = Path("CLAUDE_SONNET_4_5_agent_conversations/anomaly_interpreter")
    
    # 1. Identify weekly report
    weekly_file = logs_dir / "interpreter_2025-12-13 to 2025-12-19_20251223_095144.json"
    
    # 2. Identify daily reports
    daily_files = [
        "interpreter_2025-12-13 to 2025-12-13_20251223_105100.json",
        "interpreter_2025-12-14 to 2025-12-14_20251223_104041.json",
        "interpreter_2025-12-15 to 2025-12-15_20251223_103206.json",
        "interpreter_2025-12-16 to 2025-12-16_20251223_104045.json",
        "interpreter_2025-12-17 to 2025-12-17_20251223_102209.json",
        "interpreter_2025-12-18 to 2025-12-18_20251223_102331.json",
        "interpreter_2025-12-19 to 2025-12-19_20251223_102436.json"
    ]
    
    # Load weekly report content
    with open(weekly_file, 'r', encoding='utf-8') as f:
        weekly_data = json.load(f)
        # In our logs, the final interpretation is in the last STEP_RESPONSE or in a compiled format
        # but the JSON has a specific structure. Let's find the final synthesis.
        weekly_analysis = ""
        for entry in reversed(weekly_data.get('conversation_log', [])):
            if entry.get('metadata', {}).get('step') == 'EXECUTIVE_SYNTHESIS':
                weekly_analysis = entry.get('content', '')
                break
    
    if not weekly_analysis:
        print("❌ Could not find executive synthesis in weekly report")
        return

    # Load daily reports content
    daily_analyses = []
    for filename in daily_files:
        path = logs_dir / filename
        date_str = filename.split('_')[1].split(' to ')[0]
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            daily_analysis_text = ""
            for entry in reversed(data.get('conversation_log', [])):
                if entry.get('metadata', {}).get('step') == 'EXECUTIVE_SYNTHESIS':
                    daily_analysis_text = entry.get('content', '')
                    break
            
            if daily_analysis_text:
                daily_analyses.append({
                    'date': date_str,
                    'analysis': daily_analysis_text,
                    'anomalies': [] # Not strictly needed for the stratified mode
                })
    
    print(f"📊 Loaded weekly report and {len(daily_analyses)} daily reports.")
    
    # Initialize Summary Agent (using O4_MINI for faster/cheaper testing if desired, or stay with default)
    # The user is having issues with SONNET 4.5 in prod, so let's test with the configured default.
    agent = AnomalySummaryAgent(environment="local") # Local to use temp_creds
    
    print("🤖 Running stratified summary...")
    result = await agent.generate_comprehensive_summary_stratified(
        weekly_comparative_analysis=weekly_analysis,
        daily_single_analyses=daily_analyses,
        date_flight_local="2025-12-19"
    )
    
    # Save result to a file
    output_path = "test_summary_output.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"✅ Summary generated and saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(test_summary_consolidation())

