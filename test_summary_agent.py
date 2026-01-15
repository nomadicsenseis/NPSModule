import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType

async def test_summary_consolidation():
    print("🚀 Starting Summary Agent Test with real Interpreter logs...")
    
    # Path to logs
    logs_dir = Path("CLAUDE_SONNET_4_5_agent_conversations/anomaly_interpreter")
    
    # 1. Identify weekly report
    weekly_file = logs_dir / "interpreter_2025-12-28 to 2026-01-02_20260107_134420.json"
    
    # 2. Identify daily reports (ordered chronologically)
    daily_files = [
        "interpreter_2025-12-28 to 2025-12-28_20260107_142701.json",
        "interpreter_2025-12-29 to 2025-12-29_20260107_141956.json",
        "interpreter_2025-12-30 to 2025-12-30_20260107_142224.json",
        "interpreter_2025-12-31 to 2025-12-31_20260107_140455.json",
        "interpreter_2026-01-01 to 2026-01-01_20260107_140426.json",
        "interpreter_2026-01-02 to 2026-01-02_20260107_140346.json"
    ]
    
    # Load weekly report content
    with open(weekly_file, 'r', encoding='utf-8') as f:
        weekly_data = json.load(f)
        weekly_analysis = ""
        for entry in reversed(weekly_data.get('conversation_log', [])):
            if entry.get('metadata', {}).get('step') == 'EXECUTIVE_SYNTHESIS':
                weekly_analysis = entry.get('content', '')
                break
    
    if not weekly_analysis:
        print(f"❌ Could not find executive synthesis in weekly report: {weekly_file}")
        return

    # Load daily reports content
    daily_analyses = []
    for filename in daily_files:
        path = logs_dir / filename
        # Extract date from filename: interpreter_YYYY-MM-DD to YYYY-MM-DD...
        date_str = filename.split('_')[1].split(' to ')[0]
        
        if not path.exists():
            print(f"⚠️ Warning: File not found {path}")
            continue

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
                    'anomalies': [] 
                })
    
    print(f"📊 Loaded weekly report and {len(daily_analyses)} daily reports.")
    
    # Initialize Summary Agent
    # Environment "local" to use temporary credentials if needed
    agent = AnomalySummaryAgent(environment="local")
    
    print("🤖 Running stratified summary (Steps 1, 2, 3 and 4)...")
    result = await agent.generate_comprehensive_summary_stratified(
        weekly_comparative_analysis=weekly_analysis,
        daily_single_analyses=daily_analyses,
        date_flight_local="2026-01-02"
    )
    
    # Save result to a file
    output_path = "test_summary_output_final.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"✅ Summary generated and saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(test_summary_consolidation())
