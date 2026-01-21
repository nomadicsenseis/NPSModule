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
    
    # 1. Identify weekly report (range format: YYYY-MM-DD to YYYY-MM-DD)
    weekly_file = logs_dir / "interpreter_2026-01-11 to 2026-01-17_20260121_112958.json"
    
    # 2. Identify daily reports (ordered chronologically)
    daily_files = [
        "interpreter_2026-01-11 to 2026-01-11_20260121_121944.json",
        "interpreter_2026-01-12 to 2026-01-12_20260121_121722.json",
        "interpreter_2026-01-13 to 2026-01-13_20260121_121615.json",
        "interpreter_2026-01-14 to 2026-01-14_20260121_120430.json",
        "interpreter_2026-01-15 to 2026-01-15_20260121_115334.json",
        "interpreter_2026-01-16 to 2026-01-16_20260121_114952.json",
        "interpreter_2026-01-17 to 2026-01-17_20260121_115327.json",
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

    print(f"✅ Weekly report loaded: {len(weekly_analysis)} chars")

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
                print(f"  ✅ Daily {date_str}: {len(daily_analysis_text)} chars")
    
    print(f"📊 Loaded weekly report and {len(daily_analyses)} daily reports.")
    
    # Initialize Summary Agent
    # Environment "local" to use temporary credentials if needed
    agent = AnomalySummaryAgent(environment="local")
    
    print("🤖 Running stratified summary (Steps 1-6)...")
    result = await agent.generate_comprehensive_summary_stratified(
        weekly_comparative_analysis=weekly_analysis,
        daily_single_analyses=daily_analyses,
        date_flight_local="2026-01-17",
        segment="Global"
    )
    
    # Save result to a file
    output_path = "test_summary_output_final.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"✅ Summary generated and saved to {output_path}")
    
    # Also extract and save Adaptive Card JSON separately
    if "---ADAPTIVE_CARD_JSON---" in result:
        parts = result.split("---ADAPTIVE_CARD_JSON---")
        if len(parts) > 1:
            adaptive_card = parts[1].strip()
            with open("test_adaptive_card.json", 'w', encoding='utf-8') as f:
                f.write(adaptive_card)
            print(f"✅ Adaptive Card saved to test_adaptive_card.json ({len(adaptive_card)} chars)")

if __name__ == "__main__":
    asyncio.run(test_summary_consolidation())
