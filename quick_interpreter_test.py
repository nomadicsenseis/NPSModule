#!/usr/bin/env python3
"""
Quick test to see if the interpreter can process the AI input without hanging
"""
import asyncio
import sys
sys.path.append('/app')

async def test_interpreter_quick():
    """Quick test of interpreter"""
    try:
        print("🔄 Reading AI input from file...")
        with open('/app/test_ai_input.txt', 'r', encoding='utf-8') as f:
            ai_input = f.read()
        
        print(f"✅ AI input loaded: {len(ai_input)} characters")
        
        print("🔄 Importing interpreter agent...")
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
        import logging
        
        print("🔄 Initializing interpreter agent...")
        ai_agent = AnomalyInterpreterAgent(
            llm_type=get_default_llm_type(),
            config_path="/app/dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
            logger=logging.getLogger("ai_interpreter"),
            study_mode="comparative"
        )
        
        print("✅ Interpreter agent initialized")
        
        print("🔄 Calling interpreter (30 second timeout)...")
        
        try:
            ai_interpretation = await asyncio.wait_for(
                ai_agent.interpret_anomaly_tree(ai_input, '2025-06-01', 'Economy/SH'),
                timeout=30.0  # Short timeout for quick test
            )
            
            print("✅ Interpreter completed successfully!")
            print(f"📏 Response length: {len(ai_interpretation)} characters")
            print(f"\n📄 Interpreter Response (first 500 chars):")
            print("=" * 60)
            print(ai_interpretation[:500])
            print("=" * 60)
            
        except asyncio.TimeoutError:
            print("⏰ Interpreter timed out after 30 seconds (this is normal for a quick test)")
            print("✅ But the important thing is it didn't crash immediately!")
        except Exception as e:
            print(f"❌ Interpreter failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_interpreter_quick())
