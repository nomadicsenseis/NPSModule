"""
Iberia NPS Deep Research - AI-powered anomaly detection and explanation system.

This package provides tools for:
- Automatic NPS anomaly detection across hierarchical customer segments
- Multi-source root cause investigation (Power BI, NCS incidents, customer verbatims)
- AI-powered explanations via OpenAI or AWS Bedrock
- Executive-level reporting with actionable insights

Usage:
    # CLI
    nps-weekly --date-flight-local "2025-01-15" --environment prod
    nps-research --study-mode comparative --segment "Global/LH"

    # Python API
    from dashboard_analyzer import run_weekly_comprehensive_analysis
    await run_weekly_comprehensive_analysis(...)
"""

__version__ = "0.1.0"
__author__ = "Iberia Data Team"

# Main entry points for programmatic use
from dashboard_analyzer.weekly_deep_research import run_weekly_comprehensive_analysis
from dashboard_analyzer.deep_research_period import execute_analysis_flow

__all__ = [
    "run_weekly_comprehensive_analysis",
    "execute_analysis_flow",
    "__version__",
]

