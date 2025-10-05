#!/usr/bin/env python3
"""
NPS Anomaly Detection System - Main Wrapper
Routes execution to specialized scripts based on mode
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Simple wrapper that routes to specialized scripts"""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Parse --mode argument
    mode = 'weekly'  # default
    remaining_args = []
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--mode' and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        else:
            remaining_args.append(args[i])
            i += 1
    
    # Route to the appropriate script
    if mode == 'weekly':
        script = script_dir / 'weekly_deep_research.py'
        print(f"🔀 Routing to Weekly Deep Research...")
    elif mode == 'custom':
        script = script_dir / 'deep_research_period.py'
        print(f"🔀 Routing to Custom Period Analysis...")
    else:
        print(f"❌ Error: Invalid mode '{mode}'. Choose 'weekly' or 'custom'.")
        sys.exit(1)
    
    # Execute the appropriate script
    try:
        result = subprocess.run(
            [sys.executable, str(script)] + remaining_args,
            cwd=script_dir.parent  # Run from parent directory (dashboard_analyzer parent)
        )
        sys.exit(result.returncode)
            except Exception as e:
        print(f"❌ Error executing {script.name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

