"""Scratch script to dry-run the agent loop on a piece of data.
Requires ANTHROPIC_API_KEY in the environment.
"""

import os
import pandas as pd
import saqc
from dotenv import load_dotenv

from src.agent import run_agent
from src.inspect_data import load_series

def main():
    load_dotenv()
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Please set ANTHROPIC_API_KEY in .env before running this script.")
        return
        
    # Create some dummy data if we don't have a dataset handy
    print("Setting up a toy dataset...")
    idx = pd.date_range("2024-01-01", periods=100, freq="15min")
    data = pd.DataFrame({"value": range(100)}, index=idx)
    qc = saqc.SaQC(data)
    
    print("Running the agent loop (max_steps=5 for safety)...")
    final_qc, clean_df, report = run_agent(qc, max_steps=5, log_dir="logs")
    
    print("\n" + "="*50)
    print("AGENT REPORT:")
    print("="*50)
    print(report)
    
    if clean_df is not None:
        print("\nClean DataFrame was successfully extracted!")
        print(clean_df.head())
    
    print("\nLogs should be written to the logs/ directory.")

if __name__ == "__main__":
    main()
