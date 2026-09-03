import pandas as pd
import json
from pathlib import Path
import sys

# Ensure app is in path
sys.path.append(str(Path(__file__).resolve().parent))
from app.streamlit_app import build_decisions_from_log
from src.inspect_data import load_series

def main():
    csv_path = Path("data/mock2/01467200_l1.csv")
    jsonl_path = Path("data/mock2/run_20260826_183408.jsonl")
    out_json = Path("data/mock/01467200_l1_decisions.json")
    
    # We also need a mock_clean.csv to provide the flags for the plot
    out_clean_csv = Path("data/mock/01467200_l1_clean.csv")

    df = load_series(csv_path)
    if "datetime" not in df.columns:
        df = df.reset_index()

    clean_df = df.copy()
    if "flag" not in clean_df.columns:
        clean_df["flag"] = None

    decisions = build_decisions_from_log(str(jsonl_path), clean_df)

    # Reconstruct the flags column from the parsed log
    for d in decisions:
        for ts in d.get("flagged_timestamps", []):
            clean_df.loc[clean_df["datetime"] == ts, "flag"] = d["flagged_by"]

    # Save decisions
    with open(out_json, "w") as f:
        json.dump(decisions, f, indent=2)

    # Save clean df for flags
    clean_df.to_csv(out_clean_csv, index=False)
    
    print(f"Successfully converted to {out_json} and {out_clean_csv}")

if __name__ == "__main__":
    main()
