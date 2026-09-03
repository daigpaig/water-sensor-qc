"""Ablation study script.

Runs the agent multiple times on the same dataset, disabling one or more tools
at a time, to see how detection and imputation scores are affected.

Usage:
    python -m src.workbench.ablation data/injected/02054550/l1/02054550_l1.csv \
        --disable flag_spike_unilof flag_constants
"""
import argparse
import sys
from pathlib import Path

import saqc
from src.agent import run_agent, TOOL_SCHEMAS
from src.inspect_data import load_series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an ablation study on the QC agent.")
    parser.add_argument("series", type=Path, help="Dataset CSV")
    parser.add_argument(
        "--disable", nargs="+", required=True,
        help="List of tool names to disable (e.g., flag_spike_unilof, impute_rolling)"
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Directory for output files. Default: same directory as the input CSV."
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/ablation",
        help="Directory for JSONL run logs (default: logs/ablation/).",
    )
    args = parser.parse_args(argv)

    out_dir = args.outdir or args.series.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.series.stem
    
    # 1. Filter the tool schemas
    disabled_set = set(args.disable)
    ablated_schemas = [t for t in TOOL_SCHEMAS if t["name"] not in disabled_set]
    
    # Check if we actually disabled anything
    if len(ablated_schemas) == len(TOOL_SCHEMAS):
        print(f"Warning: none of {disabled_set} matched any tool names.")
        return 1
        
    print(f"Ablation run on {args.series.name}")
    print(f"Disabled tools: {disabled_set}")
    print(f"Active tools: {len(ablated_schemas)} / {len(TOOL_SCHEMAS)}")

    # We need to temporarily patch TOOL_SCHEMAS in src.agent
    import src.agent
    original_schemas = src.agent.TOOL_SCHEMAS
    src.agent.TOOL_SCHEMAS = ablated_schemas

    try:
        df = load_series(args.series)
        data = df.set_index("datetime")
        qc = saqc.SaQC(data)
        
        # Run agent
        final_qc, clean_df, report, summary = run_agent(
            qc, max_steps=25, log_dir=args.log_dir
        )
        
        # Output results
        suffix = "_ablation_" + "_".join(args.disable)
        
        if clean_df is not None:
            out = clean_df.copy()
            if out.index.name == "datetime" or "datetime" not in out.columns:
                out = out.reset_index()
            out.to_csv(out_dir / f"{stem}{suffix}_clean.csv", index=False)
            
        print("\nAgent finished.")
        print(f"Cost: ${summary.total_cost:.3f}")
        print(f"Run `evaluate.py` on the resulting log file in {args.log_dir} to see the impact.")
        
    finally:
        # Restore schemas just in case this is run interactively
        src.agent.TOOL_SCHEMAS = original_schemas

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
