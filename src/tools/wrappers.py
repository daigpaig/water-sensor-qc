"""SaQC-wrapping tool functions.

Each function wraps a SaQC 2.8 method and returns the tool-result dict defined in
CLAUDE.md §5. Utility (inspect_dataset, get_flag_summary, export_clean_data),
detection (flag_range, flag_constants, flag_spike_unilof, flag_zscore,
flag_jumps, flag_nan), and action (impute_rolling, correct_drift) tools.

NOTE: verify every SaQC method name/signature against the SaQC 2.8 API before use.

Implemented in Phase 2 (see CLAUDE.md §12).
"""
