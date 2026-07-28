"""JSON tool schemas for the Anthropic Messages API.

One entry per QC tool. Each entry follows the Anthropic tool-use format:
  { "name": ..., "description": ..., "input_schema": { "type": "object", ... } }

Parameter descriptions and valid ranges come from CLAUDE.md §7 (tool inventory) and
§7.2 (practical parameter ranges measured via param_sweep). Keep this file in sync
with wrappers.py — every function exposed there must have a schema here.

Tools are grouped:
  Utility    — inspect_dataset, get_flag_summary, export_clean_data
  Detection  — flag_range, flag_constants, flag_plateau, flag_spike_unilof,
                flag_zscore, flag_jumps, flag_nan
  Action     — impute_rolling

Implemented in Phase 2.
"""

# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------

_INSPECT_DATASET = {
    "name": "inspect_dataset",
    "description": (
        "ALWAYS call this first. Summarises the loaded time series: number of rows, "
        "time range, inferred sampling frequency, NaN count and percentage, and per-column "
        "min / max / mean / std. Read the summary before deciding which detection tools to "
        "run or what parameter values to use."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            }
        },
        "required": [],
    },
}

_GET_FLAG_SUMMARY = {
    "name": "get_flag_summary",
    "description": (
        "Returns a count of flagged timestamps broken down by the tool that raised each "
        "flag. Call this after running all detection tools and before export_clean_data to "
        "see a consolidated picture of what was found."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            }
        },
        "required": [],
    },
}

_EXPORT_CLEAN_DATA = {
    "name": "export_clean_data",
    "description": (
        "Exports the current state of the dataset as a DataFrame with a 'flag' column "
        "naming the tool that flagged each row, or None for clean rows. Call this as the "
        "final step before writing the cleaned CSV."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            }
        },
        "required": [],
    },
}

# ---------------------------------------------------------------------------
# Detection tools
# ---------------------------------------------------------------------------

_FLAG_RANGE = {
    "name": "flag_range",
    "description": (
        "Physical-gate check: flags any value outside [min, max]. "
        "Use this as a hard sanity filter, not a sensitivity knob. "
        "For turbidity: min=0 always; max=1000–2000 NTU depending on the gauge regime. "
        "Raise max only if the current value clips real storm-flush extremes. "
        "Lower max only to catch a known over-range hardware fault. "
        "Run this before spike/plateau detectors so the physical outliers don't skew "
        "neighbourhood statistics."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "min": {
                "type": ["number", "null"],
                "description": (
                    "Lower bound (inclusive). Values strictly below this are flagged. "
                    "For turbidity always set to 0. Pass null to skip the lower bound check."
                ),
                "default": None,
            },
            "max": {
                "type": ["number", "null"],
                "description": (
                    "Upper bound (inclusive). Values strictly above this are flagged. "
                    "Practical range for turbidity: 1000–2000 NTU. "
                    "Pass null to skip the upper bound check."
                ),
                "default": None,
            },
        },
        "required": [],
    },
}

_FLAG_CONSTANTS = {
    "name": "flag_constants",
    "description": (
        "Detects a stuck sensor: flags runs of values that barely change over a rolling "
        "window. A sensor that stops responding keeps transmitting its last valid reading, "
        "producing a flat stretch. "
        "IMPORTANT — thresh must be much smaller than the series noise sd (§7.1): "
        "for turbidity, keep thresh ≤ 0.05 NTU. Values above 0.5 will swallow the whole "
        "series. "
        "Increase thresh only if the sensor's noise sd is unusually large. "
        "Use window 3h–12h; shorter windows produce more false positives on naturally "
        "steady overnight periods. min_periods=2 is safe to leave at default."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "thresh": {
                "type": "number",
                "description": (
                    "Maximum allowed variation within the window. Values are flagged when "
                    "the range of readings within the window is ≤ thresh. "
                    "Practical range: ≤ 0.05 NTU for turbidity. Default 0.01."
                ),
                "default": 0.01,
            },
            "window": {
                "type": ["string", "null"],
                "description": (
                    "Rolling window size as a pandas offset string, e.g. '3h', '6h', '12h'. "
                    "Practical range: '3h'–'12h'. Required — do not leave null."
                ),
                "default": None,
            },
            "min_periods": {
                "type": "integer",
                "description": "Minimum number of observations required in the window. Default 2.",
                "default": 2,
            },
        },
        "required": ["thresh", "window"],
    },
}

_FLAG_PLATEAU = {
    "name": "flag_plateau",
    "description": (
        "Detects an offset plateau: a segment displaced from its surroundings (e.g. debris "
        "temporarily stuck on a sensor), whose values need not be constant. Complements "
        "flag_constants — use both because they detect different failure modes (§7.1). "
        "WARNING — this tool is crash-prone (raises ValueError on some inputs) and slow "
        "(uses multiprocessing). The wrapper already catches ValueError and carries on. "
        "Set min_length well below the expected plateau duration: for a 25 h plateau, "
        "'1h' and '3h' work, '6h' and above miss it entirely. "
        "Practical range for min_length: '1h'–'3h' (default '1h'). "
        "Rely primarily on flag_constants; use flag_plateau as a secondary check."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "min_length": {
                "type": ["string", "null"],
                "description": (
                    "Minimum plateau duration as a pandas offset string, e.g. '1h', '3h'. "
                    "Must be set well below the true plateau length. Practical range: '1h'–'3h'."
                ),
                "default": None,
            },
            "max_length": {
                "type": ["string", "null"],
                "description": (
                    "Maximum plateau duration as a pandas offset string. Pass null for no upper limit."
                ),
                "default": None,
            },
            "min_jump": {
                "type": ["number", "null"],
                "description": (
                    "Minimum absolute jump at the plateau boundary to qualify as a plateau. "
                    "Pass null to use the SaQC default."
                ),
                "default": None,
            },
            "granularity": {
                "type": ["string", "null"],
                "description": "Time granularity for plateau detection. Pass null to use SaQC default.",
                "default": None,
            },
        },
        "required": [],
    },
}

_FLAG_SPIKE_UNILOF = {
    "name": "flag_spike_unilof",
    "description": (
        "Primary spike detector. Uses the Local Outlier Factor (LOF) algorithm: a point is "
        "flagged if its local density is much lower than that of its n nearest neighbours. "
        "This is the preferred spike detector — it typically outperforms flag_zscore "
        "(§7.2). Run it before flag_zscore. "
        "Key parameter: thresh (the LOF ratio threshold). Practical range: 1.2–2.0, default 1.5. "
        "The LOF ratio is scale-invariant, so the same range applies across gauges. "
        "Increase thresh (toward 2.0) when: the base is naturally spiky, anomalies are sparse "
        "(low contamination level), or you see many false positives. "
        "Decrease thresh (toward 1.2) when: missing spikes in dense-anomaly data, or the "
        "base is unusually calm."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "n": {
                "type": "integer",
                "description": (
                    "Number of nearest neighbours for the LOF calculation. "
                    "Default 20; rarely needs changing."
                ),
                "default": 20,
            },
            "thresh": {
                "type": ["number", "null"],
                "description": (
                    "LOF ratio threshold. A point whose LOF score exceeds this is flagged as a spike. "
                    "Practical range: 1.2–2.0. Default 1.5. "
                    "Increase for fewer false positives; decrease to catch more spikes."
                ),
                "default": None,
            },
            "density": {
                "type": "string",
                "description": "Density estimation method. 'auto' is the correct default — do not change.",
                "default": "auto",
            },
            "slope_correct": {
                "type": "boolean",
                "description": (
                    "Whether to apply slope correction before LOF. Leave True (default) — "
                    "it improves detection on trending segments."
                ),
                "default": True,
            },
        },
        "required": [],
    },
}

_FLAG_ZSCORE = {
    "name": "flag_zscore",
    "description": (
        "Backup spike detector using a rolling z-score. Flags points that deviate more than "
        "thresh standard deviations from the local rolling mean/median. "
        "Use method='modified' with a ~12h window as the default configuration. "
        "IMPORTANT — on quantised data (e.g. turbidity rounded to 0.1 NTU), method='modified' "
        "can flag hundreds of segments at any thresh because the MAD collapses to ~0 in flat "
        "windows. If you see this, increase thresh significantly or switch to method='standard'. "
        "Practical range for thresh: 6–12 (default 8 for modified). "
        "Increase thresh (toward 12) on spikier/more-variable gauges or when seeing false positives. "
        "Decrease (toward 6) on calm gauges or when missing spikes. "
        "Run flag_spike_unilof first — this is the backup when UniLOF misses isolated spikes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "method": {
                "type": "string",
                "enum": ["standard", "modified"],
                "description": (
                    "Z-score method. 'modified' uses the median and MAD (more robust to outliers). "
                    "'standard' uses mean and sd. Default 'modified' with window ~12h."
                ),
                "default": "modified",
            },
            "window": {
                "type": ["string", "null"],
                "description": (
                    "Rolling window as a pandas offset string, e.g. '12h', '6h'. "
                    "Use ~12h for modified method. Required — do not leave null."
                ),
                "default": None,
            },
            "thresh": {
                "type": "number",
                "description": (
                    "Z-score threshold. Points above this (in absolute value) are flagged. "
                    "For modified method: practical range 6–12, default 8. "
                    "For standard method: default 3."
                ),
                "default": 8.0,
            },
        },
        "required": ["window"],
    },
}

_FLAG_JUMPS = {
    "name": "flag_jumps",
    "description": (
        "Detects permanent level shifts: a sudden step-change where the sensor jumps to a "
        "new level and stays there (e.g. sensor physically displaced, bad recalibration, "
        "sediment accumulation). "
        "WARNING — this tool is an aid, not a reliable classifier (§7.2). There are only ~3 "
        "level_shift episodes per dataset, and flag_jumps also fires on sharp storm limbs and "
        "recovery ramps. Treat its flags as 'look here' signals; the agent must reason about "
        "context before deciding to delete or keep each flagged segment. "
        "Practical range for thresh: 1–5 NTU (default 2). "
        "Increase thresh to cut false positives from storm ramps. "
        "Decrease thresh to catch small but genuine level shifts. "
        "window is required — use 1h–6h."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "thresh": {
                "type": "number",
                "description": (
                    "Minimum absolute step size (in data units) to flag as a jump. "
                    "Practical range: 1–5 NTU for turbidity. Default 2."
                ),
                "default": 2.0,
            },
            "window": {
                "type": ["string", "null"],
                "description": (
                    "Look-back/look-forward window as a pandas offset string, e.g. '1h', '3h'. "
                    "Required — do not leave null."
                ),
                "default": None,
            },
        },
        "required": ["thresh", "window"],
    },
}

_FLAG_NAN = {
    "name": "flag_nan",
    "description": (
        "Flags every missing (NaN) timestamp in the series. Gaps arise from power outages, "
        "network failures, maintenance, or removed sensors. "
        "Run this after all spike/plateau/jump detectors so the gap flags are counted "
        "separately from detection flags. The flags feed get_flag_summary and the imputation "
        "decision: only short gaps should be passed to impute_rolling."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            }
        },
        "required": [],
    },
}

# ---------------------------------------------------------------------------
# Action tools
# ---------------------------------------------------------------------------

_IMPUTE_ROLLING = {
    "name": "impute_rolling",
    "description": (
        "Fills NaN gaps using a rolling window median (or other aggregation). "
        "Set max_gap to the longest gap you are willing to impute — any gap longer than "
        "max_gap is left as NaN and reported in the result. "
        "Do NOT impute long outages (hours to days); rolling median on a large gap produces "
        "flat, unrealistic values that degrade data quality more than leaving them as NaN. "
        "WORKFLOW: call inspect_dataset then flag_nan first to see gap sizes, then decide "
        "an appropriate max_gap before calling this tool. "
        "Practical range for window and max_gap: '1h'–'6h'. Default 3h. "
        "window must be at least as large as max_gap so the roller has enough context "
        "to bridge the gap (§7.1). "
        "func='median' is more robust than 'mean' near anomalous neighbours — keep it. "
        "The result includes n_gaps_filled, n_gaps_skipped_large, and a gaps_summary list "
        "so you can audit exactly what was and was not filled."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "window": {
                "type": ["string", "null"],
                "description": (
                    "Rolling window size as a pandas offset string, e.g. '3h', '6h'. "
                    "Must be >= max_gap so the roller can bridge the full gap. "
                    "Practical range: '1h'–'6h'. Required — do not leave null."
                ),
                "default": None,
            },
            "func": {
                "type": "string",
                "enum": ["median", "mean"],
                "description": "Aggregation function for the rolling window. Default 'median'.",
                "default": "median",
            },
            "min_periods": {
                "type": "integer",
                "description": (
                    "Minimum number of non-NaN values required in the window to produce an "
                    "imputed value. 0 means impute even when most of the window is NaN. Default 0."
                ),
                "default": 0,
            },
            "max_gap": {
                "type": ["string", "null"],
                "description": (
                    "Maximum gap duration to impute, as a pandas offset string, e.g. '3h', '1h'. "
                    "Gaps longer than this are skipped and left as NaN. "
                    "Always set this — do not impute long maintenance outages or multi-hour dropouts. "
                    "Practical range: '1h'–'6h'. If null, all gaps up to window size are filled."
                ),
                "default": None,
            },
        },
        "required": ["window"],
    },
}

# ---------------------------------------------------------------------------
# Public list — pass this directly to the Anthropic Messages API `tools` argument
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    # Utility
    _INSPECT_DATASET,
    _GET_FLAG_SUMMARY,
    _EXPORT_CLEAN_DATA,
    # Detection
    _FLAG_RANGE,
    _FLAG_CONSTANTS,
    _FLAG_PLATEAU,
    _FLAG_SPIKE_UNILOF,
    _FLAG_ZSCORE,
    _FLAG_JUMPS,
    _FLAG_NAN,
    # Action
    _IMPUTE_ROLLING,
]

# Convenience: look up a schema by tool name
TOOL_SCHEMA_BY_NAME: dict[str, dict] = {s["name"]: s for s in TOOL_SCHEMAS}
