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
  Context    — describe_points, describe_point (from context.py, not SaQC:
                they measure the shape around a flagged timestamp so the agent
                can tell a storm from an artifact), plus four single-question
                primitives: slope_context, excursion_context, recovery_context,
                level_shift_context, noise_context
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
        "naming the tool that flagged each row, or None for clean rows, AND writes the "
        "machine-readable flag log. Call this as the final step. "
        "PASS `decisions`: this is where your verdicts are recorded, and the `verdict` "
        "field in them IS this run's answer — whether each segment is genuinely "
        "anomalous. A flag on its own is only a candidate; a flagged row with no "
        "decision counts as no claim at all, so an export without decisions produces a "
        "log that says you concluded nothing. Cover every flagged segment. The result "
        "reports how many flagged rows are still undecided and which of your spans "
        "matched no flagged row; if either is non-zero, call this tool once more with "
        "corrected decisions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "decisions": {
                "type": "array",
                "description": (
                    "One entry per segment you judged. Each carries TWO separate calls: "
                    "`verdict` (is this segment genuinely anomalous — the answer you are "
                    "scored on) and `action` (what to do with the values). Where two "
                    "entries cover the same row, the NARROWEST one wins, so a broad "
                    "catch-all can never override a specific verdict and the order you "
                    "list them in does not matter. A span that covers no flagged row at "
                    "all is reported back to you, not silently dropped."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "string",
                            "description": (
                                "First timestamp of the segment, ISO 8601 "
                                "(e.g. '2024-06-01T03:00:00'). For a single point, set "
                                "start only."
                            ),
                        },
                        "end": {
                            "type": "string",
                            "description": (
                                "Last timestamp of the segment, ISO 8601, INCLUSIVE. "
                                "Omit for a single-point decision."
                            ),
                        },
                        "verdict": {
                            "type": "string",
                            "enum": ["anomaly", "normal"],
                            "description": (
                                "YOUR ANSWER for this segment, and the thing this run is "
                                "scored on. 'anomaly' — the values are genuinely faulty "
                                "(sensor artifact, stuck run, missing data). 'normal' — "
                                "a detector fired, you inspected it, and it is real "
                                "water: a storm peak, a first flush, a genuine extreme "
                                "event. This is a claim about the DATA, separate from "
                                "what you do about it, so say 'normal' whenever the "
                                "measurements say the reading is real, even though a "
                                "flag is sitting on it."
                            ),
                        },
                        "anomaly_type": {
                            "type": "string",
                            "enum": ["spike", "plateau", "level_shift", "gap"],
                            "description": (
                                "REQUIRED when verdict is 'anomaly'; omit entirely when "
                                "it is 'normal'. Which of the four failures this is — "
                                "YOUR classification, scored against the labels as such. "
                                "Do not simply echo the detector that fired: flag_jumps "
                                "firing on a sharp one-sample excursion that recovers "
                                "immediately is a spike, whatever tool found it."
                            ),
                        },
                        "action": {
                            "type": "string",
                            "enum": ["delete", "correct", "keep", "impute"],
                            "description": (
                                "What to DO with the values, which is a separate question "
                                "from the verdict. "
                                "delete — an artifact, remove the value. "
                                "correct — replace with a defensible value. "
                                "keep — the value stays exactly as recorded. "
                                "impute — a gap short enough to fill. "
                                "MUST be 'keep' when verdict is 'normal' — you cannot "
                                "call a value real water and then delete it. When verdict "
                                "is 'anomaly' this is normally delete / correct / impute, "
                                "but 'keep' is allowed for an anomaly you cannot treat — "
                                "a gap longer than any defensible imputation window, say "
                                "— provided your `reason` says why it is being left "
                                "alone. Genuine extreme events are KEPT, not corrected "
                                "away."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why, citing the numbers you measured — "
                                "'3 samples wide, robust_z 6.2, recovered in 2' is a "
                                "justification; 'looked like a spike' is not."
                            ),
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["clear", "judgement-call"],
                            "description": (
                                "How close this call was. 'clear' means the evidence was "
                                "one-sided and any careful reader would agree. "
                                "'judgement-call' means it could reasonably have gone the "
                                "other way — the measurements conflicted, the point sat "
                                "near a threshold, or you overrode a default. "
                                "Be honest: a narrow high-z excursion whose fall decays is "
                                "a judgement call, not a clear one, and marking it clear "
                                "hides the very decision a reviewer needs to check. "
                                "REQUIRED — there is no default. On a typical record "
                                "roughly 10-15% of decisions should be 'judgement-call'; "
                                "a run that marks everything 'clear' is not confident, it "
                                "is unhelpful, because the reviewer is left with nothing "
                                "to check. Deletions of narrow excursions inside a noisy "
                                "stretch are the usual judgement calls."
                            ),
                        },
                        "deliberation": {
                            "type": "string",
                            "description": (
                                "REQUIRED when difficulty is 'judgement-call' (the call "
                                "fails without it). Write out the thinking a reviewer "
                                "would need to check you: which measurements pointed "
                                "which way, what you weighed against what, what you "
                                "considered and rejected, and what would have changed "
                                "your mind. Several sentences. This is not a longer "
                                "`reason` — `reason` states the conclusion, this shows "
                                "the working. Omit it for 'clear' decisions."
                            ),
                        },
                    },
                    "required": ["start", "verdict", "action", "reason", "difficulty"],
                },
            },
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
                "type": "string",
                "description": (
                    "Minimum plateau duration as a pandas offset string, e.g. '1h', '3h'. "
                    "Must be set well below the true plateau length. Practical range: '1h'–'3h'. "
                    "Required by SaQC 2.8 — defaults to '1h' if you omit it, never null."
                ),
                "default": "1h",
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
        "sediment accumulation). It compares the MEAN of the window before each point with "
        "the MEAN of the window after it, and flags where they differ by more than thresh. "
        "SET thresh FROM inspect_dataset's jump_scale BLOCK — it measures that exact "
        "difference on this record and reports its percentiles. Use "
        "jump_scale.recommended_thresh with jump_scale.recommended_window. Do NOT set thresh "
        "from the mean, the std, or a remembered NTU figure: measured on the three project "
        "gauges, the workable threshold is 6.2, 13.8 and 75.7 NTU, and the last of those is "
        "33x that gauge's robust sigma. A run that set thresh=2-3 NTU flagged 2,865 rows "
        "across two years — every storm limb in the record — and could conclude nothing "
        "from them. "
        "WARNING — even correctly tuned, this tool is an aid, not a reliable classifier "
        "(§7.2, §7.6). There are only ~1-3 level_shift episodes per dataset, and flag_jumps still "
        "fires on sharp storm limbs and recovery ramps. At the recommended threshold expect "
        "~150 candidates over a two-year record, of which a handful at most are real. Treat "
        "its flags as 'look here' signals and triage them with describe_points / "
        "level_shift_context (step_sharpness is the discriminator) before deciding. "
        "Raise thresh to jump_scale's p99_9 if there are more candidates than you can "
        "triage; lower it only if the result is empty."
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
                    "Minimum difference between the mean of the preceding window and the mean "
                    "of the following window, in data units, to flag as a jump. There is no "
                    "portable default: take inspect_dataset's jump_scale.recommended_thresh "
                    "(the p99 of this series' own window-mean difference). Across the project "
                    "gauges that value is 6.2 / 13.8 / 75.7 NTU — a fixed number cannot serve "
                    "all three."
                ),
            },
            "window": {
                "type": ["string", "null"],
                "description": (
                    "Look-back/look-forward window as a pandas offset string. Required — do "
                    "not leave null. Use jump_scale.recommended_window ('6h'), and take thresh "
                    "from the SAME window's entry in jump_scale.by_window: the statistic is "
                    "window-dependent, so a threshold measured at 3h is wrong at 12h."
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
# Context tools (src/agent_tools/context.py)
#
# Only the two aggregators are exposed. The eight primitives behind them
# (slope_context, excursion_context, recovery_context, level_shift_context,
# flatness_context, neighbourhood_stats, gap_context, historical_context) are
# library functions: offering them individually would spend the 25-call budget
# on nine near-identical calls when describe_point returns all of them at once.
# ---------------------------------------------------------------------------

_DESCRIBE_POINTS = {
    "name": "describe_points",
    "description": (
        "Describes the SHAPE of the data around a list of timestamps — use this on a "
        "detector's flagged_datetimes to decide whether each flag is a sensor artifact or "
        "real water. A detector tells you WHERE a statistical rule fired; this tells you "
        "what the surrounding points look like, which is the evidence you need to choose "
        "delete / correct / keep. "
        "Returns one compact row per timestamp: value, robust_z (distance from the local "
        "median in robust sigmas), width_samples (how many samples the excursion spans at "
        "half height), peak_sharpness, fall_rise_ratio, samples_to_recover (how long until "
        "the series returns to its pre-event baseline), noise_ratio and step_sigmas_local "
        "(see below), and a reads_like hint. "
        "HOW TO READ IT — measured on this project's datasets: an artifact SPIKE has "
        "|robust_z| >= 3, width 2-3 samples and recovers in ~2 samples; a genuine STORM "
        "peak has |robust_z| ~1.3, width 7-25 samples and takes 12+ samples to recover. "
        "Width and recovery time are the reliable discriminators. Do NOT rely on "
        "fall_rise_ratio alone — it was measured and does not separate storms from spikes. "
        "SCAN THE noise_ratio COLUMN BEFORE JUDGING ANY ROW. It says how much more the "
        "data moves around that point than in a typical window of this record; "
        "step_sigmas_local says how far the point moves relative to its own neighbours "
        "(a real artifact ~17x, a false positive ~1.6x). When a run of rows all show a "
        "high noise_ratio and a low step_sigmas_local, you are looking at ONE noisy or "
        "fast-moving stretch, not that many separate sensor failures — the reads_like "
        "hint says 'noisy-stretch' for those. Judge and record the stretch with a single "
        "decision span; call noise_context for its bounds and for why it is busy. "
        "The reads_like label is a hint, not a verdict: you decide the action and justify it. "
        "This is one call for a whole detector output, so prefer it over describe_point when "
        "you have more than one timestamp to judge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "ats": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Timestamps to describe, ISO 8601 (e.g. '2024-06-01T03:00:00'). "
                    "Pass the flagged_datetimes from a detector result, or a subset of them. "
                    "A timestamp that is not in the series is reported in 'errors', not "
                    "silently rounded."
                ),
            },
            "max_points": {
                "type": "integer",
                "description": (
                    "Maximum timestamps to describe in this call. Default 100, hard ceiling "
                    "300. Any extras are reported in n_truncated and named in the message — "
                    "never dropped silently — so call again with the remainder if you need "
                    "them. Cost is about 90 tokens per point and does not grow per point "
                    "with batch size, so describing 100 points costs roughly what one "
                    "describe_point call on a single point costs six times over: measuring "
                    "a detector's whole output is cheap, and far cheaper than deleting a "
                    "reading you never looked at."
                ),
                "default": 100,
            },
            "window": {
                "type": "string",
                "description": (
                    "Neighbourhood size for the local statistics, as a pandas offset string. "
                    "Practical range '3h'-'12h'. Default '6h'."
                ),
                "default": "6h",
            },
        },
        "required": ["ats"],
    },
}

_DESCRIBE_POINT = {
    "name": "describe_point",
    "description": (
        "Full shape analysis of ONE timestamp — the detailed version of describe_points. "
        "Use it when a single point needs a careful decision and the compact row from "
        "describe_points was not enough. "
        "Returns eight nested measurement blocks: slope (gradient into and out of the point, "
        "plus the single-sample steps either side), excursion (width at half height, "
        "sharpness, size in robust sigmas), recovery (samples until the series returns to its "
        "pre-event baseline), level_shift (median before vs after, step in robust sigmas, and "
        "how long the new level held), flatness (length of the unchanged-value run — the "
        "stuck-sensor signature), neighbourhood (local median, robust sigma, robust z, "
        "percentile), gap (distance to the nearest missing run), and history (whether this "
        "series ever reached this level elsewhere, and in how many separate episodes — a level "
        "reached in 40 separate episodes is part of the regime, not an outlier). "
        "Every block also carries a plain-language 'message'. The reads_like label is a hint "
        "from these numbers, not a verdict."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "at": {
                "type": "string",
                "description": (
                    "The timestamp to describe, ISO 8601 (e.g. '2024-06-01T03:00:00'). "
                    "Must be a timestamp present in the series."
                ),
            },
            "n_before": {
                "type": "integer",
                "description": (
                    "Samples before the point used for the incoming gradient (the window "
                    "includes the point). Default 3 = 45 min on a 15-min grid, the scale "
                    "at which the fall/rise ratio separates a flush event from an artifact; "
                    "widening it past ~6 reverses that signal (see slope_context)."
                ),
                "default": 3,
            },
            "n_after": {
                "type": "integer",
                "description": (
                    "Samples after the point used for the outgoing gradient. Keep equal to "
                    "n_before. Default 3."
                ),
                "default": 3,
            },
            "window": {
                "type": "string",
                "description": (
                    "Neighbourhood size for the local statistics and flatness check, as a "
                    "pandas offset string. Practical range '3h'-'12h'. Default '6h'."
                ),
                "default": "6h",
            },
            "shift_window": {
                "type": "string",
                "description": (
                    "Window either side used for the before/after level comparison. "
                    "Practical range '12h'-'48h'. Default '24h'."
                ),
                "default": "24h",
            },
        },
        "required": ["at"],
    },
}

# ---------------------------------------------------------------------------
# Context primitives (2026-08-10)
#
# §7.3 gave schemas only to the two aggregators, on the grounds that nine
# near-identical tools would eat the 25-call cap. Three primitives are now exposed
# as well, because they are the ones that answer the §6 decisions the aggregate
# blurs: WIDTH and RECOVERY TIME are what actually separate a storm peak from a
# spike (§7.3's measured table), and level_shift is the weakest label, so the
# agent needs to interrogate it directly rather than trust `reads_like`. The other
# five stay library functions — describe_point already returns them all.
#
# A fourth, noise_context, was added 2026-08-13 for a different reason: it is the
# only one that measures the STRETCH rather than the point, and without it a run
# reads a noisy hour as fifty separate sensor failures. Its compact numbers ride
# along in every describe_points row precisely because that failure shows up at
# triage time, across a cluster of points, not on any single one of them.
#
# These take one timestamp and cost one call each, so they are for the handful of
# genuinely hard calls. describe_points remains the way to triage many points at
# once; reaching for a primitive on every flagged row will exhaust the cap.
# ---------------------------------------------------------------------------

_SLOPE_CONTEXT = {
    "name": "slope_context",
    "description": (
        "Compares the gradient RISING INTO a timestamp with the gradient FALLING OUT "
        "of it, over a short window (default 3 samples = 45 minutes either side). "
        "This is the test that separates a small first-flush event from a debris "
        "strike, and it is the one to reach for when a point has a high robust_z and "
        "a narrow width but you are not sure it is an artifact. "
        "READ fall_rise_ratio: a real flush RISES in one sample and DECAYS over "
        "3-5, so its fall gradient is gentler than its rise and the ratio sits "
        "around 0.75 or below. A debris strike falls as fast as it rose: ratio "
        "around 1.0. Measured on 31 flush events vs 14 confirmed artifacts, a "
        "'ratio < 0.8 means real water' rule catches 77% of flush events while "
        "wrongly sparing 14% of true artifacts. "
        "THE WINDOW IS CRITICAL AND THE SIGNAL REVERSES IF YOU WIDEN IT. At 45 min "
        "the separation is strong; by 90 min the decay has finished, the window "
        "fills with flat surroundings and the ordering flips. Leave n_before and "
        "n_after at 3 unless you know the excursion is longer, and never conclude "
        "'the ratio says artifact' from a wide window. "
        "Also returns the single-sample step into and out of the point, and the "
        "direction. Use it alongside width and recovery time, not instead of them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "at": {
                "type": "string",
                "description": "The timestamp to measure around, ISO 8601. Must exist in the series.",
            },
            "n_before": {
                "type": "integer",
                "description": (
                    "Samples used for the rising gradient, including the point itself. "
                    "Default 3 (45 min on a 15-min grid). Raising it past ~6 destroys "
                    "the signal — see the description."
                ),
                "default": 3,
            },
            "n_after": {
                "type": "integer",
                "description": (
                    "Samples used for the falling gradient, including the point itself. "
                    "Keep equal to n_before so the ratio compares like with like. Default 3."
                ),
                "default": 3,
            },
        },
        "required": ["at"],
    },
}

_EXCURSION_CONTEXT = {
    "name": "excursion_context",
    "description": (
        "Measures the WIDTH of the excursion containing one timestamp — the single most "
        "reliable separator between a spike and a storm peak (§7.3, measured: spikes are "
        "2-3 samples wide, storm peaks 7-25). "
        "Width is taken at half height: the run of samples that stay past the halfway mark "
        "between the local baseline and the peak. Also returns peak_sharpness (largest "
        "single-sample move as a fraction of the whole excursion — near 1.0 means it "
        "happened in one sample and is artifact-like; 0.1 means it built over many), the "
        "excursion size in robust sigmas, direction, start/end, and an `isolated` flag that "
        "is true at 1-2 samples wide. "
        "Use this when describe_points left a point genuinely ambiguous and width is the "
        "question. Do NOT use the rise-vs-fall gradient ratio to make this call — it was "
        "measured and it does not separate the two (§7.3)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "at": {
                "type": "string",
                "description": (
                    "Timestamp at or inside the excursion, ISO 8601 "
                    "(e.g. '2024-06-01T03:00:00'). Must exist in the series."
                ),
            },
            "baseline_window": {
                "type": "string",
                "description": (
                    "Window either side whose median defines the baseline the excursion is "
                    "measured from. Must be comfortably longer than the excursion itself or "
                    "the baseline is dragged into it. Practical range '6h'-'24h'. Default '12h'."
                ),
                "default": "12h",
            },
        },
        "required": ["at"],
    },
}

_RECOVERY_CONTEXT = {
    "name": "recovery_context",
    "description": (
        "Measures how long the series takes to return to its pre-event baseline after one "
        "timestamp — the second of the two measurements that actually separate a spike from "
        "a storm (§7.3: an artifact spike recovers in 1-2 samples, a storm takes 12-30, a "
        "level shift never recovers). "
        "Recovery requires n_confirm consecutive samples back inside tolerance_sigmas of the "
        "baseline, so one sample dipping through the band does not count. Returns whether it "
        "recovered at all, samples and minutes to recovery, and the baseline used. "
        "IMPORTANT — leave anchor='excursion' unless you have a specific reason. Mid-storm, "
        "the hours before the POINT are already storm, so a point-anchored baseline reports "
        "an instant recovery for an event nowhere near over (§7.3)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "at": {
                "type": "string",
                "description": "Timestamp to measure recovery from, ISO 8601. Must exist in the series.",
            },
            "baseline_window": {
                "type": "string",
                "description": (
                    "Window whose median is the baseline to return to, ending at the "
                    "excursion's onset when anchor='excursion'. Default '6h'."
                ),
                "default": "6h",
            },
            "anchor": {
                "type": "string",
                "enum": ["excursion", "point"],
                "description": (
                    "Where the baseline is measured from. 'excursion' (default, and almost "
                    "always correct) anchors at the onset of the excursion containing the "
                    "point. 'point' anchors at the point itself and will understate recovery "
                    "for anything mid-event."
                ),
                "default": "excursion",
            },
            "tolerance_sigmas": {
                "type": "number",
                "description": (
                    "How close to the baseline counts as recovered, in robust sigmas. "
                    "Raise it on a noisy series that never settles exactly. Default 2.0."
                ),
                "default": 2.0,
            },
            "max_search": {
                "type": "string",
                "description": (
                    "How far forward to look before giving up and reporting recovered=false. "
                    "Default '7D'."
                ),
                "default": "7D",
            },
        },
        "required": ["at"],
    },
}

_NOISE_CONTEXT = {
    "name": "noise_context",
    "description": (
        "Quantifies HOW NOISY the data is around one timestamp, and whether the point "
        "stands out from that noise. Reach for this the moment a detector returns a "
        "cluster of hits packed into a short stretch — that is the signature of one "
        "noisy period, not of many separate sensor failures, and this is the only tool "
        "that can tell you which you are looking at. "
        "WHY YOU NEED IT: every other measurement scores a point against the WHOLE "
        "record, so a point reads as extreme whether its neighbours are flat or "
        "thrashing. Measured on this project's data, the record-scaled step is 18.7 on "
        "a real artifact and 16.7 on a false positive — it cannot separate them at all. "
        "Rescaled against the local neighbourhood the same move reads 17.1 vs 1.6. "
        "READ TWO NUMBERS. `noise_ratio`: how much more the data moves here than in a "
        "typical window of this record (1.0 = ordinary, 10 = ten times as busy). "
        "`point_step_sigmas_local`: how far the point moves relative to how far its own "
        "neighbours are moving — this is the discriminator. A real artifact jumps much "
        "further than the samples around it (median 17x); a false positive is doing "
        "what everything near it is doing (median 1.6x). "
        "THE RULE, measured: when noise_ratio > 3 AND point_step_sigmas_local < 5, the "
        "point is a false positive about four times in five (it spares 76.7% of false "
        "positives while losing 3.0% of genuine spikes). `point_unremarkable_here` is "
        "that test, precomputed. "
        "`variation_kind` splits the two reasons a stretch can be busy: 'noise-like' "
        "means the series reverses direction constantly and the SENSOR is noisy here; "
        "'directional' means it is climbing or falling steadily, which is real water "
        "moving fast (a storm limb). Neither should be deleted point by point, but only "
        "the first is a data-quality problem at all. "
        "`episode_start` / `episode_end` bound the noisy stretch, so you can write ONE "
        "decision span covering it instead of one decision per flagged row."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "at": {
                "type": "string",
                "description": "The timestamp to measure around, ISO 8601. Must exist in the series.",
            },
            "window": {
                "type": "string",
                "description": (
                    "Half-width of the neighbourhood whose noise is measured, as a pandas "
                    "offset string. LEAVE THIS AT '90min'. It was swept: the rule above "
                    "spares 83.7 / 76.7 / 72.4 / 61.0% of false positives at +/-1 / 1.5 / "
                    "2 / 3 h. Widen it and the window fills with calm surroundings, the "
                    "local scale collapses back toward the record-wide one, and the "
                    "measurement degrades into the global one it exists to replace."
                ),
                "default": "90min",
            },
            "elevated_ratio": {
                "type": "number",
                "description": (
                    "How many times the record-typical noise counts as an elevated stretch. "
                    "Sets both `noise_regime` and the bounds of the reported episode. "
                    "Default 3.0."
                ),
                "default": 3.0,
            },
        },
        "required": ["at"],
    },
}

_LEVEL_SHIFT_CONTEXT = {
    "name": "level_shift_context",
    "description": (
        "Compares the level before one timestamp with the level after it, and measures how "
        "long the new level actually held. Returns median before vs after, the step in robust "
        "sigmas of the pre-window, step_sharpness, and hold_minutes. "
        "Use it on flag_jumps hits, which are the hardest call in this project: flag_jumps "
        "fires on every sharp change and most sharp changes in turbidity are storm limbs. "
        "SHARPNESS is the discriminator that works — a recalibration moves most of its "
        "magnitude in one sample, a storm spreads it over hours. "
        "Read hold_minutes as a duration, not a permanence test: a few samples means it was a "
        "spike, hours-to-days means a genuinely shifted segment, running to the end of the "
        "record means a permanent recalibration. "
        "Even with these numbers a flash-flood onset is sharp and sustained and cannot be "
        "told from a real step in one series — this measurement narrows the question, it does "
        "not settle it. Default to KEEP (§6)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Column name that holds the measurement values. Default 'value'.",
                "default": "value",
            },
            "at": {
                "type": "string",
                "description": "The candidate step timestamp, ISO 8601. Must exist in the series.",
            },
            "window": {
                "type": "string",
                "description": (
                    "How much series either side of the point forms the before and after "
                    "levels. Practical range '6h'-'48h'. Default '12h'."
                ),
                "default": "12h",
            },
            "skip_samples": {
                "type": "integer",
                "description": (
                    "Samples either side of the point excluded from both medians, so the "
                    "transition itself does not pollute the levels it is being measured "
                    "against. Default 2."
                ),
                "default": 2,
            },
            "hold_sigmas": {
                "type": "number",
                "description": (
                    "How far the series may wander from the new median while still counting "
                    "as holding that level, in robust sigmas. Default 3.0."
                ),
                "default": 3.0,
            },
        },
        "required": ["at"],
    },
}

# ---------------------------------------------------------------------------
# Action tools
# ---------------------------------------------------------------------------

_CORRECT_LEVEL_SHIFT = {
    "name": "correct_level_shift",
    "description": (
        "CORRECT a level shift by shifting its window back by the step measured at its "
        "own two edges. THIS IS THE RIGHT ACTION FOR A LEVEL SHIFT, not delete: a shift "
        "is an OFFSET, so the water underneath moved normally and the shape inside the "
        "window is real data sitting at the wrong height. Deleting it throws away hours "
        "of good record; subtracting the offset recovers it. Measured on this project's "
        "data, correcting the injected shift took the interior error from 6.95 FNU to "
        "0.78 FNU against the true water, and changed zero rows outside the window. "
        "Take `start` and `end` from find_shift_windows. Both edges are measured and "
        "reported separately: a clean level shift steps up and back down by the same "
        "amount, so if `edges_agree` comes back false the window may be a storm rather "
        "than an offset — check it before relying on the correction. After calling this, "
        "record the segment as verdict 'anomaly', anomaly_type 'level_shift', action "
        "'correct'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {
                "type": "string",
                "description": "First timestamp of the shifted window, ISO 8601.",
            },
            "end": {
                "type": "string",
                "description": "Last timestamp of the shifted window, ISO 8601.",
            },
            "edge_window": {
                "type": "string",
                "description": (
                    "How much series either side of each edge to average when measuring "
                    "the step. Default '2h' — wide enough to average out 5-minute noise, "
                    "narrow enough not to reach into a storm beyond the window."
                ),
            },
        },
        "required": ["start", "end"],
    },
}


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

_RAMP_CONTEXT = {
    "name": "ramp_context",
    "description": (
        "Measures the SHAPE ABOVE the 90-minute scale: how long the series took to climb "
        "to this point, and how long it took to come back down. slope_context looks 45 "
        "minutes either side, which separates a debris strike from a small flush; this "
        "looks hours, which is what separates a storm PEAK from both. A point at the top "
        "of a climb lasting hours is the top of something the water was already doing; an "
        "artifact is not climbing to anything. Reach for it when a point is narrow and "
        "high-z and you are about to delete it but the surrounding hours look like they "
        "were going somewhere. Returns rise/fall duration in minutes, magnitude, and a "
        "monotonic_fraction (share of steps moving the expected way; 1.0 is a clean climb, "
        "0.5 is noise). It reports evidence and does not decide: on this project's "
        "INJECTED data the two populations overlap heavily, because spikes are injected on "
        "top of rising limbs about as often as real water sits on them, so weigh a ramp "
        "alongside width, recovery and noise rather than on its own."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {"type": "string", "default": "value",
                      "description": "Column holding the measurements. Default 'value'."},
            "at": {"type": "string",
                   "description": "Timestamp to measure around, ISO 8601. Must exist in the series."},
            "max_window": {"type": "string", "default": "6h",
                           "description": ("How far either side to look for the foot of the "
                                           "ramp. Default '6h'; widen only for a very slow "
                                           "river.")},
        },
        "required": ["at"],
    },
}

_PRECIP_CONTEXT_POINTS = {
    "name": "precip_context_points",
    "description": (
        "WAS IT RAINING? Checks a LIST of timestamps against nearby rainfall — the one "
        "piece of evidence that comes from OUTSIDE the turbidity series. Turbidity rises "
        "because rain washes sediment in, so an excursion with rain behind it has a "
        "physical cause and an artifact does not. USE THIS ON EVERY POINT YOU ARE ABOUT TO "
        "CALL A SPIKE, including the ones that look clear-cut — that is the whole point: "
        "the calls that look obvious from the series alone are exactly the ones this can "
        "overturn. Points you are NOT calling spikes do not need it. Reports several lag "
        "windows before each point (0-1h, 1-3h, 3-12h) because rain leads turbidity by an "
        "amount that depends on the catchment. TWO LIMITS RIDE WITH EVERY ANSWER: the "
        "nearest station with full coverage is about 31 km away, so PRESENCE of rain is "
        "strong evidence and ABSENCE is weak — a summer storm cell can miss the station "
        "entirely. And if no data has been pulled the result says so explicitly; that is "
        "never the same as 'it did not rain'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ats": {"type": "array", "items": {"type": "string"},
                    "description": ("Timestamps to check, ISO 8601. Pass every timestamp "
                                    "you intend to give a spike verdict.")},
            "max_points": {"type": "integer", "default": 300,
                           "description": "Maximum timestamps per call. Default 300."},
        },
        "required": ["ats"],
    },
}

_PRECIP_CONTEXT = {
    "name": "precip_context",
    "description": (
        "Rainfall around ONE timestamp — the single-point version of "
        "precip_context_points. Use it when a single decision turns on whether it rained; "
        "for auditing a set of spike candidates use the batch form instead. Returns rain "
        "totals over 0-1h, 1-3h and 3-12h before the point plus 3h after, and names the "
        "station and its distance so you can weigh how much the answer is worth."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "at": {"type": "string", "description": "Timestamp to check, ISO 8601."},
        },
        "required": ["at"],
    },
}

_FIND_SHIFT_WINDOWS = {
    "name": "find_shift_windows",
    "description": (
        "TURNS JUMP EDGES INTO LEVEL-SHIFT WINDOWS. flag_jumps marks the TRANSITION — one "
        "row where the level changed — but a level shift is the whole span that sits at "
        "the wrong level, and that is what has to be reported and corrected. Scoring an "
        "edge against a window is why level_shift recall was 0.9%: the detector found "
        "every onset and still scored zero. Pass this flag_jumps' flagged_datetimes and it "
        "pairs them into candidate windows, measures each one, and ranks them. Measured: "
        "row recall goes from ~0.5% to 76%. Each window reports interior_sigmas (how far "
        "the inside sits from its surroundings) and the sharpness of both edges (1.0 = the "
        "level moved in ONE sample; 0.1 = it ramped over hours). BOTH matter: a storm is "
        "also elevated between two jumps, so elevation alone cannot separate them — the "
        "biggest storm on one dataset scored higher than the real shift. Sharp edges are "
        "what distinguish a recalibration from weather. If you accept a window, write ONE "
        "decision span covering the whole thing, not just its edges."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {"type": "string", "default": "value",
                      "description": "Column holding the measurements. Default 'value'."},
            "ats": {"type": "array", "items": {"type": "string"},
                    "description": "OPTIONAL, and you should normally OMIT it. Left out, "
                                   "every timestamp flag_jumps flagged is read straight "
                                   "from the flag history, which is what you want. Pass a "
                                   "list only to ask about specific edges — a subset that "
                                   "happens to leave out a real shift's two edges makes "
                                   "that shift invisible, and the result looks normal."},
            "min_hours": {"type": "number", "default": 1.0,
                          "description": "Shortest window to consider, in hours."},
            "max_hours": {"type": "number", "default": 48.0,
                          "description": "Longest window to consider, in hours."},
        },
        "required": ["ats"],
    },
}

_SHIFT_WINDOW_CONTEXT = {
    "name": "shift_window_context",
    "description": (
        "Measures ONE candidate level-shift window: the interior mean against its "
        "surroundings, in robust sigmas, plus how abrupt each edge is. The single-window "
        "version of find_shift_windows — use it when you already know the span you want "
        "judged. reads_like is 'level-shift-like' (elevated AND both edges abrupt), "
        "'event-like' (elevated but at least one edge ramps — a storm), or 'not-shifted'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {"type": "string", "default": "value",
                      "description": "Column holding the measurements. Default 'value'."},
            "start": {"type": "string", "description": "Window start, ISO 8601."},
            "end": {"type": "string", "description": "Window end, ISO 8601."},
        },
        "required": ["start", "end"],
    },
}

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
    # Context
    _DESCRIBE_POINTS,
    _DESCRIBE_POINT,
    _SLOPE_CONTEXT,
    _EXCURSION_CONTEXT,
    _RECOVERY_CONTEXT,
    _LEVEL_SHIFT_CONTEXT,
    _NOISE_CONTEXT,
    _RAMP_CONTEXT,
    _FIND_SHIFT_WINDOWS,
    _SHIFT_WINDOW_CONTEXT,
    # Outside evidence (§7.6)
    _PRECIP_CONTEXT_POINTS,
    _PRECIP_CONTEXT,
    # Action
    _CORRECT_LEVEL_SHIFT,
    _IMPUTE_ROLLING,
]

# Convenience: look up a schema by tool name
TOOL_SCHEMA_BY_NAME: dict[str, dict] = {s["name"]: s for s in TOOL_SCHEMAS}
