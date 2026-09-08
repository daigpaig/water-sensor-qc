# `data/comparison/` — one turbidity series per gauge, in both approval states

Regenerate with `python -m src.datasets.pull_comparison`. The CSVs are gitignored (they are
re-downloadable); this README is not.

## Read this first: it is a split, not a pair

NWIS serves the **current** state of each record. When USGS approves a period, the
provisional values it used to carry are **overwritten in place** — there is no public
archive of superseded provisional values and the instantaneous-values service has no
revision / as-of parameter. **The same timestamps therefore cannot be retrieved in two
states.** A genuine row-for-row before/after pair would have to have been captured live,
before approval, and kept.

What a live record does give you is a moving **approval boundary**. Everything older
carries qualifier `A` — approved, meaning USGS record processing (TM 1-D3) has applied
fouling and calibration-drift corrections prorated between field visits and deleted
clearly-erroneous data. Everything newer carries `P` — provisional, essentially as the
sensor reported it. Same site, same sensor, same 15-min grid, opposite sides of the
processing step.

So compare the two sides as **distributions**, not as counterparts. They are different
calendar periods, so seasonality confounds a naive comparison; each manifest records a
`season_matched_approved_window` (the same calendar dates one year earlier) as the fairer
slice of the approved side to compare against.

## The three gauges

Chosen from a 24-gauge screen (`scratchpad/screen_prov_vs_approved.py`) for geographic and
turbidity-regime spread, a consistent 15-min step, and dense coverage on *both* sides.

| Site | Station | Geography / regime | Approved | Provisional | Boundary |
| --- | --- | --- | --- | --- | --- |
| `03447687` | French Broad R nr Fletcher, NC | S. Appalachian mountain river; moderate (median ~8 FNU) | 94,074 rows · 1018 d · 96.2% | 10,078 rows · 106 d · 98.9% | 2026-04-14 |
| `02198840` | Savannah R at I-95 nr Port Wentworth, GA | Atlantic tidal coastal plain; moderate-high (~13 FNU) | 84,731 rows · 894 d · 98.8% | 21,711 rows · 231 d · 98.0% | 2025-12-10 |
| `06818000` | Missouri R at St Joseph, MO | Great Plains large river; high (~24 FNU) | 75,935 rows · 926 d · 85.4% | 17,188 rows · 198 d · 90.4% | 2026-01-12 |

All three split cleanly: zero approved rows after the boundary, zero timestamp overlap.

The project's third injection base `08041770` (LNVA Canal, TX) **cannot** appear here — its
record ends 2025-11-19 with zero provisional rows. `06818000` stands in; it was retired as
an *injection base* for 14% missing rows, but CLAUDE.md §9 notes those are overwhelmingly
isolated single-sample dropouts, which does not disqualify it for a distribution comparison.

## Layout

```
data/comparison/<site>/
├── <site>_turbidity_63680_approved.csv      datetime, value, qualifier
├── <site>_turbidity_63680_provisional.csv   datetime, value, qualifier
└── <site>_comparison_manifest.json          boundary, per-side stats, caveat
```

`datetime` is ISO-8601 **UTC, timezone-naive**; `value` is turbidity in FNU (param `63680`);
`qualifier` is the USGS code. Approved-side qualifiers include the modified approved codes
`A e` (estimated), `A, >` (over-range) and `A, R` (revised) — USGS approved those readings,
so they stay on the approved side. As in `data/raw/`, real gaps appear as **missing rows**,
not NaN, until something re-grids onto the 15-min grid.
