# Legacy 15-minute datasets (retired 2026-08-18)

This directory holds the project's **original 15-minute** turbidity bases and the
nine injected datasets built from them. They were retired when the project moved
to **5-minute** bases, and are kept for provenance and for re-checking any
measurement in `CLAUDE.md` that was made against them.

**Nothing here is live.** `src.datasets.inject` globs `data/raw/approved/` and only
that (CLAUDE.md §9), and `data/injected/` is where the current datasets live, so
no code path reaches into this folder. Do not point one at it — if you need these
series for a comparison, copy them out rather than widening a glob.

## Layout

```
data/legacy_15min/
├── raw/
│   ├── approved/      # the three retired 15-min approved bases (gitignored, regenerable)
│   └── provisional/   # unvetted 15-min pulls, audit scratch only (gitignored)
└── injected/          # the nine §5 triples built from those bases (TRACKED — ground truth)
    └── <gauge>/l<level>/{<name>.csv,<name>_labels.csv,<name>_manifest.json}
```

The raw CSVs are gitignored like `data/raw/` because they are regenerable from
NWIS. The injected datasets stay tracked: their label files are generated ground
truth that a fixed seed alone does not make recoverable once the injector changes.

## The retired gauges

All three are 15-min, 100% USGS-approved, ≥95% complete, and calm, over
2023-07-01 → 2025-07-01:

| Site | Name | Regime |
| --- | --- | --- |
| `03447687` | French Broad R nr Fletcher, NC | S. Appalachia; median ~8 FNU |
| `02198840` | Savannah R at I-95 nr Port Wentworth, GA | tidal river; median ~13 FNU |
| `08041770` | LNVA Canal at Beaumont, TX | managed canal; median ~28 FNU |

To regenerate the raw pulls:

```
python -m src.datasets.pull_usgs --sites 03447687 02198840 08041770 \
    --start 2023-07-01 --end 2025-07-01 --outdir data/legacy_15min/raw/approved
```

## Why they were retired

The project moved to 5-minute sampling. Cadence is not cosmetic here: several
§7 measurements are expressed in *samples* rather than time (`flag_spike_unilof`'s
`n≈20` neighbourhood, `slope_context`'s 3-sample window, `noise_context`'s ±90-min
block), so a 3× denser record changes what each of those windows spans. Any number
in `CLAUDE.md` §7.2, §7.3 or §7.4 that was measured on these series is a **stale
starting point** for the 5-min bases, not a tuned value.
