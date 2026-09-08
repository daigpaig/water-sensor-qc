"""Is flagPlateau actually a LEVEL-SHIFT detector on this data? (§13)

flagPlateau's target is "a segment displaced from its surroundings, values need not be
constant". §9 injects a level_shift as a BOUNDED window (4-24h) at a different level,
because a permanent step cannot be expressed in the §5 label contract. Those are close
to the same description — and flagPlateau finds 0 of the injected PLATEAUS, which are
frozen at the local level and so invisible to it by construction (inject.py: the level
defaults to the reading at `start`).

Run from a file behind a __main__ guard — flagPlateau's workers re-import it (§7.1).
"""
import glob
import warnings

import pandas as pd
import saqc

warnings.filterwarnings("ignore")


def main() -> None:
    print(f"{'dataset':<16}{'flagged':>9}{'of level_shift':>16}{'of plateau':>13}"
          f"{'of spike':>10}{'of gap':>9}")
    for path in sorted(glob.glob("data/turbidity/injected/*/l*/*_l?.csv")):
        stem = path.rsplit("/", 1)[-1][:-4]
        v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
        lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                          parse_dates=["datetime"]).set_index("datetime")
        try:
            out = saqc.SaQC(pd.DataFrame({"value": v})).flagPlateau(
                "value", min_length="1h")
        except ValueError as exc:
            print(f"{stem:<16}   raised {type(exc).__name__}: {exc}")
            continue
        hit = set(v.index[(out.flags["value"] > 0).to_numpy()])
        row = f"{stem:<16}{len(hit):>9}"
        for kind in ("level_shift", "plateau", "spike", "gap"):
            truth = set(lab.index[lab.anomaly_type == kind])
            frac = f"{len(hit & truth)}/{len(truth)}" if truth else "-"
            row += f"{frac:>16}" if kind == "level_shift" else (
                f"{frac:>13}" if kind == "plateau" else
                f"{frac:>10}" if kind == "spike" else f"{frac:>9}")
        print(row, flush=True)


if __name__ == "__main__":
    main()
