"""Why does flagPlateau flag 6,747 rows where flagConstants flags 119? (§13)

Run Q called flag_plateau(min_length='1h') and it flagged 3.2% of the record, which then
masked those rows out of flagJumps (144 jumps -> 76) and cost the run its level shift.
The wrapper omits min_jump, so SaQC requires NO minimum offset: any ~flat stretch of an
hour qualifies however small its displacement.

Run from a file, never a heredoc — flagPlateau's workers re-import __main__ (§7.1).
"""
import warnings

import numpy as np
import pandas as pd
import saqc



def main() -> None:

    v = pd.read_csv("data/injected/01467200/l1/01467200_l1.csv",
                    parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv("data/injected/01467200/l1/01467200_l1_labels.csv",
                      parse_dates=["datetime"]).set_index("datetime")
    truth_all = lab.index[lab.anomaly_type == "plateau"]
    # a 30k-row slice around the labelled plateaus: flagPlateau is ~minutes on 210k rows
    lo = truth_all.min() - pd.Timedelta("7d")
    hi = truth_all.min() + pd.Timedelta("60d")
    v = v.loc[lo:hi]
    truth = set(truth_all[(truth_all >= lo) & (truth_all <= hi)])
    robust_sigma = 1.4826 * float((v - v.median()).abs().median())
    print(f"record: {len(v):,} rows, robust_sigma {robust_sigma:.2f} FNU, "
          f"{len(truth)} labelled plateau rows\n")

    print(f"{'min_jump':>10}{'flagged':>10}{'% of record':>13}{'true plateau found':>20}")
    for mj in (None, 0.5, 1.0, 2.0, 5.0):
        kw = {"min_length": "1h"}
        if mj is not None:
            kw["min_jump"] = mj
        try:
            out = saqc.SaQC(pd.DataFrame({"value": v})).flagPlateau("value", **kw)
        except ValueError as exc:
            print(f"{str(mj):>10}   raised {type(exc).__name__}: {exc}")
            continue
        hit = set(v.index[(out.flags["value"] > 0).to_numpy()])
        print(f"{str(mj):>10}{len(hit):>10}{len(hit)/len(v):>12.1%}"
              f"{len(hit & truth):>12} / {len(truth)}")


if __name__ == "__main__":
    main()
