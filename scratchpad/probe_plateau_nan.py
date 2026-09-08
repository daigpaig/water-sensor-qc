"""Is flagPlateau flagging NaN RUNS as plateaus? (§13)

min_jump does not explain run Q's 6,747 flags (a sweep changed almost nothing, and
min_jump=None flagged the FEWEST). But 6,747 sits suspiciously close to the record's
7,222 NaN rows. Run from a file behind a __main__ guard — flagPlateau's workers
re-import the module (§7.1).
"""
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")


def main() -> None:
    v = pd.read_csv("data/turbidity/injected/01467200/l1/01467200_l1.csv",
                    parse_dates=["datetime"]).set_index("datetime")["value"]
    nan_mask = v.isna()
    print(f"record {len(v):,} rows, {int(nan_mask.sum()):,} NaN\n")

    out = saqc.SaQC(pd.DataFrame({"value": v})).flagPlateau("value", min_length="1h")
    hit = (out.flags["value"] > 0).to_numpy()
    print(f"flagPlateau(min_length='1h') flagged {int(hit.sum()):,} rows")
    print(f"   of which NaN in the input : {int((hit & nan_mask.to_numpy()).sum()):,}")
    print(f"   of which real readings    : {int((hit & ~nan_mask.to_numpy()).sum()):,}")

    # and on the same record with the gaps dropped
    dense = v.dropna()
    out2 = saqc.SaQC(pd.DataFrame({"value": dense})).flagPlateau("value", min_length="1h")
    hit2 = int((out2.flags["value"] > 0).sum())
    print(f"\nsame call on the series with NaN rows REMOVED ({len(dense):,} rows): "
          f"{hit2:,} flagged")


if __name__ == "__main__":
    main()
