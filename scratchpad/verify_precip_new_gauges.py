"""Verify the rain-station candidates for 02054550 and 040851385 (CLAUDE.md §7.7, §9.4).

The site catalog's begin/end describes the SITE, not its instantaneous series, so a
station can advertise a decade of record while only reporting unit values recently.
This measures what each candidate ACTUALLY returns over the project window, so
`pull_precip.PRECIP_STATIONS` is populated from evidence rather than the catalog.

    python scratchpad/verify_precip_new_gauges.py
"""
from verify_precip import check   # noqa: E402  (same directory)

CANDIDATES = {
    "02054550": [
        ("371520080015100", "MET STN Hidden Valley, Roanoke", 7.0),
        ("371824080002600", "MET STN Rt 117, Roanoke", 9.2),
        ("371518079591700", "MET STN Shrine Hill Park, Roanoke", 10.7),
        ("371709079580800", "MET STN Fire Stn 5, Roanoke", 12.1),
    ],
    "040851385": [
        ("04085078", "Dutchman Creek at Hansen Rd, Ashwaubenon", 7.9),
        ("04072150", "Duck Creek near Howard", 9.5),
        ("040850684", "Ashwaubenon Creek at Grant St", 11.7),
        ("04085108", "East River at CTH ZZ nr Greenleaf", 18.6),
    ],
}

for turb, cands in CANDIDATES.items():
    print(f"\n=== precipitation near turbidity gauge {turb} ===")
    for site, name, km in cands:
        r = check(site)
        if "error" in r:
            print(f"  {km:5.1f} km  {site:<16} {name:<42} -> {r['error']}")
            continue
        print(f"  {km:5.1f} km  {site:<16} {name:<42} -> "
              f"{r['n']:,} obs @ {r['dt_min']:.0f}min, {r['first']}..{r['last']} "
              f"({r['span_days']:.0f}d, {r['complete_pct']}% complete), "
              f"{r['total_in']:.0f} in total, {r['approved_pct']:.0f}% approved")
