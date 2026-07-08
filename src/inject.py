"""Synthetic anomaly injection into clean segments.

Injects the five failure types (spike, plateau, level_shift, gap, drift) at
recorded locations, writes ground-truth label files, uses a fixed random seed,
and builds three contamination levels (~3%, 7%, 12%). See CLAUDE.md §5, §9.

Implemented in Phase 1 (see CLAUDE.md §12).
"""
