"""Seed splits for rigorous SECE development — do not cross-contaminate."""

# Locked after Task B — never use for Phase 4+ architecture decisions until final confirm
HELDOUT_SEEDS = [0, 1, 2, 7, 13, 99, 123, 2024, 2026]
CLEAN_SEEDS = HELDOUT_SEEDS

# Use for architecture tuning (Phase 4 dev runs)
DEV_SEEDS = [3, 4, 5, 6, 8, 9, 10, 11, 14]

# Contaminated by Phase 1/2/3 iteration — dev/debug only
CONTAMINATED_SEEDS = {42}
