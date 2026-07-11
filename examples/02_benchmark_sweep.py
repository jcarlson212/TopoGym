"""Sweep a benchmark's certified metadata — the experiment-planning view."""

import topogym.benchmarks as bench

for collection in bench.benchmark_names():
    print(f"\n=== {collection} ===")
    header = f"{'env':26s} {'betti_z2':14s} {'genus':6s} {'sccs':5s} note"
    print(header)
    print("-" * len(header))
    for row in bench.benchmark_metadata(collection):
        genus = row["genus"] if row["genus"] is not None else (
            f"x{row['demigenus']}" if row.get("demigenus") is not None else "-"
        )
        asym = row["asymmetry"]
        note = ""
        if asym["n_absorbing_sccs"]:
            note = f"{asym['n_absorbing_sccs']} absorbing region(s)!"
        elif not asym["is_symmetric"]:
            note = "directed but safe"
        if row["h1_torsion"]:
            note += " torsion: " + ",".join(row["h1_torsion"])
        print(
            f"{row['name']:26s} {str(row['betti_z2']):14s} "
            f"{str(genus):6s} {asym['n_sccs']:<5d} {note}"
        )
