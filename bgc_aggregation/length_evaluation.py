"""Paired group-bootstrap evidence for length extrapolation, without ESM dependencies."""

import csv
import random
from collections import defaultdict
from pathlib import Path

from .common import MAX_RESIDUES, stable_seed


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def paired_group_bootstrap(rows, replicates=2000, seed=42):
    """Resample whole groups; retain protein-weighted means and paired baselines."""
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if not rows:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    totals = [(sum(r["model_raw_mse"] for r in group),
               sum(r["baseline_raw_mse"] for r in group), len(group))
              for _, group in sorted(groups.items())]
    model = sum(t[0] for t in totals) / len(rows)
    baseline = sum(t[1] for t in totals) / len(rows)
    result = {"proteins": len(rows), "groups": len(groups),
              "model_raw_mse": model, "baseline_raw_mse": baseline,
              "raw_mse_difference": model - baseline,
              "relative_mse_reduction": 1 - model / baseline if baseline > 0 else None,
              "model_cosine_distance": sum(r["model_cosine_distance"] for r in rows) / len(rows),
              "baseline_cosine_distance": sum(r["baseline_cosine_distance"] for r in rows) / len(rows),
              "raw_mse_difference_ci95": None, "relative_mse_reduction_ci95": None,
              "bootstrap_replicates": replicates, "bootstrap_seed": seed,
              "evidence": "insufficient_groups"}
    if len(groups) < 2:
        return result
    rng = random.Random(seed)
    differences, reductions = [], []
    for _ in range(replicates):
        sampled = [totals[rng.randrange(len(totals))] for _ in totals]
        size = sum(t[2] for t in sampled)
        m, b = sum(t[0] for t in sampled) / size, sum(t[1] for t in sampled) / size
        differences.append(m - b)
        if b > 0:
            reductions.append(1 - m / b)
    interval = [quantile(differences, p) for p in (0.025, 0.975)]
    result["raw_mse_difference_ci95"] = interval
    if len(reductions) == replicates:
        result["relative_mse_reduction_ci95"] = [quantile(reductions, p) for p in (0.025, 0.975)]
    result["evidence"] = "improved" if interval[1] < 0 else "worse" if interval[0] > 0 else "inconclusive"
    return result


def length_report(rows, experiment, training_inputs, split, replicates=2000, seed=42):
    # Alternate views never change the primary length-extrapolation conclusion.
    rows = [r for r in rows if r["view"] == 0]
    if len({r["parent_id"] for r in rows}) != len(rows):
        raise ValueError("Length evaluation requires one deployment-view observation per natural protein")
    cutoff, maximum_chunks = experiment["length_cutoff"], training_inputs["max_chunks"]
    cohorts = {
        "short": [r for r in rows if r["length"] < cutoff],
        "near_boundary_short": [r for r in rows if 1024 <= r["length"] < cutoff],
        "long": [r for r in rows if r["length"] >= cutoff],
        "long_within_chunk_range": [r for r in rows if r["length"] >= cutoff and r["chunks"] <= maximum_chunks],
        "long_beyond_chunk_range": [r for r in rows if r["length"] >= cutoff and r["chunks"] > maximum_chunks],
    }

    def summarize(selected, name):
        result = {"overall": paired_group_bootstrap(selected, replicates, stable_seed(seed, name)),
                  "by_family": {}}
        for family in sorted({r["family"] for r in selected}):
            result["by_family"][family] = paired_group_bootstrap(
                [r for r in selected if r["family"] == family], replicates, stable_seed(seed, [name, family]))
        return result

    # 1500 cutoff gives 1-512, 513-1023, 1024-1499, 1500-1536,
    # 1537-1792, 1793-2046. Include custom cutoffs without overlapping bins.
    boundaries = sorted({b for b in (512, 1023, cutoff - 1, ((cutoff - 2) // 512 + 1) * 512,
                                     1792, MAX_RESIDUES) if 1 <= b <= MAX_RESIDUES})
    bins, lower = {}, 1
    for upper in boundaries:
        name = f"{lower}-{upper}"
        bins[name] = {"min_length": lower, "max_length": upper,
                      **summarize([r for r in rows if lower <= r["length"] <= upper], name)}
        lower = upper + 1
    report = {"protocol": experiment["protocol"], "length_cutoff": cutoff,
              "teacher_max_length": MAX_RESIDUES, "split": split,
              "training_inputs": training_inputs, "bootstrap_replicates": replicates,
              "bootstrap_seed": seed, "ci_method": "paired percentile bootstrap of groups, protein-weighted means",
              "cohorts": {name: summarize(selected, name) for name, selected in cohorts.items()},
              "length_bins": bins,
              "interpretation": "Improvement refers to raw MSE versus weighted chunk means on this held-out cohort. "
                                "Intervals are per cohort, not adjusted for multiple comparisons. Results do not "
                                "establish generalization beyond the teacher limit or biological accuracy."}
    return report, cohorts


def write_length_table(report, output):
    fields = ["section", "cohort", "family", "proteins", "groups", "model_raw_mse", "baseline_raw_mse",
              "model_cosine_distance", "baseline_cosine_distance", "raw_mse_difference",
              "difference_ci95_low", "difference_ci95_high", "relative_mse_reduction",
              "reduction_ci95_low", "reduction_ci95_high", "evidence"]
    with (Path(output) / "length_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for section in ("cohorts", "length_bins"):
            for name, summaries in report[section].items():
                for family, summary in [("ALL", summaries["overall"]), *summaries["by_family"].items()]:
                    row = {"section": section, "cohort": name, "family": family}
                    if summary:
                        row.update(summary)
                        for prefix, key in (("difference", "raw_mse_difference_ci95"),
                                            ("reduction", "relative_mse_reduction_ci95")):
                            ci = summary[key] or (None, None)
                            row.update({prefix + "_ci95_low": ci[0], prefix + "_ci95_high": ci[1]})
                    else:
                        row.update(proteins=0, groups=0, evidence="no_samples")
                    writer.writerow(row)


def plot_length_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind, filename in (("error", "length_error.svg"), ("gain", "length_gain.svg")):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        for axis, family in zip(axes, ("ALL", "NRPS", "PKS")):
            values = []
            for summary in report["length_bins"].values():
                value = summary["overall"] if family == "ALL" else summary["by_family"].get(family)
                if value:
                    values.append(((summary["min_length"] + summary["max_length"]) / 2, value))
            x = [v[0] for v in values]
            if kind == "error":
                axis.plot(x, [v[1]["model_raw_mse"] for v in values], "o-", label="Aggregator")
                axis.plot(x, [v[1]["baseline_raw_mse"] for v in values], "o--", label="Weighted mean")
                axis.set_ylabel("Mean raw MSE")
            else:
                valid = [(position, v) for position, v in values if v["relative_mse_reduction"] is not None]
                axis.plot([v[0] for v in valid], [v[1]["relative_mse_reduction"] for v in valid], "o-",
                          label="Relative MSE reduction")
                for position, v in valid:
                    ci = v["relative_mse_reduction_ci95"]
                    if ci:
                        axis.vlines(position, ci[0], ci[1], color="C0", alpha=0.6)
                axis.axhline(0, color="grey", linewidth=0.8)
                axis.set_ylabel("1 - aggregator MSE / baseline MSE")
            axis.axvline(report["length_cutoff"], color="black", linestyle=":", label="Length cutoff")
            chunk_boundary = report["training_inputs"]["max_chunks"] * 512
            if chunk_boundary < MAX_RESIDUES:
                axis.axvline(chunk_boundary, color="grey", linestyle="--", label="Training chunk limit")
            axis.set_title(family)
            axis.set_xlabel("Protein length (aa; bin midpoint)")
            axis.legend(fontsize=7)
        figure.suptitle(f"{report['split']} / training views: {report['training_inputs']['views']}")
        figure.tight_layout()
        figure.savefig(Path(output) / filename)
        plt.close(figure)
