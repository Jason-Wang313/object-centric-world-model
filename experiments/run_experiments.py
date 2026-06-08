"""Run controlled object-centric Best-of-N experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from object_centric_best_of_n.audit import write_claim_status, write_final_audit
from object_centric_best_of_n.envs import make_scene
from object_centric_best_of_n.learned_model import train_and_evaluate
from object_centric_best_of_n.metrics import aggregate_seed_metrics, deployment_gate_from_metrics, exact_law_prediction_error, selection_record
from object_centric_best_of_n.object_model import ObjectCentricFutureGenerator
from object_centric_best_of_n.plotting import write_all_figures
from object_centric_best_of_n.selection import SELECTORS
from object_centric_best_of_n.theory import law_validation_row


SCENARIOS = ["good", "swap", "merge_split", "occlusion", "hidden_property", "raw"]
SELECTOR_ORDER = ["raw", "identity_consistent", "property_calibrated", "targeted_probe", "combined_repair", "random", "oracle"]


def _parse_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run(root: Path, mode: str, ns: list[int], seeds: list[int]) -> dict[str, object]:
    start = time.time()
    results = root / "results"
    tables = results / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, float | int | str]] = []
    law_rows: list[dict[str, float | int | str]] = []
    generator = ObjectCentricFutureGenerator(seed=17)
    trials = 500 if mode == "smoke" else 1800

    for seed in seeds:
        for scenario in SCENARIOS:
            scene = make_scene(
                seed=10_000 + seed,
                occlusion=scenario in {"occlusion", "raw"},
                hidden_property=scenario in {"hidden_property", "raw"},
                crossing=scenario in {"occlusion", "swap", "raw"},
            )
            for n in ns:
                candidates = generator.generate_candidates(
                    scene,
                    n=n,
                    scenario=scenario,
                    seed=seed * 1009 + n * 37 + len(scenario),
                )
                for selector_name in SELECTOR_ORDER:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                    seed_rows.append(selection_record("A_controlled_object_binding", scenario, selector_name, n, seed, selected, candidates))
                if scenario == "raw":
                    law_rows.append(
                        {
                            "scenario": scenario,
                            **law_validation_row(
                                [candidate.real_utility for candidate in candidates],
                                [candidate.score for candidate in candidates],
                                n=n,
                                trials=trials,
                                seed=seed + 42,
                            ),
                        }
                    )

    seed_df = pd.DataFrame(seed_rows)
    main = aggregate_seed_metrics(seed_df)
    repair_metrics = main[main["selector"].isin(["raw", "identity_consistent", "property_calibrated", "targeted_probe", "combined_repair", "random", "oracle"])].copy()
    law_df = pd.DataFrame(law_rows)

    seed_df.to_csv(tables / "seed_metrics.csv", index=False)
    main.to_csv(tables / "main_metrics.csv", index=False)
    repair_metrics.to_csv(tables / "repair_metrics.csv", index=False)
    law_df.to_csv(tables / "exact_law_validation.csv", index=False)

    learned_metrics, _ = train_and_evaluate(results, seed=123 if mode == "smoke" else 456)
    learned_row = learned_metrics.as_dict()
    pd.DataFrame([learned_row]).to_csv(tables / "learned_metrics.csv", index=False)

    write_all_figures(main, seed_df, law_df, figures)
    claims = write_claim_status(root)
    gate = deployment_gate_from_metrics(main)
    summary = {
        "mode": mode,
        "ns": ns,
        "seeds": seeds,
        "n_seed_rows": int(seed_df.shape[0]),
        "n_main_rows": int(main.shape[0]),
        "deployment_gate": gate,
        "exact_law_mean_absolute_error": exact_law_prediction_error(law_df),
        "learned_metrics": learned_row,
        "passes_claim_audit": claims["passes_claim_audit"],
        "runtime_seconds": round(time.time() - start, 3),
    }
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_final_audit(root, command_results={f"experiments --mode {mode}": "pass"})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--root", default=Path.cwd())
    parser.add_argument("--ns", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    default_ns = [1, 4, 16] if args.mode == "smoke" else [1, 2, 4, 8, 16, 32, 64]
    default_seeds = [0, 1] if args.mode == "smoke" else list(range(8))
    summary = run(Path(args.root), args.mode, _parse_ints(args.ns, default_ns), _parse_ints(args.seeds, default_seeds))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
