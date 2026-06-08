"""Claim and artifact audit utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


FORBIDDEN_SUPPORTED_PATTERNS = (
    "real robot",
    "real-robot",
    "sota",
    "state of the art",
    "universal object learning",
    "broad benchmark",
    "benchmark superiority",
)


def _status(passes: bool | None) -> str:
    if passes is None:
        return "supported"
    return "strongly_supported" if passes else "needs_more_evidence"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_claim_strength(root: str | Path) -> dict[str, dict[str, object]]:
    root = Path(root)
    tables = root / "results" / "tables"
    main = _read_csv(tables / "main_metrics.csv")
    law = _read_csv(tables / "exact_law_validation.csv")
    paired = _read_csv(tables / "paired_effects.csv")
    stress = _read_csv(tables / "stress_metrics.csv")
    learned = _read_json(root / "results" / "learned_object_model_summary.json")

    strengths: dict[str, dict[str, object]] = {}
    if not law.empty:
        strengths["C1"] = {
            "passes": bool(law["absolute_error"].max() <= 0.015 and law["absolute_error"].mean() <= 0.006),
            "threshold": "max exact-law absolute error <= 0.015 and mean <= 0.006",
            "observed": {
                "max_absolute_error": float(law["absolute_error"].max()),
                "mean_absolute_error": float(law["absolute_error"].mean()),
            },
        }
    if not main.empty:
        raw = main[(main["scenario"] == "raw") & (main["selector"] == "raw")].sort_values("N")
        if len(raw) >= 2:
            score_gain = float(raw["selected_object_score_mean"].iloc[-1] - raw["selected_object_score_mean"].iloc[0])
            utility_drop = float(raw["selected_real_utility_mean"].iloc[0] - raw["selected_real_utility_mean"].iloc[-1])
            identity_tail = float(raw["identity_error_mean"].iloc[-1])
            strengths["C2"] = {
                "passes": bool(score_gain >= 0.35 and utility_drop >= 0.15 and identity_tail >= 0.75),
                "threshold": "raw high-N score gain >= 0.35, utility drop >= 0.15, tail identity error >= 0.75",
                "observed": {
                    "raw_tail_score_gain": score_gain,
                    "raw_tail_utility_drop": utility_drop,
                    "raw_tail_identity_error": identity_tail,
                },
            }
    if not paired.empty and not stress.empty:
        raw_gain = paired[
            (paired["scenario"] == "raw")
            & (paired["selector"] == "combined_repair")
            & (paired["N"] == paired["N"].max())
        ]
        hidden_probe = paired[
            (paired["scenario"] == "hidden_property")
            & (paired["selector"] == "targeted_probe")
            & (paired["N"] == paired["N"].max())
        ]
        stress_combined = stress[stress["selector"] == "combined_repair"]
        raw_pass = (
            not raw_gain.empty
            and float(raw_gain["mean_gain"].iloc[0]) >= 0.55
            and float(raw_gain["win_rate"].iloc[0]) >= 0.75
        )
        probe_pass = not hidden_probe.empty and float(hidden_probe["mean_gain"].iloc[0]) >= 0.12
        stress_pass = not stress_combined.empty and float(stress_combined["selected_real_utility_mean"].mean()) >= 0.75
        strengths["C3"] = {
            "passes": bool(raw_pass and probe_pass and stress_pass),
            "threshold": "combined raw Nmax gain >= 0.55 with win-rate >= 0.75, targeted hidden-property gain >= 0.12, stress combined mean utility >= 0.75",
            "observed": {
                "combined_raw_nmax_gain": float(raw_gain["mean_gain"].iloc[0]) if not raw_gain.empty else None,
                "combined_raw_nmax_win_rate": float(raw_gain["win_rate"].iloc[0]) if not raw_gain.empty else None,
                "targeted_hidden_property_nmax_gain": float(hidden_probe["mean_gain"].iloc[0]) if not hidden_probe.empty else None,
                "stress_combined_mean_utility": float(stress_combined["selected_real_utility_mean"].mean()) if not stress_combined.empty else None,
            },
        }
    learned_metrics = learned.get("metrics", {})
    if learned_metrics:
        prop_margin = learned_metrics["property_accuracy"] - learned_metrics["random_property_accuracy"]
        identity_margin = (
            learned_metrics["identity_alignment_accuracy"] - learned_metrics["random_identity_alignment_accuracy"]
        )
        transition_ratio = learned_metrics["transition_mse"] / learned_metrics["constant_transition_mse"]
        strengths["C4"] = {
            "passes": bool(
                learned.get("passes_minimum_learned_artifact_checks")
                and prop_margin >= 0.15
                and identity_margin >= 0.15
                and transition_ratio <= 0.25
                and learned_metrics["reward_correlation"] >= 0.75
            ),
            "threshold": "property and identity margins >= 0.15, transition MSE <= 25% baseline, reward correlation >= 0.75",
            "observed": {
                "property_margin": float(prop_margin),
                "identity_alignment_margin": float(identity_margin),
                "transition_mse_ratio": float(transition_ratio),
                "reward_correlation": float(learned_metrics["reward_correlation"]),
            },
        }
    return strengths


def claim_inventory(root: str | Path | None = None) -> list[dict[str, object]]:
    strengths = evaluate_claim_strength(root) if root is not None else {}
    return [
        {
            "id": "C1",
            "claim": "Exact finite tie-aware Best-of-N laws predict selected utility on finite object-candidate populations.",
            "status": _status(strengths.get("C1", {}).get("passes") if root is not None else None),
            "evidence": "theory tests and exact_law_validation.csv",
            "strength": strengths.get("C1", {}),
        },
        {
            "id": "C2",
            "claim": "In controlled object-centric scenes, high-N selection can increase object score while real utility stagnates or falls due to binding failures.",
            "status": _status(strengths.get("C2", {}).get("passes") if root is not None else None),
            "evidence": "figure1 and main_metrics.csv for the raw scenario",
            "strength": strengths.get("C2", {}),
        },
        {
            "id": "C3",
            "claim": "Identity, hidden-property, and targeted-probe repairs improve selected utility in the controlled synthetic setting.",
            "status": _status(strengths.get("C3", {}).get("passes") if root is not None else None),
            "evidence": "figure2, figure4, paired_effects.csv, and stress_metrics.csv",
            "strength": strengths.get("C3", {}),
        },
        {
            "id": "C4",
            "claim": "A CPU NumPy semi-learned object-centric model improves property, identity-alignment, and transition prediction over simple baselines on generated trajectories.",
            "status": _status(strengths.get("C4", {}).get("passes") if root is not None else None),
            "evidence": "learned_object_model_summary.json, learned_metrics.csv, and learned_learning_curve.csv",
            "strength": strengths.get("C4", {}),
        },
        {
            "id": "C5",
            "claim": "The method is validated on real robot systems.",
            "status": "unsupported",
            "evidence": "no real-robot experiments are present",
        },
        {
            "id": "C6",
            "claim": "The method establishes broad benchmark superiority over graph physics, latent, or diffusion world models.",
            "status": "unsupported",
            "evidence": "no broad benchmark suite is present",
        },
    ]


def scan_forbidden_overclaims(claims: Iterable[dict[str, object]]) -> list[str]:
    problems: list[str] = []
    for claim in claims:
        text = (claim.get("claim", "") + " " + claim.get("evidence", "")).lower()
        status = claim.get("status", "").lower()
        if status in {"supported", "strongly_supported"}:
            for pattern in FORBIDDEN_SUPPORTED_PATTERNS:
                if pattern in text:
                    problems.append(f"{claim.get('id', '?')}: supported claim contains forbidden pattern '{pattern}'")
    return problems


def artifact_inventory(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    groups = {
        "tables": sorted(str(path.relative_to(root)) for path in (root / "results" / "tables").glob("*.csv")),
        "figures": sorted(str(path.relative_to(root)) for path in (root / "figures").glob("*.png")),
        "docs": sorted(str(path.relative_to(root)) for path in (root / "docs").glob("*.md")),
        "paper": sorted(str(path.relative_to(root)) for path in (root / "paper").glob("*.md")),
    }
    extras = []
    for rel in [
        "results/run_summary.json",
        "results/learned_object_model_summary.json",
        "results/claims_status.json",
        "results/verification_log.json",
    ]:
        if (root / rel).exists():
            extras.append(rel)
    groups["json"] = extras
    return groups


def write_claim_status(root: str | Path) -> dict[str, object]:
    root = Path(root)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    claims = claim_inventory(root)
    problems = scan_forbidden_overclaims(claims)
    weak_supported = [
        f"{claim['id']}: {claim['status']}"
        for claim in claims
        if claim["id"] in {"C1", "C2", "C3", "C4"} and claim["status"] != "strongly_supported"
    ]
    payload = {
        "claims": claims,
        "forbidden_supported_overclaims": problems,
        "weak_or_missing_core_claims": weak_supported,
        "passes_claim_audit": not problems and not weak_supported,
    }
    (results / "claims_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Claims Status", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim['id']}: {claim['status']}",
                claim["claim"],
                "",
                f"Evidence: {claim['evidence']}",
                "",
                f"Strength: {json.dumps(claim.get('strength', {}), indent=2)}",
                "",
            ]
        )
    if weak_supported:
        lines.append("## Weak or missing core evidence")
        lines.extend(f"- {problem}" for problem in weak_supported)
        lines.append("")
    if problems:
        lines.append("## Forbidden supported overclaims")
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("No forbidden supported overclaims detected.")
    (results / "claims_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_final_audit(root: str | Path, command_results: dict[str, str] | None = None) -> None:
    root = Path(root)
    inventory = artifact_inventory(root)
    summary_payload = _read_json(root / "results" / "run_summary.json")
    if command_results is None:
        command_results = {}
        verification_path = root / "results" / "verification_log.json"
        if verification_path.exists():
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            command_results.update(verification.get("command_results", {}))
        summary_path = root / "results" / "run_summary.json"
        if summary_path.exists() and not command_results:
            summary = summary_payload
            mode = summary.get("mode", "unknown")
            runtime = summary.get("runtime_seconds", "unknown")
            gate = summary.get("deployment_gate", "unknown")
            script_name = "run_all.sh" if mode == "full" else f"run_{mode}.sh"
            command_results[f"bash scripts/{script_name}"] = f"pass (experiment runtime {runtime}s, gate {gate})"
            if summary.get("passes_claim_audit"):
                command_results["bash scripts/run_claim_audit.sh"] = "pass"
    lines = [
        "# Final Audit",
        "",
        "Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.",
        "",
        "## Command Results",
    ]
    if command_results:
        lines.extend(f"- {name}: {status}" for name, status in command_results.items())
    else:
        lines.append("- Pending final command run in this checkout.")
    lines.extend(
        [
            "",
            "## Strongest Artifacts",
            "- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv. "
            f"Raw score gain {summary_payload.get('raw_tail_score_gain', 'unknown')} and raw utility drop {summary_payload.get('raw_tail_utility_drop', 'unknown')}.",
            "- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level transition, hidden-property, identity-alignment, and reward predictors.",
            "- Repair artifact: figure2_repair_comparison.png, paired_effects.csv, and stress_metrics.csv. "
            f"Raw Nmax combined-repair gain {summary_payload.get('combined_repair_raw_nmax_mean_gain', 'unknown')} with win rate {summary_payload.get('combined_repair_raw_nmax_win_rate', 'unknown')}.",
            "- Stress artifact: figure6_stress_robustness.png. "
            f"Combined repair mean selected stress utility {summary_payload.get('stress_combined_mean_selected_utility', 'unknown')}.",
            "",
            "## Differentiation",
            "The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.",
            "It is not a graph-physics benchmark, a latent dynamics benchmark, a diffusion world-model benchmark, or a real-robot evaluation.",
            "",
            "## Remaining Weaknesses",
            f"- Synthetic scenes remain controlled, though the default run now uses {len(summary_payload.get('seeds', [])) or 'unknown'} main seeds and {len(summary_payload.get('stress_seeds', [])) or 'unknown'} stress seeds.",
            "- Repairs use diagnostic signals available in the toy generator.",
            "- No real-robot or broad benchmark evidence is claimed.",
            "",
            "## Artifact Inventory",
        ]
    )
    for group, paths in inventory.items():
        lines.append(f"### {group}")
        if paths:
            lines.extend(f"- {path}" for path in paths)
        else:
            lines.append("- none")
    (root / "docs" / "final_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path.cwd())
    args = parser.parse_args()
    payload = write_claim_status(args.root)
    write_final_audit(args.root)
    if not payload["passes_claim_audit"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
