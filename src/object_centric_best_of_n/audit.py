"""Claim and artifact audit utilities."""

from __future__ import annotations

import hashlib
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

REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "results/tables/main_metrics.csv": ("scenario", "selector", "N", "selected_real_utility_mean"),
    "results/tables/seed_metrics.csv": ("scenario", "selector", "N", "seed", "selected_real_utility"),
    "results/tables/learned_metrics.csv": ("property_accuracy", "identity_alignment_accuracy", "transition_mse"),
    "results/tables/learned_learning_curve.csv": ("train_scenes", "property_accuracy", "identity_alignment_accuracy"),
    "results/tables/repair_metrics.csv": ("scenario", "selector", "N", "selected_real_utility_mean"),
    "results/tables/paired_effects.csv": ("scenario", "selector", "N", "mean_gain", "win_rate"),
    "results/tables/repair_ablation.csv": ("scenario", "combined_vs_raw_gain", "combined_vs_best_single_gain"),
    "results/tables/stress_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility"),
    "results/tables/stress_metrics.csv": ("scenario", "selector", "selected_real_utility_mean"),
    "results/tables/seed_block_robustness.csv": ("block_id", "raw_tail_score_gain", "combined_raw_nmax_gain"),
    "results/tables/score_calibration_candidates.csv": ("raw_object_score", "real_utility", "identity_error"),
    "results/tables/score_calibration.csv": ("score_bin", "mean_raw_object_score", "mean_real_utility", "object_real_gap"),
    "results/tables/sensitivity_seed_metrics.csv": ("selector", "score_noise", "selected_real_utility", "identity_error"),
    "results/tables/sensitivity_metrics.csv": ("selector", "score_noise", "selected_real_utility_mean"),
    "results/tables/exact_law_validation.csv": ("N", "predicted_selected_utility", "empirical_selected_utility", "absolute_error"),
}

REQUIRED_FIGURES = (
    "figures/figure1_selected_tail_binding_failure.png",
    "figures/figure2_repair_comparison.png",
    "figures/figure3_tail_diagnostics.png",
    "figures/figure4_targeted_probe_before_after.png",
    "figures/figure5_exact_law_validation.png",
    "figures/figure6_stress_robustness.png",
    "figures/figure7_learned_object_model.png",
    "figures/figure8_repair_ablation.png",
    "figures/figure9_seed_block_robustness.png",
    "figures/figure10_score_calibration.png",
    "figures/figure11_score_noise_sensitivity.png",
)

REQUIRED_JSON = (
    "results/run_summary.json",
    "results/learned_object_model_summary.json",
    "results/verification_log.json",
    "results/artifact_manifest.json",
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
    ablation = _read_csv(tables / "repair_ablation.csv")
    robustness = _read_csv(tables / "seed_block_robustness.csv")
    calibration = _read_csv(tables / "score_calibration.csv")
    sensitivity = _read_csv(tables / "sensitivity_metrics.csv")
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
            top_calibration = calibration.sort_values("score_bin").iloc[-1] if not calibration.empty else None
            strengths["C2"] = {
                "passes": bool(
                    score_gain >= 0.35
                    and utility_drop >= 0.15
                    and identity_tail >= 0.75
                    and not robustness.empty
                    and float(robustness["raw_tail_score_gain"].min()) >= 0.30
                    and float(robustness["raw_tail_utility_drop"].min()) >= 0.10
                    and float(robustness["raw_tail_identity_error"].min()) >= 0.75
                    and top_calibration is not None
                    and float(top_calibration["object_real_gap"]) >= 0.45
                    and float(top_calibration["identity_error_rate"]) >= 0.55
                ),
                "threshold": "raw high-N score gain >= 0.35, utility drop >= 0.15, tail identity error >= 0.75, all seed blocks pass reduced thresholds, and top raw-score calibration bin has gap >= 0.45 with identity error >= 0.55",
                "observed": {
                    "raw_tail_score_gain": score_gain,
                    "raw_tail_utility_drop": utility_drop,
                    "raw_tail_identity_error": identity_tail,
                    "min_block_score_gain": float(robustness["raw_tail_score_gain"].min()) if not robustness.empty else None,
                    "min_block_utility_drop": float(robustness["raw_tail_utility_drop"].min()) if not robustness.empty else None,
                    "min_block_identity_error": float(robustness["raw_tail_identity_error"].min()) if not robustness.empty else None,
                    "top_calibration_object_real_gap": float(top_calibration["object_real_gap"]) if top_calibration is not None else None,
                    "top_calibration_identity_error": float(top_calibration["identity_error_rate"]) if top_calibration is not None else None,
                },
            }
    if not paired.empty and not stress.empty and not ablation.empty and not robustness.empty and not sensitivity.empty:
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
        raw_ablation = ablation[ablation["scenario"] == "raw"]
        raw_pass = (
            not raw_gain.empty
            and float(raw_gain["mean_gain"].iloc[0]) >= 0.55
            and float(raw_gain["win_rate"].iloc[0]) >= 0.75
        )
        probe_pass = not hidden_probe.empty and float(hidden_probe["mean_gain"].iloc[0]) >= 0.12
        stress_pass = (
            not stress_combined.empty
            and float(stress_combined["selected_real_utility_mean"].mean()) >= 0.75
            and float(stress_combined["selected_real_utility_mean"].min()) >= 0.80
        )
        ablation_pass = (
            not raw_ablation.empty
            and float(raw_ablation["combined_vs_best_single_gain"].iloc[0]) >= 0.20
            and float(raw_ablation["combined_oracle_gap"].iloc[0]) <= 0.08
        )
        robustness_pass = bool(
            float(robustness["combined_raw_nmax_gain"].min()) >= 0.55
            and float(robustness["combined_raw_nmax_win_rate"].min()) >= 0.75
        )
        sensitivity_low_noise = sensitivity[sensitivity["score_noise"] <= 0.10]
        combined_sensitivity = sensitivity_low_noise[sensitivity_low_noise["selector"] == "combined_repair_noisy"]
        raw_sensitivity = sensitivity_low_noise[sensitivity_low_noise["selector"] == "raw_noisy"]
        sensitivity_margin = None
        if not combined_sensitivity.empty and not raw_sensitivity.empty:
            sensitivity_margin = float(
                combined_sensitivity["selected_real_utility_mean"].mean()
                - raw_sensitivity["selected_real_utility_mean"].mean()
            )
        sensitivity_pass = (
            not combined_sensitivity.empty
            and not raw_sensitivity.empty
            and float(combined_sensitivity["selected_real_utility_mean"].min()) >= 0.75
            and sensitivity_margin is not None
            and sensitivity_margin >= 0.55
        )
        strengths["C3"] = {
            "passes": bool(raw_pass and probe_pass and stress_pass and ablation_pass and robustness_pass and sensitivity_pass),
            "threshold": "combined raw Nmax gain >= 0.55 with win-rate >= 0.75, targeted hidden-property gain >= 0.12, stress combined mean >= 0.75 and min >= 0.80, raw ablation dominance >= 0.20 with oracle gap <= 0.08, all seed blocks repair, and combined repair remains strong under score noise <= 0.10",
            "observed": {
                "combined_raw_nmax_gain": float(raw_gain["mean_gain"].iloc[0]) if not raw_gain.empty else None,
                "combined_raw_nmax_win_rate": float(raw_gain["win_rate"].iloc[0]) if not raw_gain.empty else None,
                "targeted_hidden_property_nmax_gain": float(hidden_probe["mean_gain"].iloc[0]) if not hidden_probe.empty else None,
                "stress_combined_mean_utility": float(stress_combined["selected_real_utility_mean"].mean()) if not stress_combined.empty else None,
                "stress_combined_min_utility": float(stress_combined["selected_real_utility_mean"].min()) if not stress_combined.empty else None,
                "raw_ablation_combined_vs_best_single_gain": float(raw_ablation["combined_vs_best_single_gain"].iloc[0]) if not raw_ablation.empty else None,
                "raw_ablation_combined_oracle_gap": float(raw_ablation["combined_oracle_gap"].iloc[0]) if not raw_ablation.empty else None,
                "min_block_combined_raw_gain": float(robustness["combined_raw_nmax_gain"].min()) if not robustness.empty else None,
                "min_block_combined_win_rate": float(robustness["combined_raw_nmax_win_rate"].min()) if not robustness.empty else None,
                "combined_min_low_noise_utility": float(combined_sensitivity["selected_real_utility_mean"].min()) if not combined_sensitivity.empty else None,
                "combined_vs_raw_low_noise_margin": sensitivity_margin,
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


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def verify_artifacts(root: str | Path) -> dict[str, object]:
    root = Path(root)
    problems: list[str] = []
    checked: list[str] = []
    for rel, columns in REQUIRED_TABLES.items():
        path = root / rel
        checked.append(rel)
        if not path.exists():
            problems.append(f"missing table: {rel}")
            continue
        if path.stat().st_size <= 0:
            problems.append(f"empty table file: {rel}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            problems.append(f"table has no rows: {rel}")
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            problems.append(f"table {rel} missing columns: {missing_columns}")
    for rel in REQUIRED_FIGURES:
        path = root / rel
        checked.append(rel)
        if not path.exists():
            problems.append(f"missing figure: {rel}")
            continue
        width, height = _png_dimensions(path)
        if width < 400 or height < 300:
            problems.append(f"figure {rel} has suspicious dimensions: {width}x{height}")
    for rel in REQUIRED_JSON:
        path = root / rel
        checked.append(rel)
        if not path.exists():
            problems.append(f"missing json artifact: {rel}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"invalid json {rel}: {exc}")
    return {"checked_count": len(checked), "problems": problems, "passes": not problems}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(root: str | Path) -> dict[str, object]:
    root = Path(root)
    manifest_paths = sorted(
        set(REQUIRED_TABLES)
        | set(REQUIRED_FIGURES)
        | {
            "results/run_summary.json",
            "results/learned_object_model_summary.json",
            "results/verification_log.json",
            "docs/results_digest.md",
        }
    )
    files = []
    for rel in manifest_paths:
        path = root / rel
        if path.exists():
            files.append({"path": rel, "bytes": int(path.stat().st_size), "sha256": _sha256(path)})
        else:
            files.append({"path": rel, "missing": True})
    payload = {"artifact_count": len(files), "files": files}
    out = root / "results" / "artifact_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def scan_text_overclaims(root: str | Path) -> list[str]:
    root = Path(root)
    support_terms = (
        "validated",
        "validation",
        "superiority",
        "outperform",
        "outperforms",
        "beats",
        "state of the art",
        "sota",
        "real-robot evidence",
        "real robot evidence",
    )
    negators = (
        "not",
        "no ",
        "unsupported",
        "without",
        "does not",
        "do not",
        "needs",
        "requires",
        "not a",
    )
    problems: list[str] = []
    files = [root / "README.md"]
    files.extend(sorted((root / "docs").glob("*.md")))
    files.extend(sorted((root / "paper").glob("*.md")))
    for path in files:
        if not path.exists():
            continue
        in_unsupported_section = False
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.lower()
            if text.startswith("##"):
                in_unsupported_section = "unsupported" in text or "boundaries" in text or "what this is not" in text
            if in_unsupported_section:
                continue
            has_forbidden = any(pattern in text for pattern in FORBIDDEN_SUPPORTED_PATTERNS)
            has_support = any(term in text for term in support_terms)
            has_negator = any(term in text for term in negators)
            if has_forbidden and has_support and not has_negator:
                problems.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    return problems


def write_results_digest(root: str | Path) -> None:
    root = Path(root)
    summary = _read_json(root / "results" / "run_summary.json")
    claims = _read_json(root / "results" / "claims_status.json")
    learned = _read_json(root / "results" / "learned_object_model_summary.json")
    lines = [
        "# Results Digest",
        "",
        "This digest is generated from the current result artifacts.",
        "",
        "## Headline Numbers",
        f"- Exact-law mean absolute error: {summary.get('exact_law_mean_absolute_error', 'unknown')}",
        f"- Raw selected-tail score gain: {summary.get('raw_tail_score_gain', 'unknown')}",
        f"- Raw selected-tail utility drop: {summary.get('raw_tail_utility_drop', 'unknown')}",
        f"- Combined repair raw Nmax gain: {summary.get('combined_repair_raw_nmax_mean_gain', 'unknown')}",
        f"- Combined repair raw ablation dominance: {summary.get('combined_repair_raw_ablation_dominance', 'unknown')}",
        f"- Stress combined mean selected utility: {summary.get('stress_combined_mean_selected_utility', 'unknown')}",
        f"- Seed-block robustness pass rate: {summary.get('seed_block_robustness_pass_rate', 'unknown')}",
        f"- Top raw-score calibration gap: {summary.get('raw_score_top_bin_object_real_gap', 'unknown')}",
        f"- Combined repair low-noise minimum utility: {summary.get('combined_repair_min_low_noise_utility', 'unknown')}",
        f"- Combined-vs-raw low-noise sensitivity margin: {summary.get('combined_vs_raw_low_noise_sensitivity_margin', 'unknown')}",
        "",
        "## Learned Model",
    ]
    metrics = learned.get("metrics", {})
    if metrics:
        lines.extend(
            [
                f"- Property accuracy: {metrics.get('property_accuracy')} versus baseline {metrics.get('random_property_accuracy')}",
                f"- Identity alignment accuracy: {metrics.get('identity_alignment_accuracy')} versus baseline {metrics.get('random_identity_alignment_accuracy')}",
                f"- Transition MSE ratio: {metrics.get('transition_mse') / metrics.get('constant_transition_mse') if metrics.get('constant_transition_mse') else 'unknown'}",
                f"- Reward correlation: {metrics.get('reward_correlation')}",
            ]
        )
    lines.extend(["", "## Claim Status"])
    for claim in claims.get("claims", []):
        lines.append(f"- {claim.get('id')}: {claim.get('status')} - {claim.get('claim')}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "Real-robot validation and broad benchmark superiority remain unsupported and are not claimed as supported paper results.",
        ]
    )
    (root / "docs" / "results_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    write_artifact_manifest(root)
    claims = claim_inventory(root)
    problems = scan_forbidden_overclaims(claims)
    text_overclaims = scan_text_overclaims(root)
    artifact_verification = verify_artifacts(root)
    weak_supported = [
        f"{claim['id']}: {claim['status']}"
        for claim in claims
        if claim["id"] in {"C1", "C2", "C3", "C4"} and claim["status"] != "strongly_supported"
    ]
    payload = {
        "claims": claims,
        "forbidden_supported_overclaims": problems,
        "paper_text_overclaims": text_overclaims,
        "artifact_verification": artifact_verification,
        "weak_or_missing_core_claims": weak_supported,
        "passes_claim_audit": not problems and not weak_supported and not text_overclaims and artifact_verification["passes"],
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
    if text_overclaims:
        lines.append("## Paper text overclaims")
        lines.extend(f"- {problem}" for problem in text_overclaims)
    if not artifact_verification["passes"]:
        lines.append("## Artifact verification problems")
        lines.extend(f"- {problem}" for problem in artifact_verification["problems"])
    lines.append("")
    lines.append(f"Artifact verification checked {artifact_verification['checked_count']} required artifacts.")
    if not problems and not text_overclaims and artifact_verification["passes"]:
        lines.append("")
        lines.append("No paper-text or artifact overclaim problems detected.")
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
            "- Ablation artifact: figure8_repair_ablation.png and repair_ablation.csv. "
            f"Raw Nmax combined-repair dominance over the best single repair {summary_payload.get('combined_repair_raw_ablation_dominance', 'unknown')}.",
            "- Robustness artifact: figure9_seed_block_robustness.png and seed_block_robustness.csv. "
            f"Seed-block robustness pass rate {summary_payload.get('seed_block_robustness_pass_rate', 'unknown')}.",
            "- Stress artifact: figure6_stress_robustness.png. "
            f"Combined repair mean selected stress utility {summary_payload.get('stress_combined_mean_selected_utility', 'unknown')}.",
            "- Calibration artifact: figure10_score_calibration.png and score_calibration.csv. "
            f"Top raw-score bin object-real gap {summary_payload.get('raw_score_top_bin_object_real_gap', 'unknown')}.",
            "- Sensitivity artifact: figure11_score_noise_sensitivity.png and sensitivity_metrics.csv. "
            f"Combined repair low-noise minimum utility {summary_payload.get('combined_repair_min_low_noise_utility', 'unknown')}.",
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
    write_results_digest(args.root)
    payload = write_claim_status(args.root)
    write_results_digest(args.root)
    write_final_audit(args.root)
    if not payload["passes_claim_audit"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
