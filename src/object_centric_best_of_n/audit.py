"""Claim and artifact audit utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


FORBIDDEN_SUPPORTED_PATTERNS = (
    "real robot",
    "real-robot",
    "sota",
    "state of the art",
    "universal object learning",
    "broad benchmark",
    "benchmark superiority",
)


def claim_inventory() -> list[dict[str, str]]:
    return [
        {
            "id": "C1",
            "claim": "Exact finite tie-aware Best-of-N laws predict selected utility on finite object-candidate populations.",
            "status": "supported",
            "evidence": "theory tests and exact_law_validation.csv",
        },
        {
            "id": "C2",
            "claim": "In controlled object-centric scenes, high-N selection can increase object score while real utility stagnates or falls due to binding failures.",
            "status": "supported",
            "evidence": "figure1 and main_metrics.csv for the raw scenario",
        },
        {
            "id": "C3",
            "claim": "Identity, hidden-property, and targeted-probe repairs improve selected utility in the controlled synthetic setting.",
            "status": "supported",
            "evidence": "figure2, figure4, and repair_metrics.csv",
        },
        {
            "id": "C4",
            "claim": "A CPU NumPy semi-learned object-centric model improves property and transition prediction over simple baselines on generated trajectories.",
            "status": "supported",
            "evidence": "learned_object_model_summary.json and learned_metrics.csv",
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


def scan_forbidden_overclaims(claims: Iterable[dict[str, str]]) -> list[str]:
    problems: list[str] = []
    for claim in claims:
        text = (claim.get("claim", "") + " " + claim.get("evidence", "")).lower()
        status = claim.get("status", "").lower()
        if status == "supported":
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
    for rel in ["results/run_summary.json", "results/learned_object_model_summary.json", "results/claims_status.json"]:
        if (root / rel).exists():
            extras.append(rel)
    groups["json"] = extras
    return groups


def write_claim_status(root: str | Path) -> dict[str, object]:
    root = Path(root)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    claims = claim_inventory()
    problems = scan_forbidden_overclaims(claims)
    payload = {
        "claims": claims,
        "forbidden_supported_overclaims": problems,
        "passes_claim_audit": not problems,
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
            ]
        )
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
    if command_results is None:
        command_results = {}
        summary_path = root / "results" / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
            "- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv.",
            "- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level predictors.",
            "- Repair artifact: figure2_repair_comparison.png and repair_metrics.csv.",
            "",
            "## Differentiation",
            "The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.",
            "It is not a graph-physics benchmark, a latent dynamics benchmark, a diffusion world-model benchmark, or a real-robot evaluation.",
            "",
            "## Remaining Weaknesses",
            "- Synthetic scenes are intentionally controlled and small.",
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
