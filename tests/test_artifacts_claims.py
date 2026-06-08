import json
from pathlib import Path

from object_centric_best_of_n.audit import (
    claim_inventory,
    scan_forbidden_overclaims,
    scan_text_overclaims,
    verify_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_required_docs_paper_and_readme_exist():
    required = [
        "README.md",
        "docs/theory.md",
        "docs/claims.md",
        "docs/differentiation_from_best_of_n_wam.md",
        "docs/differentiation_from_prior_projects.md",
        "docs/reviewer_attacks.md",
        "docs/final_audit.md",
        "paper/abstract.md",
        "paper/intro.md",
        "paper/method.md",
        "paper/theory.md",
        "paper/experiments.md",
        "paper/related_work.md",
        "paper/limitations.md",
        "paper/checklist.md",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    assert not missing


def test_generated_artifacts_exist_after_smoke_or_full_run():
    required = [
        "results/tables/main_metrics.csv",
        "results/tables/seed_metrics.csv",
        "results/tables/learned_metrics.csv",
        "results/tables/learned_learning_curve.csv",
        "results/tables/repair_metrics.csv",
        "results/tables/paired_effects.csv",
        "results/tables/repair_ablation.csv",
        "results/tables/observable_repair_metrics.csv",
        "results/tables/exact_law_validation.csv",
        "results/tables/stress_seed_metrics.csv",
        "results/tables/stress_metrics.csv",
        "results/tables/seed_block_robustness.csv",
        "results/tables/score_calibration_candidates.csv",
        "results/tables/score_calibration.csv",
        "results/tables/sensitivity_seed_metrics.csv",
        "results/tables/sensitivity_metrics.csv",
        "results/tables/negative_control.csv",
        "results/tables/learned_ablation.csv",
        "results/tables/ood_seed_metrics.csv",
        "results/tables/ood_metrics.csv",
        "results/tables/model_family_proxy_seed_metrics.csv",
        "results/tables/model_family_proxy_metrics.csv",
        "results/tables/statistical_audit.csv",
        "results/run_summary.json",
        "results/learned_object_model_summary.json",
        "results/verification_log.json",
        "results/artifact_manifest.json",
        "results/claims_status.md",
        "results/claims_status.json",
        "docs/results_digest.md",
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
        "figures/figure12_negative_control.png",
        "figures/figure13_learned_ablation.png",
        "figures/figure14_ood_object_count_stress.png",
        "figures/figure15_model_family_proxies.png",
        "figures/figure16_statistical_audit.png",
        "figures/figure17_observable_repair.png",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    assert not missing


def test_claim_audit_keeps_forbidden_claims_unsupported():
    assert scan_forbidden_overclaims(claim_inventory()) == []
    assert scan_text_overclaims(ROOT) == []
    verification = verify_artifacts(ROOT)
    assert verification["passes"], verification["problems"]
    status_path = ROOT / "results" / "claims_status.json"
    if status_path.exists():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        core = {claim["id"]: claim["status"] for claim in payload["claims"] if claim["id"] in {"C1", "C2", "C3", "C4"}}
        assert all(status == "strongly_supported" for status in core.values())
        unsupported = {claim["claim"]: claim["status"] for claim in payload["claims"] if "real robot" in claim["claim"].lower() or "benchmark" in claim["claim"].lower()}
        assert unsupported
        assert all(status == "unsupported" for status in unsupported.values())
