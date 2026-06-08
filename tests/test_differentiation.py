from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_distinguish_from_wam_and_graph_physics():
    wam = (ROOT / "docs" / "differentiation_from_best_of_n_wam.md").read_text(encoding="utf-8").lower()
    prior = (ROOT / "docs" / "differentiation_from_prior_projects.md").read_text(encoding="utf-8").lower()
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "finite best-of-n law" in wam
    assert "object-centric" in wam
    assert "not a graph physics" in prior
    assert "graph physics" in readme
